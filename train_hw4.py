import argparse
import math
import os
import random
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from net.model import PromptIR
from utils.hw4_dataset import HW4TrainDataset, HW4ValDataset, list_images_sorted


def parse_args():
    parser = argparse.ArgumentParser(description="Train PromptIR on hw4_release_dataset from scratch.")
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/release_folder/hw4_realse_dataset",
        help="Root folder of hw4_release_dataset.",
    )
    parser.add_argument("--save_dir", type=str, default="ckpt/hw4", help="Checkpoint output folder.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers.")
    parser.add_argument("--patch_size", type=int, default=128, help="Training crop size.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio from training set.")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed.")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training.")
    parser.add_argument("--save_every", type=int, default=20, help="Save periodic checkpoint every N epochs.")
    parser.add_argument("--prompt_len", type=int, default=2, help="Number of prompt components per prompt block. Default 2 for HW4 (rain/snow).")
    parser.add_argument("--num_expert", type=int, default=1, help="Top-k prompt experts used for sparse routing.")
    parser.add_argument("--use_cpr", action="store_true", help="Enable contrastive prompt regularization (CPR).")
    parser.add_argument("--neg_num", type=int, default=1, help="Number of negative prompt samples per iteration when CPR is enabled.")
    parser.add_argument("--lambda_cpr", type=float, default=0.1, help="Weight of CPR loss term.")
    parser.add_argument("--cpr_margin", type=float, default=0.01, help="Margin used by CPR ranking loss.")
    parser.add_argument("--use_tur", action="store_true", help="Enable task-uncertainty regularization (TUR) with per-task learnable sigma.")
    parser.add_argument("--lambda_tur", type=float, default=0.1, help="Weight of TUR regularization term. Should match scale of base loss (~0.1).")
    parser.add_argument("--tur_eps", type=float, default=1e-8, help="Numerical epsilon for TUR normalization.")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Checkpoint path to resume training (e.g., ckpt/hw4/last.pth).",
    )
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        help="Automatically resume from <save_dir>/last.pth if it exists.",
    )
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_train_val(data_root: str, val_ratio: float, seed: int):
    degraded_dir = os.path.join(data_root, "train", "degraded")
    names = list_images_sorted(degraded_dir)
    if len(names) < 10:
        raise RuntimeError("Training set is too small to split into train/val.")

    rng = random.Random(seed)
    rng.shuffle(names)

    val_len = max(1, int(len(names) * val_ratio))
    val_names = sorted(names[:val_len])
    train_names = sorted(names[val_len:])
    return train_names, val_names


def pad_to_multiple(x: torch.Tensor, multiple: int = 8):
    _, _, h, w = x.shape
    new_h = int(math.ceil(h / multiple) * multiple)
    new_w = int(math.ceil(w / multiple) * multiple)
    pad_h = new_h - h
    pad_w = new_w - w
    if pad_h == 0 and pad_w == 0:
        return x, h, w
    x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, h, w


def psnr_batch(restored: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((restored - target) ** 2, dim=(1, 2, 3))
    mse = torch.clamp(mse, min=1e-12)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return psnr.mean().item()


def extract_task_ids(names: list, device: torch.device) -> torch.Tensor:
    """Map batch filenames to task ids: rain-* -> 0, snow-* -> 1."""
    ids = [0 if "rain" in n.lower() else 1 for n in names]
    return torch.tensor(ids, dtype=torch.long, device=device)


def compute_uncertainty_weighted_loss(
    base_per_sample: torch.Tensor,
    log_var: torch.Tensor,
    reg_weight: float = 1.0,
) -> torch.Tensor:
    # TUR objective: 0.5 * exp(-s) * L + 0.5 * lambda * s, where s = log(sigma^2).
    return (0.5 * torch.exp(-log_var) * base_per_sample + 0.5 * reg_weight * log_var).mean()


def validate(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    val_l1 = 0.0
    val_psnr = 0.0
    criterion = nn.L1Loss()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validate", leave=False):
            degraded = batch["degraded"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)

            degraded, h, w = pad_to_multiple(degraded, multiple=8)
            restored = model(degraded)[:, :, :h, :w]

            val_l1 += criterion(restored, clean).item()
            val_psnr += psnr_batch(restored, clean)

    num_batches = max(1, len(dataloader))
    return val_l1 / num_batches, val_psnr / num_batches


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    train_names, val_names = split_train_val(args.data_root, args.val_ratio, args.seed)
    train_set = HW4TrainDataset(
        root_dir=args.data_root,
        patch_size=args.patch_size,
        augment=True,
        file_list=train_names,
    )
    val_set = HW4ValDataset(root_dir=args.data_root, file_list=val_names)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )

    model = PromptIR(
        decoder=True,
        prompt_len=args.prompt_len,
        num_expert=args.num_expert,
    ).to(device)
    # Per-task log-variance: index 0=rain, 1=snow.
    # Task-level sigma avoids the per-sample collapse seen with a UEM conv head.
    task_log_sigma = nn.Parameter(torch.zeros(2, device=device))
    print("Training from scratch: pretrained weights are NOT used.")

    criterion = nn.L1Loss()
    criterion_per_sample = nn.L1Loss(reduction="none")
    params = list(model.parameters()) + ([task_log_sigma] if args.use_tur else [])
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_psnr = -1.0
    start_epoch = 1
    best_path = os.path.join(args.save_dir, "best.pth")
    last_path = os.path.join(args.save_dir, "last.pth")

    tb_dir = os.path.join(args.save_dir, "tb_logs")
    writer = SummaryWriter(log_dir=tb_dir)

    resume_path = ""
    if args.resume:
        resume_path = args.resume
    elif args.auto_resume and os.path.exists(last_path):
        resume_path = last_path

    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)

        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])
        if args.use_tur and "task_log_sigma" in ckpt:
            task_log_sigma.data.copy_(ckpt["task_log_sigma"].to(device))

        best_psnr = ckpt.get("best_psnr", best_psnr)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(
            f"Resumed from {resume_path} at epoch {start_epoch - 1}; "
            f"continue with batch_size={args.batch_size}, patch_size={args.patch_size}."
        )

    if start_epoch > args.epochs:
        print(
            f"Checkpoint epoch ({start_epoch - 1}) is already >= target epochs ({args.epochs}). "
            "No training needed."
        )
        return

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            degraded = batch["degraded"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)
            # task_ids: 0=rain, 1=snow (used by TUR per-task sigma)
            task_ids = extract_task_ids(batch["name"], device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                restored = model(degraded)
                rec_loss = criterion(restored, clean)
                rec_per_sample = criterion_per_sample(restored, clean).mean(dim=(1, 2, 3))

                # CPR: with prompt_len=2 the negative is deterministically the other expert.
                if args.use_cpr:
                    neg_losses = []
                    with torch.no_grad():
                        for _ in range(args.neg_num):
                            neg_restored = model(degraded, is_neg=True)
                            neg_losses.append(criterion(neg_restored, clean))
                    neg_loss = torch.stack(neg_losses).mean() if neg_losses else rec_loss.detach()
                    cpr_loss = torch.relu(rec_loss + args.cpr_margin - neg_loss)
                else:
                    cpr_loss = torch.zeros_like(rec_loss)

                base_per_sample = rec_per_sample + args.lambda_cpr * cpr_loss
                base_loss = base_per_sample.mean()

                # TUR: per-task sigma indexed by task_ids, avoids per-sample collapse.
                if args.use_tur:
                    log_var = torch.clamp(task_log_sigma[task_ids], min=-4.0, max=4.0)
                    loss = compute_uncertainty_weighted_loss(base_per_sample, log_var, reg_weight=args.lambda_tur)
                    tur_loss = loss - base_loss
                    sigma_r = torch.exp(0.5 * task_log_sigma[0].detach())
                    sigma_s = torch.exp(0.5 * task_log_sigma[1].detach())
                else:
                    loss = base_loss
                    tur_loss = torch.zeros_like(rec_loss)
                    sigma_r = sigma_s = None

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            # ---------- TensorBoard per-step (every 50 steps) ----------
            _step = (epoch - 1) * len(train_loader) + pbar.n
            if pbar.n % 50 == 0:
                writer.add_scalar("step/loss", loss.item(), _step)
                writer.add_scalar("step/rec_loss", rec_loss.item(), _step)
                writer.add_scalar("step/cpr_loss", cpr_loss.item(), _step)
                writer.add_scalar("step/tur_loss", tur_loss.item(), _step)
                writer.add_scalar("step/lr", optimizer.param_groups[0]["lr"], _step)
                if sigma_r is not None:
                    writer.add_scalar("step/sigma_rain", sigma_r.item(), _step)
                    writer.add_scalar("step/sigma_snow", sigma_s.item(), _step)
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                rec=f"{rec_loss.item():.4f}",
                cpr=f"{cpr_loss.item():.4f}",
                tur=f"{tur_loss.item():.4f}",
                sr=(f"{sigma_r.item():.3f}" if sigma_r is not None else "-"),
                ss=(f"{sigma_s.item():.3f}" if sigma_s is not None else "-"),
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        scheduler.step()
        train_loss = running_loss / max(1, len(train_loader))

        val_l1, val_psnr = validate(model, val_loader, device)
        print(
            f"[Epoch {epoch:03d}] train_l1={train_loss:.6f} "
            f"val_l1={val_l1:.6f} val_psnr={val_psnr:.3f}"
        )

        # ---------- TensorBoard per-epoch ----------
        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        writer.add_scalar("epoch/val_l1", val_l1, epoch)
        writer.add_scalar("epoch/val_psnr", val_psnr, epoch)
        if args.use_tur:
            writer.add_scalar("epoch/sigma_rain", torch.exp(0.5 * task_log_sigma[0].detach()).item(), epoch)
            writer.add_scalar("epoch/sigma_snow", torch.exp(0.5 * task_log_sigma[1].detach()).item(), epoch)

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_psnr": best_psnr,
            "args": vars(args),
        }
        if args.use_tur:
            state["task_log_sigma"] = task_log_sigma.data.cpu()

        torch.save(state, last_path)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            state["best_psnr"] = best_psnr
            torch.save(state, best_path)
            print(f"Saved new best checkpoint to {best_path} (PSNR={best_psnr:.3f})")

        if args.save_every > 0 and epoch % args.save_every == 0:
            epoch_path = os.path.join(args.save_dir, f"epoch_{epoch:03d}.pth")
            torch.save(state, epoch_path)

    writer.close()
    print(f"Training finished. Best PSNR={best_psnr:.3f}, checkpoint={best_path}")


if __name__ == "__main__":
    main()
