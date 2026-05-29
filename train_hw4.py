import argparse
import math
import os
import random
from pathlib import Path

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
    parser.add_argument("--prompt_len", type=int, default=5, help="Number of prompt components per prompt block.")
    parser.add_argument("--num_expert", type=int, default=1, help="Top-k prompt experts used for sparse routing.")
    parser.add_argument("--use_cpr", action="store_true", help="Enable contrastive prompt regularization (CPR).")
    parser.add_argument("--neg_num", type=int, default=2, help="Number of negative prompt samples per iteration when CPR is enabled.")
    parser.add_argument("--lambda_cpr", type=float, default=0.1, help="Weight of CPR loss term.")
    parser.add_argument("--cpr_margin", type=float, default=0.01, help="Margin used by CPR ranking loss.")
    parser.add_argument("--use_tur", action="store_true", help="Enable task-uncertainty regularization (TUR) on prompt routing.")
    parser.add_argument("--lambda_tur", type=float, default=0.05, help="Weight of TUR loss term.")
    parser.add_argument("--tur_eps", type=float, default=1e-8, help="Numerical epsilon for TUR entropy and normalization.")
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


class UncertaintyEstimationHead(nn.Module):
    def __init__(self, in_channels: int = 96):
        super().__init__()
        hidden = max(32, in_channels // 2)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.body(feat).flatten(1).squeeze(1)


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
    uem_head = UncertaintyEstimationHead(in_channels=96).to(device)
    print("Training from scratch: pretrained weights are NOT used.")

    criterion = nn.L1Loss()
    criterion_per_sample = nn.L1Loss(reduction="none")
    if not args.use_tur:
        for p in uem_head.parameters():
            p.requires_grad = False
    params = list(model.parameters()) + (list(uem_head.parameters()) if args.use_tur else [])
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_psnr = -1.0
    start_epoch = 1
    best_path = os.path.join(args.save_dir, "best.pth")
    last_path = os.path.join(args.save_dir, "last.pth")

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
        if args.use_tur and "uem_head" in ckpt:
            uem_head.load_state_dict(ckpt["uem_head"])

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
        uem_head.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            degraded = batch["degraded"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                restored, aux = model(degraded, is_neg=False, return_aux=True)
                rec_loss = criterion(restored, clean)
                rec_per_sample = criterion_per_sample(restored, clean).mean(dim=(1, 2, 3))

                if args.use_cpr:
                    neg_losses = []
                    with torch.no_grad():
                        for _ in range(args.neg_num):
                            neg_restored = model(degraded, is_neg=True)
                            neg_losses.append(criterion(neg_restored, clean))

                    neg_loss = torch.stack(neg_losses).mean() if len(neg_losses) > 0 else rec_loss.detach()
                    cpr_loss = torch.relu(rec_loss + args.cpr_margin - neg_loss)
                else:
                    cpr_loss = torch.zeros_like(rec_loss)

                base_per_sample = rec_per_sample + args.lambda_cpr * cpr_loss
                base_loss = base_per_sample.mean()

                if args.use_tur:
                    tur_feature = aux.get("tur_feature", None)
                    if tur_feature is None:
                        raise RuntimeError("Missing TUR feature from model aux output.")
                    log_var = uem_head(tur_feature)
                    log_var = torch.clamp(log_var, min=-10.0, max=10.0)
                    loss = compute_uncertainty_weighted_loss(base_per_sample, log_var, reg_weight=args.lambda_tur)
                    tur_loss = loss - base_loss
                    sigma = torch.exp(0.5 * log_var)
                else:
                    loss = base_loss
                    tur_loss = torch.zeros_like(rec_loss)
                    sigma = None

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                rec=f"{rec_loss.item():.4f}",
                cpr=f"{cpr_loss.item():.4f}",
                tur=f"{tur_loss.item():.4f}",
                sigma=(f"{sigma.mean().item():.3f}" if sigma is not None else "-"),
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        scheduler.step()
        train_loss = running_loss / max(1, len(train_loader))

        val_l1, val_psnr = validate(model, val_loader, device)
        print(
            f"[Epoch {epoch:03d}] train_l1={train_loss:.6f} "
            f"val_l1={val_l1:.6f} val_psnr={val_psnr:.3f}"
        )

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
            state["uem_head"] = uem_head.state_dict()

        torch.save(state, last_path)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            state["best_psnr"] = best_psnr
            torch.save(state, best_path)
            print(f"Saved new best checkpoint to {best_path} (PSNR={best_psnr:.3f})")

        if args.save_every > 0 and epoch % args.save_every == 0:
            epoch_path = os.path.join(args.save_dir, f"epoch_{epoch:03d}.pth")
            torch.save(state, epoch_path)

    print(f"Training finished. Best PSNR={best_psnr:.3f}, checkpoint={best_path}")


if __name__ == "__main__":
    main()
