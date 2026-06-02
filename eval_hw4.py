import argparse
import math
import os
from typing import Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from net.model import PromptIR
from utils.hw4_dataset import HW4SubmissionDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Generate hw4 submission pred.npz.")
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/release_folder/hw4_realse_dataset",
        help="Root folder of hw4_release_dataset.",
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pth) from train_hw4.py.")
    parser.add_argument(
        "--output_npz",
        type=str,
        default="data/release_folder/pred.npz",
        help="Output submission file path.",
    )
    parser.add_argument(
        "--save_png_dir",
        type=str,
        default="",
        help="Optional folder to additionally save restored PNG files.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--tta", action="store_true", help="Enable 8-fold Test-Time Augmentation (4 rotations × 2 flips).")
    return parser.parse_args()


# ── TTA helpers ───────────────────────────────────────────────────────────────
# 8 augmentations: rot0/90/180/270 × (no flip / hflip)
_TTA_AUG = [
    # (k_rot, do_hflip)  —  k_rot: number of 90-degree CCW rotations
    (0, False), (0, True),
    (1, False), (1, True),
    (2, False), (2, True),
    (3, False), (3, True),
]


def _tta_aug(x: torch.Tensor, k: int, hflip: bool) -> torch.Tensor:
    """Apply augmentation to a batch tensor (B,C,H,W)."""
    if hflip:
        x = torch.flip(x, dims=[3])
    if k:
        x = torch.rot90(x, k=k, dims=[2, 3])
    return x


def _tta_deaug(x: torch.Tensor, k: int, hflip: bool) -> torch.Tensor:
    """Invert the augmentation on a restored tensor."""
    if k:
        x = torch.rot90(x, k=-k, dims=[2, 3])   # inverse rotation
    if hflip:
        x = torch.flip(x, dims=[3])
    return x


def infer_tta(model, degraded: torch.Tensor) -> torch.Tensor:
    """Run 8-fold TTA and return the averaged output (same shape as input)."""
    acc = None
    for k, hflip in _TTA_AUG:
        aug = _tta_aug(degraded, k, hflip)
        out = model(aug)
        out = _tta_deaug(out, k, hflip)
        acc = out if acc is None else acc + out
    return acc / len(_TTA_AUG)


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


def tensor_to_np_uint8_chw(image: torch.Tensor) -> np.ndarray:
    image = torch.clamp(image, 0.0, 1.0)
    image = (image * 255.0).round().byte().cpu().numpy()
    return image


def save_png_hwc_chw(image_chw: np.ndarray, save_path: str):
    image_hwc = np.transpose(image_chw, (1, 2, 0))
    Image.fromarray(image_hwc).save(save_path)


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    # Read prompt_len / num_expert from the checkpoint's saved args so the model
    # architecture matches exactly what was used during training.
    ckpt_args = ckpt.get("args", {})
    prompt_len = ckpt_args.get("prompt_len", 5)
    num_expert  = ckpt_args.get("num_expert",  1)
    model = PromptIR(decoder=True, prompt_len=prompt_len, num_expert=num_expert).to(device)

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    dataset = HW4SubmissionDataset(root_dir=args.data_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.save_png_dir:
        os.makedirs(args.save_png_dir, exist_ok=True)

    preds: Dict[str, np.ndarray] = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference (TTA)" if args.tta else "Inference"):
            names = batch["name"]
            degraded = batch["degraded"].to(device, non_blocking=True)

            degraded, h, w = pad_to_multiple(degraded, multiple=8)
            if args.tta:
                restored = infer_tta(model, degraded)[:, :, :h, :w]
            else:
                restored = model(degraded)[:, :, :h, :w]

            for i, name in enumerate(names):
                arr = tensor_to_np_uint8_chw(restored[i])
                preds[name] = arr
                if args.save_png_dir:
                    save_path = os.path.join(args.save_png_dir, name)
                    save_png_hwc_chw(arr, save_path)

    os.makedirs(os.path.dirname(args.output_npz) or ".", exist_ok=True)
    np.savez(args.output_npz, **preds)
    print(f"Saved {len(preds)} predictions to {args.output_npz}")


if __name__ == "__main__":
    main()
