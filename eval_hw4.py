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
    return parser.parse_args()


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
    model = PromptIR(decoder=True).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
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
        for batch in tqdm(loader, desc="Inference"):
            names = batch["name"]
            degraded = batch["degraded"].to(device, non_blocking=True)

            degraded, h, w = pad_to_multiple(degraded, multiple=8)
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
