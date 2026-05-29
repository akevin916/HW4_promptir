#!/usr/bin/env python3
import argparse
import csv
import glob
import math
import os
import statistics
from collections import defaultdict

import torch


def infer_group(args_dict):
    use_cpr = bool(args_dict.get("use_cpr", False))
    use_tur = bool(args_dict.get("use_tur", False))
    if use_cpr and use_tur:
        return "rec_cpr_tur"
    if use_cpr:
        return "rec_cpr"
    if use_tur:
        return "rec_tur"
    return "rec"


def safe_float(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default


def summarize(values):
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return float("nan"), float("nan"), 0
    if len(values) == 1:
        return values[0], 0.0, 1
    return statistics.mean(values), statistics.stdev(values), len(values)


def main():
    parser = argparse.ArgumentParser(description="Aggregate loss ablation checkpoints into CSV reports.")
    parser.add_argument("--exp-root", type=str, default="experiments/loss_ablation", help="Root folder of experiments.")
    parser.add_argument("--detail-csv", type=str, default="experiments/loss_ablation/detail.csv", help="Output per-run CSV path.")
    parser.add_argument("--summary-csv", type=str, default="experiments/loss_ablation/summary.csv", help="Output summary CSV path.")
    args = parser.parse_args()

    pattern = os.path.join(args.exp_root, "**", "best.pth")
    ckpt_paths = sorted(glob.glob(pattern, recursive=True))
    if not ckpt_paths:
        raise FileNotFoundError(f"No best.pth found under: {args.exp_root}")

    rows = []
    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        args_dict = ckpt.get("args", {})
        group = infer_group(args_dict)
        row = {
            "group": group,
            "run_dir": os.path.dirname(ckpt_path),
            "ckpt_path": ckpt_path,
            "epoch": int(ckpt.get("epoch", -1)),
            "best_psnr": safe_float(ckpt.get("best_psnr", float("nan"))),
            "seed": int(args_dict.get("seed", -1)),
            "use_cpr": int(bool(args_dict.get("use_cpr", False))),
            "use_tur": int(bool(args_dict.get("use_tur", False))),
            "lambda_cpr": safe_float(args_dict.get("lambda_cpr", float("nan"))),
            "cpr_margin": safe_float(args_dict.get("cpr_margin", float("nan"))),
            "lambda_tur": safe_float(args_dict.get("lambda_tur", float("nan"))),
            "neg_num": int(args_dict.get("neg_num", -1)),
            "batch_size": int(args_dict.get("batch_size", -1)),
            "patch_size": int(args_dict.get("patch_size", -1)),
            "epochs": int(args_dict.get("epochs", -1)),
            "lr": safe_float(args_dict.get("lr", float("nan"))),
            "weight_decay": safe_float(args_dict.get("weight_decay", float("nan"))),
        }
        rows.append(row)

    os.makedirs(os.path.dirname(args.detail_csv), exist_ok=True)
    detail_fields = [
        "group",
        "run_dir",
        "ckpt_path",
        "epoch",
        "best_psnr",
        "seed",
        "use_cpr",
        "use_tur",
        "lambda_cpr",
        "cpr_margin",
        "lambda_tur",
        "neg_num",
        "batch_size",
        "patch_size",
        "epochs",
        "lr",
        "weight_decay",
    ]

    with open(args.detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for r in rows:
        grouped[r["group"]].append(r["best_psnr"])

    summary_rows = []
    for group in ["rec", "rec_cpr", "rec_tur", "rec_cpr_tur"]:
        vals = grouped.get(group, [])
        mean_psnr, std_psnr, n = summarize(vals)
        summary_rows.append(
            {
                "group": group,
                "num_runs": n,
                "best_psnr_mean": mean_psnr,
                "best_psnr_std": std_psnr,
            }
        )

    os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)
    with open(args.summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "num_runs", "best_psnr_mean", "best_psnr_std"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Detail CSV: {args.detail_csv}")
    print(f"Summary CSV: {args.summary_csv}")


if __name__ == "__main__":
    main()
