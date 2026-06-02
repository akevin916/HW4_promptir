"""
Plot training curves from experiment log files.

Usage:
    # Auto-detect all logs under exp_v2/
    python scripts/plot_training_curves.py --log_dir ckpt/exp_v2

    # Specify logs manually with custom labels
    python scripts/plot_training_curves.py \
        --logs ckpt/exp_v2/rec_cpr_seed3407.log ckpt/exp_v2/rec_cpr_tur_seed3407.log \
        --labels CPR CPR+TUR

    # Save to file instead of showing interactively
    python scripts/plot_training_curves.py --log_dir ckpt/exp_v2 --save curves.png
"""

import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── epoch line pattern  ([-\d.] handles negative train_l1 from TUR) ───────────
_EPOCH_RE = re.compile(
    r"\[Epoch\s+(\d+)\].*train_l1=([-\d.]+).*val_l1=([\d.]+).*val_psnr=([\d.]+)"
)


def parse_log(path: str) -> dict:
    epochs, train_l1, val_l1, val_psnr = [], [], [], []
    with open(path) as f:
        for line in f:
            m = _EPOCH_RE.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_l1.append(float(m.group(2)))
                val_l1.append(float(m.group(3)))
                val_psnr.append(float(m.group(4)))
    return dict(epochs=epochs, train_l1=train_l1, val_l1=val_l1, val_psnr=val_psnr)


def find_logs(log_dir: str) -> list[tuple[str, str]]:
    """Return [(label, path), ...] sorted by label."""
    result = []
    for p in sorted(Path(log_dir).glob("*.log")):
        label = p.stem  # e.g. rec_cpr_seed3407
        result.append((label, str(p)))
    return result


def _draw_metric(ax, parsed: dict, key: str, title: str, ylabel: str, higher_better: bool):
    colors = plt.cm.tab10.colors
    for i, (label, data) in enumerate(parsed.items()):
        epochs = data["epochs"]
        values = data[key]
        color  = colors[i % len(colors)]
        ax.plot(epochs, values, label=label, color=color, linewidth=1.5)
        best_idx = int(np.argmax(values) if higher_better else np.argmin(values))
        ax.scatter(
            epochs[best_idx], values[best_idx],
            color=color, s=60, zorder=5,
            marker="*" if higher_better else "v",
        )
        ax.annotate(
            f"{values[best_idx]:.3f}",
            (epochs[best_idx], values[best_idx]),
            textcoords="offset points",
            xytext=(4, 4), fontsize=7, color=color,
        )
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)


def plot(
    log_label_pairs: list[tuple[str, str]],
    save_path: str | None = None,
    split: bool = False,
    by_exp: bool = False,
):
    parsed = {}
    for label, path in log_label_pairs:
        data = parse_log(path)
        if not data["epochs"]:
            print(f"[warn] no epoch data found in {path}, skipping.")
            continue
        parsed[label] = data

    if not parsed:
        print("No data to plot.")
        return

    configs = [
        ("train_l1", "Train L1 Loss", "L1 Loss",   False),
        ("val_l1",   "Val L1 Loss",   "L1 Loss",   False),
        ("val_psnr", "Val PSNR (dB)", "PSNR (dB)",  True),
    ]

    if by_exp:
        # ── one figure per experiment, 3 subplots each ────────────────────────
        for label, data in parsed.items():
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            fig.suptitle(label, fontsize=13, fontweight="bold")
            single = {label: data}          # wrap so _draw_metric works
            for ax, (key, title, ylabel, higher_better) in zip(axes, configs):
                _draw_metric(ax, single, key, title, ylabel, higher_better)
            plt.tight_layout()
            if save_path:
                stem = Path(save_path).stem
                suffix = Path(save_path).suffix or ".png"
                safe_label = label.replace("+", "plus").replace(" ", "_")
                out = Path(save_path).parent / f"{stem}_{safe_label}{suffix}"
                plt.savefig(out, dpi=150, bbox_inches="tight")
                print(f"Saved to {out}")
            else:
                plt.show()
            plt.close()

    elif split:
        # ── one figure per metric, all experiments on same axes ───────────────
        for key, title, ylabel, higher_better in configs:
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.suptitle(title, fontsize=12, fontweight="bold")
            _draw_metric(ax, parsed, key, title, ylabel, higher_better)
            plt.tight_layout()
            if save_path:
                stem = Path(save_path).stem
                suffix = Path(save_path).suffix or ".png"
                out = Path(save_path).parent / f"{stem}_{key}{suffix}"
                plt.savefig(out, dpi=150, bbox_inches="tight")
                print(f"Saved to {out}")
            else:
                plt.show()
            plt.close()

    else:
        # ── all experiments + all metrics in one row ──────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("Training Curves", fontsize=13, fontweight="bold")
        for ax, (key, title, ylabel, higher_better) in zip(axes, configs):
            _draw_metric(ax, parsed, key, title, ylabel, higher_better)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")
        else:
            plt.show()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from log files.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--log_dir", type=str, help="Directory containing *.log files (auto-detect).")
    group.add_argument("--logs", nargs="+", type=str, help="Explicit list of log file paths.")
    parser.add_argument("--labels", nargs="+", type=str, help="Labels for each --logs entry (optional).")
    parser.add_argument("--save", type=str, default=None, help="Path to save the figure (e.g. curves.png). If omitted, display interactively.")
    parser.add_argument("--split", action="store_true", help="One figure per metric (all experiments overlaid on same axes).")
    parser.add_argument("--by-exp", action="store_true", dest="by_exp", help="One figure per experiment (3 subplots: train_l1, val_l1, val_psnr).")
    args = parser.parse_args()

    if args.log_dir:
        pairs = find_logs(args.log_dir)
        if not pairs:
            print(f"No *.log files found in {args.log_dir}")
            return
    else:
        if args.labels and len(args.labels) != len(args.logs):
            parser.error("--labels count must match --logs count")
        labels = args.labels or [Path(p).stem for p in args.logs]
        pairs = list(zip(labels, args.logs))

    print(f"Plotting {len(pairs)} experiment(s):")
    for label, path in pairs:
        data = parse_log(path)
        n = len(data["epochs"])
        best_psnr = max(data["val_psnr"]) if data["val_psnr"] else float("nan")
        print(f"  [{label}]  epochs={n}  best_val_psnr={best_psnr:.3f}")

    plot(pairs, save_path=args.save, split=args.split, by_exp=args.by_exp)


if __name__ == "__main__":
    main()
