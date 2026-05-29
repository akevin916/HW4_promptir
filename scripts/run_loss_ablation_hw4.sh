#!/usr/bin/env bash
set -euo pipefail

# Usage example:
#   bash scripts/run_loss_ablation_hw4.sh \
#     --data-root data/release_folder/hw4_realse_dataset \
#     --out-root experiments/loss_ablation \
#     --epochs 200 --batch-size 8 --patch-size 128 --num-workers 8 --amp \
#     --seeds 3407,3408,3409

DATA_ROOT="data/release_folder/hw4_realse_dataset"
OUT_ROOT="experiments/loss_ablation"
EPOCHS=200
BATCH_SIZE=8
PATCH_SIZE=128
NUM_WORKERS=8
LR=2e-4
WEIGHT_DECAY=1e-4
VAL_RATIO=0.1
SAVE_EVERY=20
PROMPT_LEN=5
NUM_EXPERT=1
NEG_NUM=2
LAMBDA_CPR=0.1
CPR_MARGIN=0.01
LAMBDA_TUR=1.0
TUR_EPS=1e-8
SEEDS="3407,3408,3409"
USE_AMP=0
RUN_EVAL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --patch-size) PATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --val-ratio) VAL_RATIO="$2"; shift 2 ;;
    --save-every) SAVE_EVERY="$2"; shift 2 ;;
    --prompt-len) PROMPT_LEN="$2"; shift 2 ;;
    --num-expert) NUM_EXPERT="$2"; shift 2 ;;
    --neg-num) NEG_NUM="$2"; shift 2 ;;
    --lambda-cpr) LAMBDA_CPR="$2"; shift 2 ;;
    --cpr-margin) CPR_MARGIN="$2"; shift 2 ;;
    --lambda-tur) LAMBDA_TUR="$2"; shift 2 ;;
    --tur-eps) TUR_EPS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --amp) USE_AMP=1; shift ;;
    --run-eval) RUN_EVAL=1; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p "$OUT_ROOT"

IFS=',' read -r -a SEED_LIST <<< "$SEEDS"

run_group() {
  local group="$1"
  local seed="$2"

  local save_dir="$OUT_ROOT/${group}_seed${seed}"
  local log_file="$save_dir/train.log"
  mkdir -p "$save_dir"

  local -a cmd=(
    python train_hw4.py
    --data_root "$DATA_ROOT"
    --save_dir "$save_dir"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --patch_size "$PATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --lr "$LR"
    --weight_decay "$WEIGHT_DECAY"
    --val_ratio "$VAL_RATIO"
    --seed "$seed"
    --save_every "$SAVE_EVERY"
    --prompt_len "$PROMPT_LEN"
    --num_expert "$NUM_EXPERT"
    --neg_num "$NEG_NUM"
    --lambda_cpr "$LAMBDA_CPR"
    --cpr_margin "$CPR_MARGIN"
    --lambda_tur "$LAMBDA_TUR"
    --tur_eps "$TUR_EPS"
  )

  if [[ "$USE_AMP" -eq 1 ]]; then
    cmd+=(--amp)
  fi

  case "$group" in
    rec)
      ;;
    rec_cpr)
      cmd+=(--use_cpr)
      ;;
    rec_tur)
      cmd+=(--use_tur)
      ;;
    rec_cpr_tur)
      cmd+=(--use_cpr --use_tur)
      ;;
    *)
      echo "Unknown group: $group"
      exit 1
      ;;
  esac

  echo "[Run] group=$group seed=$seed"
  printf '%q ' "${cmd[@]}" | tee "$save_dir/cmd.txt"
  echo
  "${cmd[@]}" 2>&1 | tee "$log_file"

  if [[ "$RUN_EVAL" -eq 1 ]]; then
    local ckpt="$save_dir/best.pth"
    local pred_npz="$save_dir/pred.npz"
    if [[ -f "$ckpt" ]]; then
      python eval_hw4.py --data_root "$DATA_ROOT" --ckpt "$ckpt" --output_npz "$pred_npz" 2>&1 | tee -a "$log_file"
    else
      echo "[Warn] missing checkpoint: $ckpt" | tee -a "$log_file"
    fi
  fi
}

GROUPS=(rec rec_cpr rec_tur rec_cpr_tur)

for g in "${GROUPS[@]}"; do
  for s in "${SEED_LIST[@]}"; do
    run_group "$g" "$s"
  done
done

echo "All experiments finished under: $OUT_ROOT"
