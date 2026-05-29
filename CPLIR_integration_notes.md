# CPLIR Integration Notes (Stage 2)

This document records the remaining CPLIR-style integration work after sparse prompt routing was added to `net/model.py`.

## 1) What was finished in this step

### A. Add training-time CPR controls in `options.py`
File: `options.py`

Added arguments:
- `--prompt_len`: number of prompt components per prompt block.
- `--num_expert`: top-k experts selected by sparse routing.
- `--use_cpr`: enable/disable contrastive prompt regularization.
- `--neg_num`: number of negative prompt samples per batch.
- `--lambda_cpr`: CPR loss weight.
- `--cpr_margin`: ranking margin for CPR.

Why:
- Needed to control sparse prompt and CPR behavior from command line.

---

### B. Integrate CPR training objective in `train.py`
File: `train.py`

Key changes:
- Model init now passes prompt routing config:
  - `PromptIR(decoder=True, prompt_len=opt.prompt_len, num_expert=opt.num_expert)`
- Forward call now explicitly uses positive path:
  - `restored = self.net(degrad_patch, is_neg=False)`
- Added CPR branch when `opt.use_cpr` is enabled:
  1. Sample `neg_num` negative restorations with `is_neg=True`.
  2. Compute average negative reconstruction loss.
  3. Compute ranking-style CPR term:
     - `cpr_loss = relu(rec_loss + margin - neg_loss)`
  4. Final loss:
     - `loss = rec_loss + lambda_cpr * cpr_loss`
- Scheduler now uses `opt.epochs` and optimizer uses `opt.lr`.

Why:
- This makes the model prefer outputs from matched prompts over mismatched prompts.

---

### C. Integrate the same CPR flow in `train_hw4.py`
File: `train_hw4.py`

Key changes:
- Added the same CLI controls (`prompt_len`, `num_expert`, `use_cpr`, `neg_num`, `lambda_cpr`, `cpr_margin`).
- Model init now passes sparse prompt config to `PromptIR`.
- Training loop now computes:
  - `rec_loss` from positive prompt path.
  - optional `cpr_loss` from negative prompt path.
  - `loss = rec_loss + lambda_cpr * cpr_loss`.
- Progress bar now reports total, rec, and cpr losses.

Why:
- HW4 script is your main training path; CPR needed there too.

## 2) CPR math used here

Positive output:
- `y_pos = f(x, is_neg=False)`

Negative outputs:
- `y_neg_i = f(x, is_neg=True)`, for `i = 1..N`

Reconstruction:
- `L_rec = L1(y_pos, y_gt)`

Negative average:
- `L_neg = mean_i L1(y_neg_i, y_gt)`

CPR ranking term:
- `L_cpr = relu(L_rec + m - L_neg)`

Final objective:
- `L_total = L_rec + lambda * L_cpr`

Interpretation:
- Minimizing `L_cpr` enforces `L_neg >= L_rec + m`, i.e., mismatched prompts should perform worse than matched prompts by margin `m`.

## 3) Torch syntax explained (important pieces)

### `torch.topk`
- Returns top-k values/indices along a dimension.
- Used in prompt routing to choose best experts.

### `tensor.gather(dim, index)`
- Reads elements from `tensor` at `index` along `dim`.
- Used to fetch only selected expert weights.

### `tensor.scatter_(dim, index, src)`
- Writes `src` values into a target tensor at `index` along `dim`.
- Used to build sparse expert-weight tensors.

### `with torch.no_grad():`
- Disables autograd graph creation in the block.
- Used for negative branch sampling to reduce memory and avoid unwanted gradient paths.

### `tensor.detach()`
- Returns a tensor disconnected from current graph.
- Used in negative prompt branch inside prompt module to stop gradients through negative prompt composition.

### `torch.relu(...)`
- Applies `max(0, x)` element-wise.
- Used for hinge/ranking-style CPR term.

### `torch.stack(list_of_tensors).mean()`
- Stacks scalar losses into one tensor and averages them.
- Used to aggregate multiple negative losses.

## 4) Recommended starting hyperparameters

- `--prompt_len 5`
- `--num_expert 1`
- `--use_cpr`
- `--neg_num 2`
- `--lambda_cpr 0.1`
- `--cpr_margin 0.01`

If unstable:
- lower `lambda_cpr` to `0.05`
- keep `neg_num` at `1` or `2`

## 5) Example command (HW4)

```bash
python train_hw4.py \
  --data_root data/release_folder/hw4_realse_dataset \
  --save_dir ckpt/hw4 \
  --epochs 200 \
  --batch_size 8 \
  --patch_size 128 \
  --num_workers 8 \
  --amp \
  --prompt_len 5 \
  --num_expert 1 \
  --use_cpr \
  --neg_num 2 \
  --lambda_cpr 0.1 \
  --cpr_margin 0.01
```

## 6) What is still optional (not required for this stage)

- Logging expert usage frequency per task (`de_id`) for interpretability.
- Perceptual-feature CPR instead of pixel-L1 CPR.
- Warmup schedule for CPR weight (`lambda_cpr`).

## 7) Loss Ablation Scripts (A/B/C/D + multi-seed)

Two helper scripts are added under `scripts/` for complete CPR/TUR experiments.

### A. Run all 4 groups with multiple seeds

File: `scripts/run_loss_ablation_hw4.sh`

Groups:
- `rec` (L_rec only)
- `rec_cpr` (L_rec + CPR)
- `rec_tur` (uncertainty-weighted TUR)
- `rec_cpr_tur` (CPR + TUR)

Example:

```bash
bash scripts/run_loss_ablation_hw4.sh \
  --data-root data/release_folder/hw4_realse_dataset \
  --out-root experiments/loss_ablation \
  --epochs 200 --batch-size 8 --patch-size 128 --num-workers 8 --amp \
  --seeds 3407,3408,3409 \
  --lambda-cpr 0.1 --cpr-margin 0.01 \
  --lambda-tur 1.0
```

Optional:
- Add `--run-eval` to also generate `pred.npz` after each run.

### B. Aggregate checkpoints into CSV report

File: `scripts/aggregate_loss_ablation.py`

Example:

```bash
python scripts/aggregate_loss_ablation.py \
  --exp-root experiments/loss_ablation \
  --detail-csv experiments/loss_ablation/detail.csv \
  --summary-csv experiments/loss_ablation/summary.csv
```

Outputs:
- `detail.csv`: one row per run (group, seed, best_psnr, hyperparameters, paths)
- `summary.csv`: mean/std of best_psnr for each group
