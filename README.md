# HW4 版：環境重建、訓練、評估

本文件提供從零開始重建環境，到啟動訓練與產生 `pred.npz` 的完整流程。

## 1) 建立乾淨環境

```bash
conda env create -f env_hw4.yml
conda activate promptir-hw4
```

驗證 CUDA 與 PyTorch：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 2) 確認資料集目錄

預設使用下列路徑：

```text
data/release_folder/hw4_realse_dataset/
  train/
    clean/
    degraded/
  test/
    degraded/
```

如果你的資料放在其他地方，訓練與評估時用 `--data_root` 指定。

## 3) 從零開始訓練（不使用預訓練權重）

`train_hw4.py` 預設就是 from scratch，不會載入任何預訓練模型。

```bash
python train_hw4.py \
  --data_root data/release_folder/hw4_realse_dataset \
  --save_dir ckpt/hw4 \
  --epochs 200 \
  --batch_size 8 \
  --patch_size 128 \
  --num_workers 8 \
  --amp
```

訓練後會產生：

- `ckpt/hw4/best.pth`：驗證 PSNR 最佳模型
- `ckpt/hw4/last.pth`：最後一個 epoch 模型

## 4) 產生提交檔 pred.npz

```bash
python eval_hw4.py \
  --data_root data/release_folder/hw4_realse_dataset \
  --ckpt ckpt/hw4/best.pth \
  --output_npz data/release_folder/pred.npz
```

`pred.npz` 內容格式：

- key: 測試影像檔名（例如 `0.png`）
- value: `uint8`、shape=`(3, H, W)` 的 numpy array

這個格式與 `data/release_folder/pred.npz` 範例一致。

## 5) 可選：同時輸出還原 PNG

```bash
python eval_hw4.py \
  --data_root data/release_folder/hw4_realse_dataset \
  --ckpt ckpt/hw4/best.pth \
  --output_npz data/release_folder/pred.npz \
  --save_png_dir output/hw4_test_png
```


## 6)

```bash
cd /home/cvml_7/Desktop/2026_class/PromptIR

# 1) 接續訓練到 200 epochs（沿用你目前較穩定設定）
conda run -n promptir-hw4 python train_hw4.py \
  --data_root data/release_folder/hw4_realse_dataset \
  --save_dir ckpt/hw4 \
  --epochs 200 \
  --batch_size 2 \
  --patch_size 64 \
  --num_workers 4 \
  --amp \
  --resume ckpt/hw4/last.pth | tee -a train_hw4.log

# 2) 訓練完成後產生提交檔 pred.npz
conda run -n promptir-hw4 python eval_hw4.py \
  --data_root data/release_folder/hw4_realse_dataset \
  --ckpt ckpt/hw4/best.pth \
  --output_npz data/release_folder/pred.npz

# 3) 檢查 pred.npz 格式
conda run -n promptir-hw4 python -c "import numpy as np; p=np.load('data/release_folder/pred.npz'); k=sorted(p.files); print('num=',len(k),'first=',k[:5],'shape=',p[k[0]].shape,'dtype=',p[k[0]].dtype)"
```

如果你要背景跑訓練（關掉終端也繼續）就用：

```bash
cd /home/cvml_7/Desktop/2026_class/PromptIR
nohup conda run -n promptir-hw4 python train_hw4.py \
  --data_root data/release_folder/hw4_realse_dataset \
  --save_dir ckpt/hw4 \
  --epochs 200 \
  --batch_size 2 \
  --patch_size 64 \
  --num_workers 4 \
  --amp \
  --resume ckpt/hw4/last.pth > train_hw4.log 2>&1 &
```

查看進度：

```bash
tail -f train_hw4.log
```