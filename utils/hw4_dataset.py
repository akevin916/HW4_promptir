import os
import random
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def list_images_sorted(folder: str) -> List[str]:
    files = [f for f in os.listdir(folder) if is_image_file(f)]
    return sorted(files)


def map_degraded_to_clean(degraded_name: str) -> str:
    stem, ext = os.path.splitext(degraded_name)
    if stem.startswith("rain-"):
        return f"rain_clean-{stem.split('rain-')[1]}{ext}"
    if stem.startswith("snow-"):
        return f"snow_clean-{stem.split('snow-')[1]}{ext}"
    raise ValueError(f"Unsupported degraded filename format: {degraded_name}")


def read_rgb(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def to_tensor(image_hwc: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image_hwc.transpose(2, 0, 1)).float() / 255.0


def paired_random_crop(degraded: np.ndarray, clean: np.ndarray, patch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w, _ = degraded.shape
    if patch_size > h or patch_size > w:
        raise ValueError(
            f"Patch size {patch_size} is larger than image size {(h, w)}."
        )

    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)
    degraded_patch = degraded[top: top + patch_size, left: left + patch_size]
    clean_patch = clean[top: top + patch_size, left: left + patch_size]
    return degraded_patch, clean_patch


def paired_augmentation(degraded: np.ndarray, clean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mode = random.randint(0, 7)

    def aug(x: np.ndarray) -> np.ndarray:
        if mode == 0:
            return x
        if mode == 1:
            return np.flipud(x).copy()
        if mode == 2:
            return np.fliplr(x).copy()
        if mode == 3:
            return np.rot90(x, 1).copy()
        if mode == 4:
            return np.rot90(x, 2).copy()
        if mode == 5:
            return np.rot90(x, 3).copy()
        if mode == 6:
            return np.flipud(np.rot90(x, 1)).copy()
        return np.fliplr(np.rot90(x, 1)).copy()

    return aug(degraded), aug(clean)


class HW4TrainDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        patch_size: int = 128,
        augment: bool = True,
        file_list: List[str] = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.degraded_dir = os.path.join(root_dir, "train", "degraded")
        self.clean_dir = os.path.join(root_dir, "train", "clean")
        self.patch_size = patch_size
        self.augment = augment

        degraded_names = file_list if file_list is not None else list_images_sorted(self.degraded_dir)
        if not degraded_names:
            raise RuntimeError(f"No training images found in {self.degraded_dir}")

        self.samples = []
        for degraded_name in degraded_names:
            clean_name = map_degraded_to_clean(degraded_name)
            degraded_path = os.path.join(self.degraded_dir, degraded_name)
            clean_path = os.path.join(self.clean_dir, clean_name)
            if not os.path.exists(clean_path):
                raise FileNotFoundError(f"Missing clean pair for {degraded_name}: {clean_path}")
            self.samples.append((degraded_path, clean_path, degraded_name))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        degraded_path, clean_path, degraded_name = self.samples[idx]
        degraded = read_rgb(degraded_path)
        clean = read_rgb(clean_path)

        degraded, clean = paired_random_crop(degraded, clean, self.patch_size)
        if self.augment:
            degraded, clean = paired_augmentation(degraded, clean)

        return {
            "name": degraded_name,
            "degraded": to_tensor(degraded),
            "clean": to_tensor(clean),
        }


class HW4ValDataset(Dataset):
    def __init__(self, root_dir: str, file_list: List[str]):
        super().__init__()
        self.root_dir = root_dir
        self.degraded_dir = os.path.join(root_dir, "train", "degraded")
        self.clean_dir = os.path.join(root_dir, "train", "clean")

        self.samples = []
        for degraded_name in file_list:
            clean_name = map_degraded_to_clean(degraded_name)
            degraded_path = os.path.join(self.degraded_dir, degraded_name)
            clean_path = os.path.join(self.clean_dir, clean_name)
            if not os.path.exists(clean_path):
                raise FileNotFoundError(f"Missing clean pair for {degraded_name}: {clean_path}")
            self.samples.append((degraded_path, clean_path, degraded_name))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        degraded_path, clean_path, degraded_name = self.samples[idx]
        degraded = read_rgb(degraded_path)
        clean = read_rgb(clean_path)

        return {
            "name": degraded_name,
            "degraded": to_tensor(degraded),
            "clean": to_tensor(clean),
        }


class HW4SubmissionDataset(Dataset):
    def __init__(self, root_dir: str):
        super().__init__()
        self.degraded_dir = os.path.join(root_dir, "test", "degraded")
        self.names = list_images_sorted(self.degraded_dir)
        if not self.names:
            raise RuntimeError(f"No submission images found in {self.degraded_dir}")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        path = os.path.join(self.degraded_dir, name)
        degraded = read_rgb(path)
        return {
            "name": name,
            "degraded": to_tensor(degraded),
        }
