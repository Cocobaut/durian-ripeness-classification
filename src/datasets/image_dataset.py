"""Dataset & preprocessing for the Vision modality (Multi-View RGB Fruit Bags).

Project: Durian Ripeness Classification via Cross-Modal Knowledge Distillation (CMKD)

Preserves the EXACT transforms and Multi-View Fruit Grouping from the improved unimodal vision notebook
(`workspaces/member1_vision_lead/02_train_unimodal_vision/train_unimodal_vision_convnextv2_improve.ipynb`):
- CLAHE on L channel (LAB) -> 384x384 -> ImageNet Normalize (+ RandomErasing in train).
- Groups all views belonging to the same `fruit_id` (e.g., `durian_002`) into a Multi-View Fruit Bag
  `[N, 3, 384, 384]` before passing through ConvNeXt V2-Base -> 256-d Projection -> Lightweight Gated Attention.

Used for:
- Teacher: sub-component of MultimodalDataset (`src/datasets/multimodal_dataset.py`).
- Student (Stage 2): `ImageDataset` used directly for validation and test evaluation (RGB-only).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 384
MIN_TRAIN_VIEWS = 4
MAX_TRAIN_VIEWS = 6
MAX_EVAL_VIEWS = 24
CAMERA_ANGLES = ["TOP", "BOT", "SA", "SB", "SC", "SD"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
_FID_RE = re.compile(r"durian_?0*(\d+)", re.IGNORECASE)


class CLAHETransform:
    """CLAHE on the L channel (LAB color space) to highlight micro-textures along spine grooves."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        # Instantiate CLAHE inside __call__ so the transform object is picklable (multi-worker DataLoader on Windows)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        lab = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        merged = cv2.merge((clahe.apply(l), a, b))
        return Image.fromarray(cv2.cvtColor(merged, cv2.COLOR_LAB2RGB))


def get_train_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
        transforms.RandomResizedCrop(image_size, scale=(0.82, 1.0), ratio=(0.92, 1.08)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
    ])


def get_eval_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
        transforms.Resize(int(image_size * 1.08)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def fruit_id_from_name(name: str) -> Optional[str]:
    m = _FID_RE.search(name)
    return f"durian_{int(m.group(1)):03d}" if m else None


def camera_angle_from_name(name: str) -> str:
    upper = name.upper()
    for a in CAMERA_ANGLES:
        if f"_{a}_" in upper or f"_{a}." in upper:
            return a
    return "UNKNOWN"


def list_rgb_images(rgb_dir: Union[str, Path]) -> pd.DataFrame:
    """List RGB images in a split: columns [path, file, fruit_id, camera_angle].

    Standard filename format in data/final: `{fruit_id}_{ANGLE}[_k]_{Ripeness}_rgb.jpg`.
    """
    rgb_dir = Path(rgb_dir)
    rows = []
    if rgb_dir.is_dir():
        for p in sorted(rgb_dir.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            fid = fruit_id_from_name(p.name)
            if fid is None:
                continue
            rows.append({
                "path": str(p),
                "file": p.name,
                "fruit_id": fid,
                "camera_angle": camera_angle_from_name(p.name),
            })
    return pd.DataFrame(rows, columns=["path", "file", "fruit_id", "camera_angle"])


def load_rgb(path: Union[str, Path]) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def load_multiview_bag_tensor(
    paths: List[str],
    transform: transforms.Compose,
    image_size: int = IMAGE_SIZE,
    is_train: bool = False,
    min_train_views: int = MIN_TRAIN_VIEWS,
    max_train_views: int = MAX_TRAIN_VIEWS,
    max_eval_views: int = MAX_EVAL_VIEWS,
    pad_views: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[torch.Tensor, int]:
    """Load a multi-view fruit bag into a 4D tensor `[N_pad, 3, H, W]` (zero-padded to `pad_views`)."""
    total_v = len(paths)
    target_pad = max(1, int(pad_views or (max_train_views if is_train else max(1, min(total_v, max_eval_views)))))

    if total_v == 0:
        return torch.zeros(target_pad, 3, image_size, image_size, dtype=torch.float32), 0

    if is_train:
        low_k = min(total_v, min_train_views)
        high_k = min(total_v, max_train_views)
        if rng is not None:
            k = int(rng.integers(low_k, high_k + 1)) if high_k > low_k else high_k
            indices = sorted(rng.choice(total_v, size=k, replace=False).tolist())
        else:
            k = int(np.random.randint(low_k, high_k + 1)) if high_k > low_k else high_k
            indices = sorted(np.random.choice(total_v, size=k, replace=False).tolist())
        selected = [paths[i] for i in indices]
    else:
        selected = paths[:max_eval_views]

    tensors = [transform(load_rgb(p)) for p in selected]
    num_valid = len(tensors)
    stacked = torch.stack(tensors, dim=0)  # [num_valid, 3, H, W]

    if stacked.size(0) < target_pad:
        pad = stacked.new_zeros((target_pad - stacked.size(0), *stacked.shape[1:]))
        stacked = torch.cat([stacked, pad], dim=0)
    elif stacked.size(0) > target_pad:
        stacked = stacked[:target_pad]
        num_valid = target_pad

    return stacked, num_valid


class ImageDataset(Dataset):
    """Multi-View RGB dataset grouping views by `fruit_id` (used for VisionStudent / Vision Baseline).

    When `multiview=True` (default):
        Each item is a Multi-View Fruit Bag: `(views [N_pad, 3, H, W], label, fruit_id, {"camera_angle": "MULTI_VIEW"})`.
    When `multiview=False`:
        Each item is a single view: `(image [3, H, W], label, fruit_id, {"camera_angle": ...})`.
    """

    def __init__(
        self,
        records: pd.DataFrame,
        label_map: Dict[str, int],
        transform: Optional[transforms.Compose] = None,
        multiview: bool = True,
        is_train: bool = False,
        image_size: int = IMAGE_SIZE,
        min_train_views: int = MIN_TRAIN_VIEWS,
        max_train_views: int = MAX_TRAIN_VIEWS,
        max_eval_views: int = MAX_EVAL_VIEWS,
        seed: int = 42,
    ):
        self.records = records[records["fruit_id"].isin(label_map)].reset_index(drop=True)
        self.label_map = label_map
        self.multiview = multiview
        self.is_train = is_train
        self.image_size = int(image_size)
        self.min_train_views = int(min_train_views)
        self.max_train_views = int(max_train_views)
        self.max_eval_views = int(max_eval_views)
        self.transform = transform or (get_train_transforms(self.image_size) if is_train else get_eval_transforms(self.image_size))
        self.rng = np.random.default_rng(seed)

        self.bags: List[Dict] = []
        if self.multiview and len(self.records) > 0:
            for fid, grp in self.records.groupby("fruit_id", sort=True):
                grp_sorted = grp.sort_values("file")
                self.bags.append({
                    "fruit_id": fid,
                    "label": int(self.label_map[fid]),
                    "paths": grp_sorted["path"].tolist(),
                    "files": grp_sorted["file"].tolist(),
                    "camera_angles": grp_sorted["camera_angle"].tolist(),
                })
            max_bag_len = max(len(b["paths"]) for b in self.bags)
            self.pad_views = (
                min(self.max_train_views, max_bag_len)
                if self.is_train
                else min(self.max_eval_views, max_bag_len)
            )
        else:
            self.pad_views = 1

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.bags) if self.multiview else len(self.records)

    def __getitem__(self, idx: int):
        if not self.multiview:
            row = self.records.iloc[idx]
            img = self.transform(load_rgb(row["path"]))
            return img, int(self.label_map[row["fruit_id"]]), row["fruit_id"], {"camera_angle": row["camera_angle"]}

        bag = self.bags[idx]
        views_tensor, _num_valid = load_multiview_bag_tensor(
            bag["paths"],
            transform=self.transform,
            image_size=self.image_size,
            is_train=self.is_train,
            min_train_views=self.min_train_views,
            max_train_views=self.max_train_views,
            max_eval_views=self.max_eval_views,
            pad_views=self.pad_views,
            rng=self.rng,
        )
        return views_tensor, bag["label"], bag["fruit_id"], {"camera_angle": "MULTI_VIEW"}
