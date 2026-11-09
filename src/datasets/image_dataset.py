"""Dataset & tiền xử lý nhánh Thị giác (RGB).

Dự án: Phân loại độ chín sầu riêng bằng Chưng cất tri thức Đa phương thức (CMKD)

Transform giữ NGUYÊN như notebook huấn luyện unimodal vision
(workspaces/member1_vision_lead/02_train_unimodal_vision/01_train_unimodal_vision_convnextv2_base.ipynb)
để backbone ConvNeXt V2 nhận đúng phân phối dữ liệu đã học:
CLAHE kênh L (LAB) -> 384x384 -> Normalize ImageNet.

Dùng cho:
- Teacher: thành phần con của MultimodalDataset (src/datasets/multimodal_dataset.py).
- Student (Giai đoạn 2): ImageDataset dùng trực tiếp (Student chỉ nhận ảnh RGB).
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
CAMERA_ANGLES = ["TOP", "BOT", "SA", "SB", "SC", "SD"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
_FID_RE = re.compile(r"durian_?0*(\d+)", re.IGNORECASE)


class CLAHETransform:
    """CLAHE trên kênh L (không gian LAB) làm nổi kết cấu vi mô rãnh gai."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        # Tạo CLAHE trong __call__ để đối tượng pickle được (DataLoader nhiều worker trên Windows)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        lab = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        merged = cv2.merge((clahe.apply(l), a, b))
        return Image.fromarray(cv2.cvtColor(merged, cv2.COLOR_LAB2RGB))


def get_train_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
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
    """Liệt kê ảnh RGB của một split: cột [path, file, fruit_id, camera_angle].

    Tên file chuẩn của data/final: `{fruit_id}_{ANGLE}[_k]_{Ripeness}_rgb.jpg`.
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
            rows.append({"path": str(p), "file": p.name, "fruit_id": fid,
                         "camera_angle": camera_angle_from_name(p.name)})
    return pd.DataFrame(rows, columns=["path", "file", "fruit_id", "camera_angle"])


def load_rgb(path: Union[str, Path]) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


class ImageDataset(Dataset):
    """Dataset ảnh RGB đơn phương thức (dùng cho Student / Baseline Vision).

    Mỗi phần tử: (image [3, H, W], label, fruit_id, {"camera_angle": ...}).
    """

    def __init__(self, records: pd.DataFrame, label_map: Dict[str, int],
                 transform: Optional[transforms.Compose] = None):
        """
        Args:
            records: DataFrame từ `list_rgb_images` (cần cột path, fruit_id, camera_angle).
            label_map: fruit_id -> label_idx (lấy từ final_multimodal_manifest.csv).
            transform: transform ảnh (mặc định get_eval_transforms()).
        """
        self.records = records[records["fruit_id"].isin(label_map)].reset_index(drop=True)
        self.label_map = label_map
        self.transform = transform or get_eval_transforms()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        row = self.records.iloc[idx]
        img = self.transform(load_rgb(row["path"]))
        return img, int(self.label_map[row["fruit_id"]]), row["fruit_id"], {"camera_angle": row["camera_angle"]}

