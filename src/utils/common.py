"""Tiện ích dùng chung toàn dự án (seed, đường dẫn, hằng số lớp)."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import torch

CLASS_ORDER = ["Unripe", "Ripe", "Overripe"]  # 0: Chưa chín, 1: Chín, 2: Chín quá
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}
IDX_TO_LABEL = {i: c for i, c in enumerate(CLASS_ORDER)}
NUM_CLASSES = len(CLASS_ORDER)

# Thư mục gốc repo = 2 cấp trên file này (src/utils/common.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def add_to_sys_path(*paths: Union[str, Path]) -> None:
    """Thêm thư mục vào sys.path (dùng cho các workspace có tên bắt đầu bằng số, không import được như package)."""
    for p in paths:
        s = str(Path(p).resolve())
        if s not in sys.path:
            sys.path.insert(0, s)


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def resolve_path(path: Union[str, Path], root: Optional[Path] = None) -> Path:
    """Đường dẫn tương đối được tính từ thư mục gốc repo (không phụ thuộc thư mục đang đứng)."""
    p = Path(path)
    return p if p.is_absolute() else (root or PROJECT_ROOT) / p


def first_existing(candidates: Iterable[Union[str, Path]], root: Optional[Path] = None) -> Optional[Path]:
    """Trả về đường dẫn đầu tiên tồn tại trong danh sách ứng viên (None nếu không có)."""
    for c in candidates:
        if c is None:
            continue
        p = resolve_path(c, root)
        if p.exists():
            return p
    return None

