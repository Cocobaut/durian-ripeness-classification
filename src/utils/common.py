"""Project-wide shared utilities (seeds, paths, class constants)."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import torch

CLASS_ORDER = ["Unripe", "Ripe", "Overripe"]  # 0: Unripe, 1: Ripe, 2: Overripe
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}
IDX_TO_LABEL = {i: c for i, c in enumerate(CLASS_ORDER)}
NUM_CLASSES = len(CLASS_ORDER)

# Repository root = 2 levels above this file (src/utils/common.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def add_to_sys_path(*paths: Union[str, Path]) -> None:
    """Add directories to sys.path (for workspace folders starting with digits that cannot be imported as packages)."""
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
    """Resolve relative paths from the repository root (independent of current working directory)."""
    p = Path(path)
    return p if p.is_absolute() else (root or PROJECT_ROOT) / p


def first_existing(candidates: Iterable[Union[str, Path]], root: Optional[Path] = None) -> Optional[Path]:
    """Return the first existing path among candidates (or None if none exist)."""
    for c in candidates:
        if c is None:
            continue
        p = resolve_path(c, root)
        if p.exists():
            return p
    return None

