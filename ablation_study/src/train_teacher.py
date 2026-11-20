"""Ablation study entrypoint to train the Multi-Modal Teacher model.

Forwards all CLI arguments to the main module at:
`src/train_teacher_and_student/train_teacher.py`

Usage (from ablation_study/):
    python src/train_teacher.py --smoke
    python src/train_teacher.py --epochs 30 --warmup-epochs 5 --batch-size 8 --device cuda
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ABLATION_ROOT = SRC_DIR.parent
TARGET_SCRIPT = SRC_DIR / "train_teacher_and_student" / "train_teacher.py"

if str(ABLATION_ROOT) not in sys.path:
    sys.path.insert(0, str(ABLATION_ROOT))

if __name__ == "__main__":
    runpy.run_path(str(TARGET_SCRIPT), run_name="__main__")

