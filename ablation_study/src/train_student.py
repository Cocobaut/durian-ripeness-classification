"""Ablation study entrypoint to train the Vision-Only Student model (Stage 2 - CMKD).

Forwards all CLI arguments to the main module at:
`src/train_teacher_and_student/train_student.py`

Usage (from ablation_study/):
    python src/train_student.py --smoke
    python src/train_student.py --epochs 30 --batch-size 16 --eval-test --device cuda
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ABLATION_ROOT = SRC_DIR.parent
TARGET_SCRIPT = SRC_DIR / "train_teacher_and_student" / "train_student.py"

if str(ABLATION_ROOT) not in sys.path:
    sys.path.insert(0, str(ABLATION_ROOT))

if __name__ == "__main__":
    runpy.run_path(str(TARGET_SCRIPT), run_name="__main__")

