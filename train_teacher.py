"""Repository root entrypoint to train the Multi-Modal Teacher model.

Forwards all CLI arguments to the main module at:
`workspaces/member1_vision_lead/04_train_teacher_and_student/train_teacher.py`

Usage (from the repository root):
    python train_teacher.py --smoke
    python train_teacher.py --epochs 30 --warmup-epochs 5 --batch-size 8 --device cuda
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
TARGET_SCRIPT = ROOT_DIR / "workspaces" / "member1_vision_lead" / "04_train_teacher_and_student" / "train_teacher.py"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if __name__ == "__main__":
    runpy.run_path(str(TARGET_SCRIPT), run_name="__main__")

