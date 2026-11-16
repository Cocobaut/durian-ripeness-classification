"""Entrypoint ở thư mục gốc repo để huấn luyện mô hình Vision-Only Student (Giai đoạn 2 - CMKD).

Chuyển tiếp toàn bộ đối số dòng lệnh sang module chính tại:
`workspaces/member1_vision_lead/04_train_teacher_and_student/train_student.py`

Cách chạy (từ thư mục gốc repo):
    python train_student.py --smoke
    python train_student.py --epochs 30 --batch-size 16 --eval-test --device cuda
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
TARGET_SCRIPT = ROOT_DIR / "workspaces" / "member1_vision_lead" / "04_train_teacher_and_student" / "train_student.py"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if __name__ == "__main__":
    runpy.run_path(str(TARGET_SCRIPT), run_name="__main__")

