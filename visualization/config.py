"""Configuration, paths, and ripeness class metadata for the Streamlit dashboard."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_BEST_CKPT = PROJECT_ROOT / "checkpoints" / "studentmodal" / "best_student.pth"
DEFAULT_LAST_CKPT = PROJECT_ROOT / "checkpoints" / "studentmodal" / "last_student.pth"

MODE_SINGLE = "Single Image Prediction"
MODE_MULTI = "Multi-View Fruit Prediction (Multiple Angles)"

AGG_MEAN_PROB = "Mean Probability (Recommended)"
AGG_MAJORITY = "Majority Vote"

SUPPORTED_IMAGE_TYPES: Tuple[str, ...] = ("jpg", "jpeg", "png", "webp", "bmp")

CLASS_METADATA: Dict[str, Dict[str, str]] = {
    "Unripe": {
        "color": "#22C55E",
        "bg_rgba": "rgba(34, 197, 94, 0.14)",
        "border_rgba": "rgba(34, 197, 94, 0.55)",
        "badge": "UNRIPE (Immature / Not Ready)",
        "description": "Firm shell and flesh, tight spine grooves, and low soluble sugar content.",
        "recommendation": "Hold for further ripening; do not harvest or consume yet.",
    },
    "Ripe": {
        "color": "#F59E0B",
        "bg_rgba": "rgba(245, 158, 11, 0.14)",
        "border_rgba": "rgba(245, 158, 11, 0.55)",
        "badge": "RIPE (Optimal Quality)",
        "description": "Balanced sweetness, softened creamy flesh, and expanded spine grooves.",
        "recommendation": "Ready for immediate sale and consumption.",
    },
    "Overripe": {
        "color": "#EF4444",
        "bg_rgba": "rgba(239, 68, 68, 0.14)",
        "border_rgba": "rgba(239, 68, 68, 0.55)",
        "badge": "OVERRIPE (Past Peak / Fermenting)",
        "description": "Excessive softening, strong aroma, and potential shell splitting.",
        "recommendation": "Prioritize immediate processing or discard if spoiled.",
    },
}

CLASS_COLOR_MAP: Dict[str, str] = {cls: meta["color"] for cls, meta in CLASS_METADATA.items()}
