"""Model loading, image preprocessing, inference, and fruit-level aggregation."""

from __future__ import annotations

import hashlib
import io
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from src.datasets.image_dataset import (
    IMAGE_SIZE,
    camera_angle_from_name,
    fruit_id_from_name,
    get_eval_transforms,
)
from src.models.student_model import VisionStudent, build_student_from_checkpoint
from src.utils.common import CLASS_ORDER
from visualization.config import AGG_MEAN_PROB


@st.cache_resource(show_spinner="Loading Vision-Only Student model...")
def load_student_model(ckpt_path: str, device_str: str) -> Tuple[VisionStudent, Dict[str, Any]]:
    """Load and cache the VisionStudent checkpoint for the selected device."""
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_student_from_checkpoint(path, device=device_str, eval_mode=True)
    meta = {
        "ckpt_name": path.name,
        "epoch": ckpt.get("epoch", "N/A"),
        "phase": ckpt.get("phase", "N/A"),
        "student_config": ckpt.get("student_config", {}),
        "num_params": sum(p.numel() for p in model.parameters()),
        "file_size_mb": path.stat().st_size / (1024 * 1024),
    }
    return model, meta


@st.cache_resource
def load_eval_transform():
    """Return the canonical evaluation transform used during Student validation/testing."""
    return get_eval_transforms(IMAGE_SIZE)


def save_uploaded_checkpoint(uploaded_ckpt) -> str:
    """Persist an uploaded .pth file by content hash so cached loading can reuse it."""
    raw_bytes = uploaded_ckpt.getvalue()
    digest = hashlib.sha256(raw_bytes).hexdigest()[:16]
    cache_path = Path(tempfile.gettempdir()) / f"durian_student_{digest}.pth"
    if not cache_path.exists():
        cache_path.write_bytes(raw_bytes)
    return str(cache_path)


def open_rgb_image(image_bytes: bytes) -> Image.Image:
    """Decode raw image bytes into an EXIF-oriented RGB PIL image."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def predict_single_image(
    model: VisionStudent,
    eval_transform,
    image_bytes: bytes,
    device_str: str,
) -> Dict[str, Any]:
    """Preprocess an RGB image with get_eval_transforms() and run VisionStudent."""
    rgb_img = open_rgb_image(image_bytes)
    tensor = eval_transform(rgb_img).unsqueeze(0).to(torch.device(device_str))

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor, return_features=False)
        probs = F.softmax(logits.float(), dim=-1)[0].detach().cpu().numpy().astype(float)
    if device_str.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    probs = probs / max(float(probs.sum()), 1e-12)
    pred_idx = int(probs.argmax())
    return {
        "rgb_image": rgb_img,
        "pred_idx": pred_idx,
        "pred_label": CLASS_ORDER[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probs": {cls: float(probs[i]) for i, cls in enumerate(CLASS_ORDER)},
        "latency_ms": latency_ms,
    }


def run_views_inference(
    model: VisionStudent,
    eval_transform,
    items: List[Dict[str, Any]],
    device_str: str,
) -> List[Dict[str, Any]]:
    """Run VisionStudent inference across one or more image items and attach filename metadata."""
    results: List[Dict[str, Any]] = []
    for item in items:
        res = predict_single_image(model, eval_transform, item["bytes"], device_str)
        res["name"] = item["name"]
        res["fruit_id"] = fruit_id_from_name(item["name"]) or "Unassigned"
        res["camera_angle"] = camera_angle_from_name(item["name"])
        results.append(res)
    return results


def aggregate_fruit_predictions(df: pd.DataFrame, strategy: str) -> Tuple[str, float, np.ndarray, float]:
    """Aggregate multiple view predictions into a single fruit-level ripeness decision."""
    mean_probs = np.array(
        [df["Prob Unripe"].mean(), df["Prob Ripe"].mean(), df["Prob Overripe"].mean()],
        dtype=float,
    )
    if strategy == AGG_MEAN_PROB or strategy.startswith("Mean Probability"):
        fruit_idx = int(mean_probs.argmax())
    else:
        pred_indices = df["Predicted Class"].map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy()
        votes = np.bincount(pred_indices, minlength=len(CLASS_ORDER))
        candidates = np.flatnonzero(votes == votes.max())
        sum_probs = np.array([df["Prob Unripe"].sum(), df["Prob Ripe"].sum(), df["Prob Overripe"].sum()])
        fruit_idx = int(candidates[np.argmax(sum_probs[candidates])])

    fruit_label = CLASS_ORDER[fruit_idx]
    fruit_conf = float(mean_probs[fruit_idx])
    consistency = float((df["Predicted Class"] == fruit_label).mean() * 100.0)
    return fruit_label, fruit_conf, mean_probs, consistency


def compute_fingerprint(
    prediction_mode: str,
    ckpt_path: str,
    device_str: str,
    items: List[Dict[str, Any]],
) -> str:
    """Create a deterministic cache key for the current inputs, mode, and model selection."""
    h = hashlib.sha256()
    h.update(f"{prediction_mode}|{ckpt_path}|{device_str}".encode("utf-8"))
    for item in items:
        h.update(item["name"].encode("utf-8"))
        h.update(hashlib.sha256(item["bytes"]).digest())
    return h.hexdigest()
