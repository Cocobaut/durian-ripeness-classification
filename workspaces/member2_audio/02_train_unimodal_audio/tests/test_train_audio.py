"""Smoke test cho train_audio.py trên dữ liệu âm thanh giả (tiếng gõ tổng hợp), chạy trên CPU.

Chạy: python -m pytest workspaces/member2_audio/02_train_unimodal_audio/tests -q
Cần ffmpeg (AudioPreprocessor dùng ffmpeg để đọc file).
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_audio as ta  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="cần ffmpeg")

SR = 48000
FREQ = {"Unripe": 900.0, "Ripe": 650.0, "Overripe": 450.0}


def synth_knocks(label: str, seed: int, seconds: float = 12.0) -> np.ndarray:
    """Chuỗi tiếng gõ: sin tắt dần, tần số phụ thuộc nhãn, cách nhau ~0.8 s, có nhiễu nền."""
    rng = np.random.default_rng(seed)
    y = 0.002 * rng.standard_normal(int(SR * seconds))
    t = np.arange(int(0.35 * SR)) / SR
    pos = 0.5
    while pos + 0.5 < seconds:
        f = FREQ[label] * rng.uniform(0.95, 1.05)
        knock = 0.6 * np.exp(-t * 18) * np.sin(2 * np.pi * f * t)
        i = int(pos * SR)
        y[i:i + len(knock)] += knock
        pos += rng.uniform(0.7, 0.9)
    return y.astype(np.float32)


@pytest.fixture(scope="module")
def fake_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("final")
    rows, fid = [], 1
    for split, n_per_class in (("train", 4), ("valid", 2), ("test", 2)):
        (root / split / "sound").mkdir(parents=True)
        for label in ta.CLASS_ORDER:
            for _ in range(n_per_class):
                for device in ("android", "iphone"):
                    path = root / split / "sound" / f"durian_{fid:03d}_{device}_{label}_knock.wav"
                    sf.write(path, synth_knocks(label, seed=fid * 10 + (device == "iphone")), SR, subtype="PCM_16")
                rows.append({"fruit_id": f"durian_{fid:03d}", "Ripeness": label,
                             "Split": split, "has_audio": True})
                fid += 1
    pd.DataFrame(rows).to_csv(root / "final_multimodal_manifest.csv", index=False)
    return root


def test_list_audio_files_and_label_check(fake_data):
    manifest = pd.read_csv(fake_data / "final_multimodal_manifest.csv")
    df = ta.list_audio_files(fake_data, "train", manifest)
    assert len(df) == 24 and set(df["label"]) == set(ta.CLASS_ORDER)
    assert set(df["device"]) == {"android", "iphone"}
    bad = manifest.copy()
    bad.loc[0, "Ripeness"] = "Overripe"  # nhãn manifest mâu thuẫn tên file -> phải báo lỗi
    with pytest.raises(ValueError):
        ta.list_audio_files(fake_data, "train", bad)


def test_model_shapes():
    model = ta.AudioResNet18()
    x = torch.randn(2, 1, 64, 306)
    logits, feats = model(x)
    assert logits.shape == (2, 3) and feats.shape == (2, 512)
    assert torch.allclose(model.forward_features(x), feats)


def test_smoke_training_end_to_end(fake_data, tmp_path):
    out = ta.main(["--data-dir", str(fake_data), "--results-dir", str(tmp_path / "results"),
                   "--ckpt-dir", str(tmp_path / "ckpt"), "--smoke", "--batch-size", "8",
                   "--device", "cpu", "--repeats", "1"])
    ckpt = torch.load(tmp_path / "ckpt" / "smoke_best_audio_unimodal.pth", map_location="cpu", weights_only=False)
    assert {"model_state_dict", "backbone_state_dict", "head_state_dict", "config"} <= set(ckpt)
    assert ckpt["feature_dim"] == 512 and ckpt["class_order"] == ta.CLASS_ORDER

    # Backbone tách riêng nạp được vào model mới
    bb = torch.load(tmp_path / "ckpt" / "smoke_audio_backbone_teacher.pth", map_location="cpu", weights_only=False)
    fresh = ta.AudioResNet18()
    fresh.backbone.load_state_dict(bb["backbone_state_dict"])

    ev = tmp_path / "results" / "smoke_eval"
    for name in ["val_metrics.csv", "test_metrics.csv", "val_cm_sample_normalized.png",
                 "val_metrics_by_device.csv", "audio_baseline_summary.csv"]:
        assert (ev / name).exists(), name
    assert (tmp_path / "results" / "smoke_history.csv").exists()
    summary = pd.read_csv(ev / "audio_baseline_summary.csv")
    assert set(summary["split"]) == {"val", "test"} and set(summary["level"]) == {"sample", "fruit"}
    assert out["best"]["epoch"] >= 1
