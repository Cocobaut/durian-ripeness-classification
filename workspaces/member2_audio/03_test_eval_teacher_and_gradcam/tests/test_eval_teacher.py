"""Test cho eval_teacher.py: dùng mô hình giả và dữ liệu ngẫu nhiên nhỏ, chạy trên CPU.

Chạy: python -m pytest workspaces/member2_audio/03_test_eval_teacher_and_gradcam/tests -q
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_teacher as et  # noqa: E402


# ------------------------------------------------------------------
# Mô hình và dữ liệu giả
# ------------------------------------------------------------------
class FakeAudioModel(nn.Module):
    """Mô hình đơn phương thức: logits suy ra từ giá trị trung bình của 3 dải đầu vào."""

    def __init__(self, return_tuple: bool = True):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(5.0))
        self.return_tuple = return_tuple

    def forward(self, x):
        logits = self.scale * x.flatten(1)[:, :3]
        return (logits, logits.detach()) if self.return_tuple else logits


class FakeTeacher(nn.Module):
    """Teacher giả: chỉ dựa vào nhánh audio, nên che audio sẽ làm hiệu năng giảm."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(5.0))

    def forward(self, images, audios, sensors):
        logits = self.w * audios.flatten(1)[:, :3] + 0.0 * images.mean() + 0.0 * sensors.mean()
        return logits, logits.detach()


def make_inputs(labels, noise=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = noise * torch.randn(len(labels), 1, 4, 4, generator=g)
    flat = x.view(len(labels), -1)
    for i, y in enumerate(labels):
        flat[i, y] += 1.0  # tín hiệu đúng lớp nằm ở vị trí y
    return x


class AudioDictDataset(Dataset):
    def __init__(self, labels, fruit_ids, devices, flip=()):
        self.x = make_inputs(labels)
        self.labels = labels
        self.fruit_ids = fruit_ids
        self.devices = devices
        for i in flip:  # cố tình làm sai một vài mẫu
            self.x.view(len(labels), -1)[i, :3] = torch.tensor([0.0, 0.0, 0.0])
            self.x.view(len(labels), -1)[i, (labels[i] + 1) % 3] = 1.0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"x": self.x[i], "label": self.labels[i], "fruit_id": self.fruit_ids[i],
                "device": self.devices[i], "file": f"f{i}.wav"}


class MultiModalDataset(Dataset):
    def __init__(self, labels, fruit_ids):
        self.audio = make_inputs(labels)
        self.labels = labels
        self.fruit_ids = fruit_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return torch.zeros(3, 8, 8), self.audio[i], torch.zeros(5), self.labels[i], self.fruit_ids[i]


LABELS = [0, 0, 1, 1, 2, 2, 0, 1, 2, 2]
FRUITS = ["durian_001", "durian_001", "durian_002", "durian_002", "durian_003",
          "durian_003", "durian_004", "durian_005", "durian_006", "durian_006"]
DEVICES = ["android", "iphone"] * 5


def make_evaluator(tmp_path, model, loader, input_mode="audio", split="val"):
    cfg = et.TeacherEvaluatorConfig(output_dir=str(tmp_path), input_mode=input_mode, split_name=split,
                                    model_name="fake", device="cpu", num_workers=0)
    return et.TeacherEvaluator(cfg, model=model, test_loader=loader)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------
def test_sample_level_audio_dict_batch(tmp_path):
    loader = DataLoader(AudioDictDataset(LABELS, FRUITS, DEVICES, flip=[2]), batch_size=4)
    ev = make_evaluator(tmp_path, FakeAudioModel(), loader)
    df = ev.evaluate_sample_level()
    required = ["fruit_id", "camera_angle", "true_label", "pred_label", "prob_unripe", "prob_ripe",
                "prob_overripe", "is_correct"]
    assert all(c in df.columns for c in required)
    assert {"device", "file"} <= set(df.columns)
    assert len(df) == len(LABELS)
    assert np.allclose(df[["prob_unripe", "prob_ripe", "prob_overripe"]].sum(axis=1), 1.0, atol=1e-5)
    assert df["is_correct"].sum() == len(LABELS) - 1


def test_plain_logits_output_supported(tmp_path):
    loader = DataLoader(AudioDictDataset(LABELS, FRUITS, DEVICES), batch_size=3)
    ev = make_evaluator(tmp_path, FakeAudioModel(return_tuple=False), loader)
    assert ev.evaluate_sample_level()["is_correct"].all()


def test_tuple_batch_unimodal(tmp_path):
    x = make_inputs(LABELS)
    data = [(x[i], LABELS[i], i // 2) for i in range(len(LABELS))]
    ev = make_evaluator(tmp_path, FakeAudioModel(), DataLoader(data, batch_size=4))
    df = ev.evaluate_sample_level()
    assert df["fruit_id"].tolist() == [i // 2 for i in range(len(LABELS))]


def test_fruit_level_strategies(tmp_path):
    ev = make_evaluator(tmp_path, FakeAudioModel(), None)
    sample_df = pd.DataFrame({
        "fruit_id": ["a", "a", "a", "b", "b"],
        "true_label": ["Ripe"] * 3 + ["Unripe"] * 2,
        "pred_label": ["Ripe", "Overripe", "Overripe", "Unripe", "Ripe"],
        "prob_unripe": [0.1, 0.0, 0.0, 0.6, 0.3],
        "prob_ripe": [0.8, 0.45, 0.45, 0.3, 0.4],
        "prob_overripe": [0.1, 0.55, 0.55, 0.1, 0.3],
        "is_correct": [True, False, False, True, False],
    })
    mean_df = ev.evaluate_fruit_level(sample_df, "mean_prob").set_index("fruit_id")
    vote_df = ev.evaluate_fruit_level(sample_df, "majority_vote").set_index("fruit_id")
    # quả a: trung bình xác suất -> Ripe (0.567), bỏ phiếu -> Overripe (2 phiếu)
    assert mean_df.loc["a", "pred_label"] == "Ripe"
    assert vote_df.loc["a", "pred_label"] == "Overripe"
    # quả b: hòa phiếu 1-1 (Unripe vs Ripe), tổng xác suất Unripe 0.9 > Ripe 0.7 -> Unripe
    assert vote_df.loc["b", "pred_label"] == "Unripe"
    assert mean_df.loc["a", "n_samples"] == 3
    for c in ["true_label", "pred_label", "prob_unripe_mean", "prob_ripe_mean", "prob_overripe_mean", "is_correct"]:
        assert c in mean_df.columns


def test_compute_metrics_perfect_and_missing_class(tmp_path):
    ev = make_evaluator(tmp_path, FakeAudioModel(), None)
    y = np.array([0, 1, 2, 0, 1, 2])
    probs = np.eye(3)[y] * 0.9 + 0.1 / 3
    m = ev.compute_metrics(y, y, probs, prefix="sample_")
    assert m["sample_accuracy"] == 1.0 and m["sample_macro_f1"] == 1.0
    assert m["sample_roc_auc_ovr_macro"] == 1.0
    assert 0 <= m["sample_ece"] <= 1 and m["sample_brier_score"] < 0.1
    for key in ["sample_balanced_accuracy", "sample_weighted_f1", "sample_cohen_kappa", "sample_mcc",
                "sample_pr_auc_macro", "sample_log_loss", "sample_f1_overripe", "sample_support_ripe"]:
        assert key in m

    # Thiếu lớp Overripe trong y_true: không crash, ROC-AUC macro = NaN
    y2 = np.array([0, 0, 1, 1])
    p2 = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4]])
    m2 = ev.compute_metrics(y2, p2.argmax(1), p2)
    assert math.isnan(m2["roc_auc_ovr_macro"])
    assert not math.isnan(m2["macro_f1"]) and m2["support_overripe"] == 0


def test_full_pipeline_audio_outputs(tmp_path):
    loader = DataLoader(AudioDictDataset(LABELS, FRUITS, DEVICES, flip=[2, 5]), batch_size=4)
    res = et.evaluate(FakeAudioModel(), loader, tmp_path, input_mode="audio", split_name="val",
                      device="cpu", model_name="fake_audio")
    for name in ["sample_predictions.csv", "fruit_predictions.csv", "metrics.csv", "metrics.json",
                 "metrics_by_device.csv", "eval.log", "cm_sample_counts.png", "cm_sample_normalized.png",
                 "cm_fruit_counts.png", "cm_fruit_normalized.png"]:
        assert (tmp_path / f"val_{name}").exists(), name
    assert not (tmp_path / "val_ablation.csv").exists()  # đơn phương thức: không ablation
    long = pd.read_csv(tmp_path / "val_metrics.csv")
    assert set(long.columns) == {"model", "split", "level", "metric", "value"}
    assert set(long["level"]) == {"sample", "fruit"}
    assert res["n_fruits"] == 6 and res["ablation"] == {}
    js = json.loads((tmp_path / "val_metrics.json").read_text(encoding="utf-8"))
    assert js["metrics"]["sample_accuracy"] == pytest.approx(0.8)


def test_val_and_test_do_not_overwrite(tmp_path):
    loader = DataLoader(AudioDictDataset(LABELS, FRUITS, DEVICES), batch_size=4)
    et.evaluate(FakeAudioModel(), loader, tmp_path, split_name="val", device="cpu")
    et.evaluate(FakeAudioModel(), loader, tmp_path, split_name="test", device="cpu")
    assert (tmp_path / "val_metrics.csv").exists() and (tmp_path / "test_metrics.csv").exists()


def test_multimodal_ablation(tmp_path):
    loader = DataLoader(MultiModalDataset(LABELS, FRUITS), batch_size=4)
    ev = make_evaluator(tmp_path, FakeTeacher(), loader, input_mode="multimodal", split="test")
    res = ev.run_full_evaluation_pipeline()
    abl = res["ablation"]
    assert set(abl) == {"full", "wo_audio", "wo_sensor", "wo_vision"}
    assert abl["full"]["sample_macro_f1"] == 1.0
    assert abl["wo_audio"]["delta_sample_macro_f1"] > 0.3  # che audio thì giảm mạnh
    assert abl["wo_sensor"]["delta_sample_macro_f1"] == pytest.approx(0.0)
    assert (tmp_path / "test_ablation.csv").exists()


def test_load_checkpoint(tmp_path):
    model = FakeAudioModel()
    with torch.no_grad():
        model.scale.fill_(7.0)
    ckpt = tmp_path / "m.pth"
    torch.save({"model_state_dict": {f"module.{k}": v for k, v in model.state_dict().items()}}, ckpt)
    fresh = FakeAudioModel()
    cfg = et.TeacherEvaluatorConfig(output_dir=str(tmp_path), input_mode="audio", device="cpu",
                                    teacher_checkpoint_path=str(ckpt))
    loaded = et.TeacherEvaluator(cfg, model=fresh).load_teacher_model()
    assert loaded.scale.item() == pytest.approx(7.0)
    assert not any(p.requires_grad for p in loaded.parameters())


def test_load_without_model_raises(tmp_path):
    cfg = et.TeacherEvaluatorConfig(output_dir=str(tmp_path), device="cpu")
    with pytest.raises(RuntimeError):
        et.TeacherEvaluator(cfg).load_teacher_model()


def test_invalid_input_mode(tmp_path):
    cfg = et.TeacherEvaluatorConfig(output_dir=str(tmp_path), input_mode="text", device="cpu")
    with pytest.raises(ValueError):
        et.TeacherEvaluator(cfg)


def test_gradcam_and_error_analyzer_still_importable():
    # Phần Tuần 4 vẫn giữ khung, chỉ cần import được
    assert hasattr(et, "TeacherGradCAMVisualizer") and hasattr(et, "TeacherErrorAnalyzer")
