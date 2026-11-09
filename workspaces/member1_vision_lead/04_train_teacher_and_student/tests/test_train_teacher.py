"""Kiểm thử toàn diện cho nhánh Multi-Modal Teacher (Dataset, Loss, Backbones, Fusion, Eval).

Chạy từ thư mục gốc dự án:
    python -m pytest workspaces/member1_vision_lead/04_train_teacher_and_student/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.image_dataset import CLAHETransform, get_eval_transforms, get_train_transforms  # noqa: E402
from src.datasets.multimodal_dataset import MultimodalDataset  # noqa: E402
from src.datasets.sensor_dataset import (  # noqa: E402
    OFFICIAL_SENSOR_COLS,
    build_sensor_table,
    load_or_fit_sensor_preprocessor,
    sensor_vectors,
)
from src.evaluators.eval_teacher import TeacherEvaluator, TeacherEvaluatorConfig  # noqa: E402
from src.losses.classification_loss import DurianClassificationLoss, compute_class_weights  # noqa: E402
from src.models.teacher_model import (  # noqa: E402
    MultiModalTeacher,
    TeacherConfig,
    build_teacher_from_checkpoint,
    teacher_checkpoint_dict,
)


def test_classification_loss_and_weights():
    labels = [0, 0, 0, 1, 1, 2]
    w = compute_class_weights(labels, num_classes=3)
    assert w.shape == (3,)
    assert w[2] > w[1] > w[0]

    # Test Cost-Sensitive Focal Loss
    crit_cs = DurianClassificationLoss(
        class_weights=w,
        label_smoothing=0.05,
        focal_gamma=2.0,
        loss_type="cost_sensitive_focal",
        cost_weight=0.5,
    )
    logits = torch.randn(4, 3, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 0])
    loss = crit_cs(logits, targets)
    loss.backward()
    assert loss.item() > 0
    assert logits.grad is not None

    # Kiểm tra tính chất Cost-Sensitive: nhầm Overripe thành Unripe (cách 2 nấc) phải bị phạt nặng hơn nhầm thành Ripe (cách 1 nấc)
    crit_pure_cost = DurianClassificationLoss(
        class_weights=None,
        label_smoothing=0.0,
        focal_gamma=0.0,
        loss_type="cost_sensitive_focal",
        cost_weight=1.0,
    )
    target_overripe = torch.tensor([2])
    logits_pred_unripe = torch.tensor([[10.0, 0.0, 0.0]])  # đoán Unripe khi thật là Overripe
    logits_pred_ripe = torch.tensor([[0.0, 10.0, 0.0]])    # đoán Ripe khi thật là Overripe
    loss_unripe = crit_pure_cost(logits_pred_unripe, target_overripe)
    loss_ripe = crit_pure_cost(logits_pred_ripe, target_overripe)
    assert loss_unripe > loss_ripe, "Nhầm Unripe <-> Overripe phải có loss cao hơn nhầm Ripe <-> Overripe"


def test_sensor_pipeline_on_real_manifest():
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "final" / "final_multimodal_manifest.csv")
    tr = build_sensor_table("train", manifest, PROJECT_ROOT / "data" / "final")
    va = build_sensor_table("valid", manifest, PROJECT_ROOT / "data" / "final")
    pre = load_or_fit_sensor_preprocessor(tr, OFFICIAL_SENSOR_COLS)
    vecs = sensor_vectors(pre, va)
    assert len(vecs) == 28
    sample_vec = next(iter(vecs.values()))
    assert sample_vec.shape == (21,)
    assert torch.isfinite(sample_vec).all()


def test_image_clahe_and_transforms():
    arr = np.random.randint(0, 255, (120, 120, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    out_train = get_train_transforms(64)(img)
    out_eval = get_eval_transforms(64)(img)
    assert out_train.shape == (3, 64, 64)
    assert out_eval.shape == (3, 64, 64)


def test_multimodal_dataset_direct_init():
    # Kiểm tra khởi tạo trực tiếp theo đúng cú pháp trong tmp.md
    ds = MultimodalDataset(
        PROJECT_ROOT / "data" / "final" / "final_multimodal_manifest.csv",
        PROJECT_ROOT / "data" / "final",
        split="valid",
        img_size=64,
        is_training=False,
    )
    assert len(ds) >= 28
    img, audio, sensor, label, fid, extras = ds[0]
    assert img.shape == (3, 64, 64)
    assert audio.shape == (1, 64, 306)
    assert sensor.shape == (21,)
    assert label in (0, 1, 2)
    assert fid.startswith("durian_")
    assert "camera_angle" in extras and "device" in extras


def test_teacher_two_phase_and_checkpoint_and_eval(tmp_path):
    cfg = TeacherConfig(
        vision_model_name="resnet18",  # dùng backbone nhẹ trong unit test để chạy tức thì trên CPU
        vision_drop_path=0.0,
        vision_dim=512,
        audio_dim=512,
        sensor_input_dim=21,
        sensor_latent_dim=32,
        proj_dim=64,
        fusion_hidden=128,
        latent_dim_mm=64,
    )
    model = MultiModalTeacher(cfg)

    # Pha 1: đóng băng backbone
    model.freeze_backbones()
    assert not any(p.requires_grad for p in model.vision_backbone.parameters())
    assert not any(p.requires_grad for p in model.audio_backbone.parameters())
    assert not any(p.requires_grad for p in model.sensor_backbone.parameters())
    assert all(p.requires_grad for p in model.fusion.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())

    # Pha 2: mở khóa toàn mạng + differential LR
    model.unfreeze_all()
    groups = model.param_groups(backbone_lr=1e-5, head_lr=5e-4)
    group_lrs = {g["name"]: g["lr"] for g in groups}
    assert group_lrs["vision"] == pytest.approx(1e-5)
    assert group_lrs["audio"] == pytest.approx(1e-5)
    assert group_lrs["sensor"] == pytest.approx(1e-5)
    assert group_lrs["fusion_head"] == pytest.approx(5e-4)

    # Forward & Zero-masking
    model.eval()
    img = torch.randn(4, 3, 64, 64)
    audio = torch.zeros(4, 1, 64, 306)  # giả lập nhánh audio bị zero-mask
    sensor = torch.randn(4, 21)
    logits, f_mm, details = model(img, audio, sensor, return_details=True)
    assert logits.shape == (4, 3)
    assert f_mm.shape == (4, 64)
    # Khi audio = 0, trọng số cổng của audio phải bằng 0
    assert torch.allclose(details["gate_weights"][:, 1], torch.zeros(4))

    # Lưu & nạp lại checkpoint Teacher
    ckpt_path = tmp_path / "best_teacher.pth"
    torch.save(teacher_checkpoint_dict(model, epoch=3, val_f1=0.85), ckpt_path)
    loaded = build_teacher_from_checkpoint(ckpt_path, device="cpu")
    logits2, f_mm2 = loaded(img, audio, sensor)
    assert torch.allclose(logits, logits2, atol=1e-6)
    assert torch.allclose(f_mm, f_mm2, atol=1e-6)


@pytest.mark.skipif(
    not (PROJECT_ROOT / "checkpoints" / "unimodal" / "image" / "best_vision_unimodal.pth").exists(),
    reason="Cần có đủ 3 checkpoint unimodal trong checkpoints/unimodal/",
)
def test_load_real_unimodal_checkpoints():
    vision_ckpt = PROJECT_ROOT / "checkpoints" / "unimodal" / "image" / "best_vision_unimodal.pth"
    audio_ckpt = PROJECT_ROOT / "checkpoints" / "unimodal" / "audio" / "audio_backbone_teacher.pth"
    sensor_ckpt = PROJECT_ROOT / "checkpoints" / "unimodal" / "sensor" / "best_sensor_unimodal.pth"

    model = MultiModalTeacher(
        vision_ckpt=vision_ckpt,
        audio_ckpt=audio_ckpt,
        sensor_ckpt=sensor_ckpt,
    )
    model.eval()
    with torch.no_grad():
        img = torch.randn(2, 3, 384, 384)
        audio = torch.randn(2, 1, 64, 306)
        sensor = torch.randn(2, 21)
        logits, f_mm = model(img, audio, sensor)
    assert logits.shape == (2, 3)
    assert f_mm.shape == (2, 256)

