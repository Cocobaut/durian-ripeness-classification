"""Tests for the Vision-Only Student pipeline (CMKD Stage 2).

Covers `train_student.py` (frozen Teacher loading, distillation train loop, RGB-only validation,
best-checkpoint selection, smoke end-to-end run), `eval_student.py` (eval dataset, split evaluation,
benchmark table) and `CrossModalDistillationLoss` edge cases.

All models use the lightweight `resnet18` backbone without pretrained weights so tests run on CPU.

Run from project root:
    python -m pytest workspaces/member1_vision_lead/04_train_teacher_and_student/tests/test_train_student.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

THIS_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = THIS_DIR.parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
for p in (PROJECT_ROOT, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load_script(name: str):
    """Import a script from SCRIPT_DIR by path (the project root has other files with the same names)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


eval_student = _load_script("eval_student")
train_student = _load_script("train_student")

from src.losses.distillation_loss import CrossModalDistillationLoss  # noqa: E402
from src.models.student_model import (  # noqa: E402
    StudentConfig,
    VisionStudent,
    build_student_from_checkpoint,
    resolve_student_backbone_name,
    student_checkpoint_dict,
)
from src.models.teacher_model import MultiModalTeacher, TeacherConfig, teacher_checkpoint_dict  # noqa: E402

MANIFEST = PROJECT_ROOT / "data" / "final" / "final_multimodal_manifest.csv"
DATA_DIR = PROJECT_ROOT / "data" / "final"
IMG = 64
LATENT = 64

needs_manifest = pytest.mark.skipif(not MANIFEST.is_file(), reason="Requires data/final/final_multimodal_manifest.csv")


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def tiny_teacher() -> MultiModalTeacher:
    cfg = TeacherConfig(
        vision_model_name="resnet18",
        vision_drop_path=0.0,
        vision_dim=256,
        audio_dim=512,
        sensor_input_dim=21,
        sensor_latent_dim=32,
        proj_dim=64,
        fusion_hidden=128,
        latent_dim_mm=LATENT,
    )
    teacher = MultiModalTeacher(cfg).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def tiny_student(latent_dim_mm: int = LATENT) -> VisionStudent:
    return VisionStudent(StudentConfig(
        backbone_name="resnet18",
        pretrained=False,
        drop_path_rate=0.0,
        num_classes=3,
        proj_hidden_dim=128,
        latent_dim_mm=latent_dim_mm,
        head_dropout=0.0,
    ))


class FakeMultimodalDataset(Dataset):
    """Mimics MultimodalDataset items: (img_bag, audio, sensor, label, fruit_id, extras)."""

    def __init__(self, n: int = 6, views: int = 2):
        g = torch.Generator().manual_seed(0)
        self.imgs = torch.randn(n, views, 3, IMG, IMG, generator=g)
        self.audio = torch.randn(n, 1, 64, 306, generator=g)
        self.sensor = torch.randn(n, 21, generator=g)
        self.labels = [i % 3 for i in range(n)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (self.imgs[i], self.audio[i], self.sensor[i], self.labels[i],
                f"durian_{i:03d}", {"camera_angle": "TOP", "device": "phone"})


class FakeImageDataset(Dataset):
    """Mimics ImageDataset (multiview) items: (img_bag, label, fruit_id, extras)."""

    def __init__(self, fruit_ids, labels, views: int = 2):
        g = torch.Generator().manual_seed(1)
        self.fruit_ids, self.labels = list(fruit_ids), list(labels)
        self.imgs = torch.randn(len(labels), views, 3, IMG, IMG, generator=g)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.imgs[i], self.labels[i], self.fruit_ids[i], {"camera_angle": "MULTI_VIEW"}


def make_criterion(**kw) -> CrossModalDistillationLoss:
    base: Dict[str, Any] = dict(alpha=1.0, beta=2.0, gamma=1.0, temperature=3.0, feat_loss_type="cosine",
                                ce_loss_type="ce", label_smoothing=0.0)
    base.update(kw)
    return CrossModalDistillationLoss(**base)


def run_epoch(student, teacher, loader, optimizer, accum_steps=1, max_batches=0):
    return train_student.train_one_epoch(
        student, teacher, loader, make_criterion(), optimizer,
        torch.amp.GradScaler("cuda", enabled=False), torch.device("cpu"),
        False, torch.float32, accum_steps, 1.0, max_batches, "test",
    )


def clone_params(module):
    return [p.detach().clone() for p in module.parameters()]


def params_equal(module, snapshot) -> bool:
    return all(torch.equal(p, s) for p, s in zip(module.parameters(), snapshot))


# ------------------------------------------------------------------------------
# 1. Distillation loss edge cases
# ------------------------------------------------------------------------------
def test_distillation_loss_invalid_args():
    with pytest.raises(ValueError):
        make_criterion(feat_loss_type="l1")
    with pytest.raises(ValueError):
        make_criterion(temperature=0.0)


def test_kd_loss_zero_when_student_matches_teacher():
    crit = make_criterion()
    logits = torch.randn(5, 3)
    assert crit.compute_kd_loss(logits, logits.clone()).item() == pytest.approx(0.0, abs=1e-6)
    assert crit.compute_kd_loss(logits, torch.randn(5, 3) * 5).item() > 0


@pytest.mark.parametrize("feat_type", ["cosine", "mse", "cosine_mse"])
def test_feat_loss_zero_when_aligned_and_scale_invariant(feat_type):
    crit = make_criterion(feat_loss_type=feat_type)
    f = torch.randn(4, LATENT)
    # Same direction, different scale -> L_feat == 0 (both cosine and L2-normalized MSE)
    assert crit.compute_feat_loss(f * 3.0, f).item() == pytest.approx(0.0, abs=1e-5)
    assert crit.compute_feat_loss(-f, f).item() > 0


def test_feat_loss_ignores_zero_teacher_features_and_shape_mismatch():
    crit = make_criterion()
    s = torch.randn(4, LATENT, requires_grad=True)
    t = torch.randn(4, LATENT)
    t[2:] = 0.0  # samples where teacher f_mm is all-zero must be masked out
    masked = crit.compute_feat_loss(s, t)
    expected = crit.compute_feat_loss(s[:2], t[:2])
    assert masked.item() == pytest.approx(expected.item(), abs=1e-6)

    all_zero = crit.compute_feat_loss(s, torch.zeros(4, LATENT))
    assert all_zero.item() == 0.0
    all_zero.backward()  # still differentiable (graph kept)

    with pytest.raises(ValueError):
        crit.compute_feat_loss(torch.randn(4, LATENT), torch.randn(4, LATENT + 1))


def test_distillation_loss_weights_and_no_grad_to_teacher():
    logits_s = torch.randn(4, 3, requires_grad=True)
    logits_t = torch.randn(4, 3, requires_grad=True)
    f_s = torch.randn(4, LATENT, requires_grad=True)
    f_t = torch.randn(4, LATENT, requires_grad=True)
    y = torch.tensor([0, 1, 2, 1])

    crit = make_criterion(alpha=1.0, beta=2.0, gamma=0.5)
    loss, parts = crit(logits_s, logits_t, f_s, f_t, y)
    combo = 1.0 * parts["loss_ce"] + 2.0 * parts["loss_kd"] + 0.5 * parts["loss_feat"]
    assert parts["loss_total"] == pytest.approx(combo, rel=1e-5)

    loss.backward()
    assert logits_s.grad is not None and f_s.grad is not None
    assert logits_t.grad is None and f_t.grad is None, "Teacher outputs must be detached"

    # Disabled components contribute exactly 0
    _, parts0 = make_criterion(beta=0.0, gamma=0.0)(logits_s, logits_t, f_s, f_t, y)
    assert parts0["loss_kd"] == 0.0 and parts0["loss_feat"] == 0.0
    assert parts0["loss_total"] == pytest.approx(parts0["loss_ce"], rel=1e-6)


# ------------------------------------------------------------------------------
# 2. Student model behaviour used by train_student.py
# ------------------------------------------------------------------------------
def test_backbone_alias_resolution():
    assert resolve_student_backbone_name("convnextv2_base") == "convnextv2_base.fcmae_ft_in1k"
    assert resolve_student_backbone_name(" ConvNeXtV2_Tiny ") == "convnextv2_tiny.fcmae_ft_in1k"
    assert resolve_student_backbone_name("resnet18") == "resnet18"
    assert resolve_student_backbone_name("some_custom_model") == "some_custom_model"


def test_student_frozen_backbone_stays_in_eval_mode():
    student = tiny_student()
    student.freeze_backbone()
    student.train()
    assert student.training
    assert not student.vision_backbone.training, "Frozen backbone must stay in eval() (BN/Dropout fixed)"
    assert student.projection_head.training and student.classifier.training

    groups = student.param_groups(0.0, 5e-4)
    assert [g["name"] for g in groups] == ["student_head"], "Phase 1 optimizer must not contain backbone params"

    student.unfreeze_all()
    student.train()
    assert student.vision_backbone.training
    assert [g["name"] for g in student.param_groups(1e-5, 5e-4)] == ["backbone", "student_head"]


def test_student_accepts_single_view_4d_input():
    student = tiny_student().eval()
    with torch.no_grad():
        logits = student(torch.randn(2, 3, IMG, IMG))
    assert logits.shape == (2, 3)


# ------------------------------------------------------------------------------
# 3. train_student.py utilities
# ------------------------------------------------------------------------------
def test_amp_settings_disabled_on_cpu():
    assert train_student.amp_settings(torch.device("cpu"), True) == (False, torch.float32)
    assert train_student.amp_settings(torch.device("cuda"), False) == (False, torch.float32)


def test_filter_rgb_indices():
    class DS:
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

    ds = DS([{"img_paths": ["a.jpg"]}, {"img_paths": []}, {"img_path": "b.jpg"}, {"img_path": None}])
    assert train_student.filter_rgb_indices(ds) == [0, 2]
    # No RGB at all -> fall back to every sample instead of an empty training set
    ds_empty = DS([{"img_paths": []}, {}])
    assert train_student.filter_rgb_indices(ds_empty) == [0, 1]


def test_is_better_selection_logic():
    best = {"fruit_macro_f1": 0.80, "val_loss": 0.50}
    f = train_student.is_better
    assert f({"fruit_macro_f1": 0.1, "val_loss": 9.0}, {}, "fruit_macro_f1"), "Empty best -> always save"
    assert f({"fruit_macro_f1": 0.85, "val_loss": 0.90}, best, "fruit_macro_f1")
    assert not f({"fruit_macro_f1": 0.75, "val_loss": 0.10}, best, "fruit_macro_f1")
    # Tie on metric -> lower val_loss breaks the tie
    assert f({"fruit_macro_f1": 0.80, "val_loss": 0.40}, best, "fruit_macro_f1")
    assert not f({"fruit_macro_f1": 0.80, "val_loss": 0.60}, best, "fruit_macro_f1")
    # val_loss selection: lower is better
    assert f({"fruit_macro_f1": 0.0, "val_loss": 0.30}, best, "val_loss")
    assert not f({"fruit_macro_f1": 1.0, "val_loss": 0.50}, best, "val_loss")


def test_atomic_torch_save(tmp_path):
    target = tmp_path / "ckpt.pth"
    train_student._atomic_torch_save({"a": torch.ones(2)}, target)
    assert target.is_file()
    assert not target.with_suffix(".pth.tmp").exists()
    assert torch.equal(torch.load(target)["a"], torch.ones(2))


def test_plot_student_history(tmp_path):
    hist = pd.DataFrame([
        {"epoch": e, "train_loss": 1.0 / e, "val_loss": 1.2 / e, "train_loss_ce": 0.5, "train_loss_kd": 0.3,
         "train_loss_feat": 0.2, "sample_macro_f1": 0.1 * e, "fruit_macro_f1": 0.12 * e}
        for e in (1, 2, 3)
    ])
    out = tmp_path / "curves.png"
    train_student.plot_student_history(hist, warmup_epochs=1, path=out)
    assert out.is_file() and out.stat().st_size > 0


# ------------------------------------------------------------------------------
# 4. Frozen Teacher loading
# ------------------------------------------------------------------------------
def _teacher_args(**kw):
    from types import SimpleNamespace
    base = dict(teacher_ckpt=None, smoke=False)
    base.update(kw)
    return SimpleNamespace(**base)


def test_load_frozen_teacher_from_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(train_student, "TEACHER_CKPT_CANDIDATES", [])
    ckpt = tmp_path / "best_teacher.pth"
    torch.save(teacher_checkpoint_dict(tiny_teacher(), epoch=1, val_f1=0.5), ckpt)

    teacher = train_student.load_frozen_teacher(_teacher_args(teacher_ckpt=str(ckpt)), torch.device("cpu"))
    assert not teacher.training
    assert not any(p.requires_grad for p in teacher.parameters())
    assert teacher.config.latent_dim_mm == LATENT


def test_load_frozen_teacher_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(train_student, "TEACHER_CKPT_CANDIDATES", [])
    with pytest.raises(FileNotFoundError):
        train_student.load_frozen_teacher(
            _teacher_args(teacher_ckpt=str(tmp_path / "nope.pth")), torch.device("cpu"))


def test_load_frozen_teacher_incompatible_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(train_student, "TEACHER_CKPT_CANDIDATES", [])
    bad = tmp_path / "old_teacher.pth"
    torch.save({"teacher_config": {}, "model_state_dict": {"garbage": torch.zeros(1)}}, bad)

    with pytest.raises(RuntimeError, match="older architecture"):
        train_student.load_frozen_teacher(_teacher_args(teacher_ckpt=str(bad)), torch.device("cpu"))

    # In --smoke mode an incompatible checkpoint falls back to a lightweight frozen Teacher
    teacher = train_student.load_frozen_teacher(_teacher_args(teacher_ckpt=str(bad), smoke=True), torch.device("cpu"))
    assert not any(p.requires_grad for p in teacher.parameters())
    assert teacher.config.latent_dim_mm == 256


# ------------------------------------------------------------------------------
# 5. Distillation training loop
# ------------------------------------------------------------------------------
def test_train_one_epoch_updates_student_only():
    torch.manual_seed(0)
    teacher, student = tiny_teacher(), tiny_student()
    loader = DataLoader(FakeMultimodalDataset(n=6), batch_size=2, shuffle=False)
    opt = torch.optim.AdamW(student.param_groups(1e-4, 1e-3))

    t_before, s_before = clone_params(teacher), clone_params(student)
    stats = run_epoch(student, teacher, loader, opt)

    assert set(stats) == {"train_loss", "train_loss_ce", "train_loss_kd", "train_loss_feat", "train_acc"}
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())
    assert stats["train_loss"] > 0 and 0.0 <= stats["train_acc"] <= 1.0
    assert params_equal(teacher, t_before), "Frozen Teacher must not change during distillation"
    assert not params_equal(student, s_before), "Student must be updated"
    assert not teacher.training


def test_train_one_epoch_phase1_keeps_backbone_fixed():
    torch.manual_seed(0)
    teacher, student = tiny_teacher(), tiny_student()
    student.freeze_backbone()
    loader = DataLoader(FakeMultimodalDataset(n=4), batch_size=2)
    opt = torch.optim.AdamW(student.param_groups(0.0, 1e-3))

    bb_before = clone_params(student.vision_backbone)
    bn_before = {k: v.clone() for k, v in student.vision_backbone.state_dict().items() if "running_mean" in k}
    head_before = clone_params(student.projection_head)
    run_epoch(student, teacher, loader, opt)

    assert params_equal(student.vision_backbone, bb_before)
    for k, v in student.vision_backbone.state_dict().items():
        if k in bn_before:
            assert torch.equal(v, bn_before[k]), f"BatchNorm stats changed in frozen backbone: {k}"
    assert not params_equal(student.projection_head, head_before)


def test_train_one_epoch_respects_max_batches_and_accumulation(monkeypatch):
    teacher, student = tiny_teacher(), tiny_student()
    loader = DataLoader(FakeMultimodalDataset(n=10), batch_size=2)  # 5 batches
    opt = torch.optim.AdamW(student.param_groups(1e-4, 1e-3))

    steps = {"n": 0}
    orig_step = opt.step

    def counting_step(*a, **k):
        steps["n"] += 1
        return orig_step(*a, **k)

    monkeypatch.setattr(opt, "step", counting_step)

    # 5 batches, accum 2 -> optimizer steps at batch 2, 4 and the final batch 5
    run_epoch(student, teacher, loader, opt, accum_steps=2)
    assert steps["n"] == 3

    # max_batches=3, accum 2 -> steps at batch 2 and the last allowed batch 3
    steps["n"] = 0
    run_epoch(student, teacher, loader, opt, accum_steps=2, max_batches=3)
    assert steps["n"] == 2


# ------------------------------------------------------------------------------
# 6. RGB-only validation
# ------------------------------------------------------------------------------
def test_validate_student_metrics_and_fruit_aggregation():
    student = tiny_student()
    # 3 fruits, 2 samples each -> fruit-level aggregation over 3 groups
    fids = ["durian_001", "durian_001", "durian_002", "durian_002", "durian_003", "durian_003"]
    labels = [0, 0, 1, 1, 2, 2]
    loader = DataLoader(FakeImageDataset(fids, labels), batch_size=4)
    m = train_student.validate_student(student, loader, torch.device("cpu"), False, torch.float32)

    assert set(m) == {"val_loss", "sample_acc", "sample_macro_f1", "fruit_acc", "fruit_macro_f1"}
    assert m["val_loss"] > 0
    for k in ("sample_acc", "sample_macro_f1", "fruit_acc", "fruit_macro_f1"):
        assert 0.0 <= m[k] <= 1.0
    assert not student.training


def test_validate_student_perfect_predictions():
    labels = [0, 1, 2, 1]

    class Oracle(torch.nn.Module):
        """Returns one-hot-ish logits matching the label order of the batch."""

        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, img, return_features=False):
            out = torch.full((img.shape[0], 3), -20.0)
            for r in range(img.shape[0]):
                out[r, labels[self.calls + r]] = 20.0
            self.calls += img.shape[0]
            return out

    loader = DataLoader(FakeImageDataset([f"durian_{i}" for i in range(4)], labels), batch_size=2)
    m = train_student.validate_student(Oracle(), loader, torch.device("cpu"), False, torch.float32)
    assert m["sample_acc"] == 1.0 and m["fruit_macro_f1"] == 1.0
    assert m["val_loss"] < 1e-3


# ------------------------------------------------------------------------------
# 7. eval_student.py
# ------------------------------------------------------------------------------
def test_manifest_fallback_dataset():
    ds = eval_student._ManifestFallbackImageDataset(["durian_001", "durian_002"], [0, 2], image_size=IMG)
    assert len(ds) == 2
    img, y, fid, extras = ds[1]
    assert img.shape == (1, 3, IMG, IMG)
    assert (y, fid) == (2, "durian_002")
    assert extras["camera_angle"] == "MULTI_VIEW"


@needs_manifest
def test_build_vision_eval_dataset_valid_split():
    ds = eval_student.build_vision_eval_dataset(MANIFEST, DATA_DIR, split="valid", image_size=IMG)
    assert len(ds) > 0
    img, y, fid, _ = ds[0]
    assert img.ndim == 4 and img.shape[1:] == (3, IMG, IMG)
    assert y in (0, 1, 2)
    assert fid.startswith("durian_")
    with pytest.raises(ValueError):
        eval_student.build_vision_eval_dataset(MANIFEST, DATA_DIR, split="bogus", image_size=IMG)


@needs_manifest
def test_evaluate_student_split_from_checkpoint(tmp_path):
    student = tiny_student().eval()
    ckpt = tmp_path / "best_student.pth"
    torch.save(student_checkpoint_dict(student, epoch=1), ckpt)

    out_dir = tmp_path / "results"
    res = eval_student.evaluate_student_split(
        ckpt, MANIFEST, DATA_DIR, "valid", out_dir,
        batch_size=2, num_workers=0, image_size=IMG, device="cpu", smoke=True,
    )
    m = res["metrics"]
    for k in ("sample_accuracy", "sample_macro_f1", "fruit_accuracy", "fruit_macro_f1"):
        assert 0.0 <= m[k] <= 1.0
    assert res["input_mode"] == "vision" and res["ablation"] == {}
    assert (out_dir / "smoke_val_metrics.json").is_file()
    assert (out_dir / "smoke_val_sample_predictions.csv").is_file()

    # The reloaded checkpoint must give identical logits to the in-memory model
    loaded = build_student_from_checkpoint(ckpt)
    x = torch.randn(2, 2, 3, IMG, IMG)
    with torch.no_grad():
        assert torch.allclose(student(x), loaded(x), atol=1e-6)


def test_generate_benchmark_comparison(tmp_path):
    fake_root = tmp_path / "project"
    student_dir = tmp_path / "results_student"
    student_dir.mkdir()

    # No result files at all -> None, nothing written
    assert eval_student.generate_benchmark_comparison(student_dir, project_root=fake_root) is None

    metrics = {"sample_accuracy": 0.9, "sample_macro_f1": 0.88, "fruit_accuracy": 0.95, "fruit_macro_f1": 0.93}
    (student_dir / "test_metrics.json").write_text(json.dumps({"metrics": metrics}), encoding="utf-8")
    t_dir = fake_root / "workspaces" / "member1_vision_lead" / "04_train_teacher_and_student" / "results_teacher"
    t_dir.mkdir(parents=True)
    (t_dir / "test_metrics.json").write_text(json.dumps({"metrics": metrics}), encoding="utf-8")

    df = eval_student.generate_benchmark_comparison(student_dir, project_root=fake_root)
    assert df is not None and len(df) == 2
    assert df["Model"].str.contains("Teacher").any() and df["Model"].str.contains("Student").any()
    md = (student_dir / "benchmark_comparison.md").read_text(encoding="utf-8")
    assert "93.00%" in md
    assert (student_dir / "benchmark_comparison.csv").is_file()


# ------------------------------------------------------------------------------
# 8. End-to-end smoke run of train_student.main()
# ------------------------------------------------------------------------------
@needs_manifest
def test_train_student_main_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(train_student, "TEACHER_CKPT_CANDIDATES", [])
    teacher_ckpt = tmp_path / "best_teacher.pth"
    torch.save(teacher_checkpoint_dict(tiny_teacher(), epoch=1, val_f1=0.5), teacher_ckpt)

    results_dir = tmp_path / "results_student"
    save_path = tmp_path / "ckpt" / "best_student.pth"
    monkeypatch.setattr(sys, "argv", [
        "train_student.py", "--smoke",
        "--teacher-ckpt", str(teacher_ckpt),
        "--student-backbone", "resnet18",
        "--proj-hidden-dim", "64",
        "--image-size", str(IMG),
        "--batch-size", "2",
        "--accum-steps", "1",
        "--num-workers", "0",
        "--device", "cpu",
        "--no-amp",
        "--fresh",
        "--save-path", str(save_path),
        "--results-dir", str(results_dir),
    ])
    train_student.main()

    best = save_path.with_name("smoke_student.pth")
    last = save_path.with_name("smoke_last_student.pth")
    assert best.is_file() and last.is_file()

    hist = pd.read_csv(results_dir / "smoke_history.csv")
    assert list(hist["epoch"]) == [1, 2]
    assert list(hist["phase"]) == ["phase1", "phase2"]
    assert (results_dir / "smoke_training_curves.png").is_file()
    assert (results_dir / "smoke_config.json").is_file()
    assert (results_dir / "smoke_val_metrics.json").is_file()

    ckpt = torch.load(best, map_location="cpu", weights_only=False)
    assert ckpt["latent_dim_mm"] == LATENT, "Student projection must match the Teacher's f_mm dim"
    assert ckpt["select_metric"] == "fruit_macro_f1"
    student = build_student_from_checkpoint(best)
    with torch.no_grad():
        assert student(torch.randn(1, 2, 3, IMG, IMG)).shape == (1, 3)
