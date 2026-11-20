"""Train the Vision-Only Student via Cross-Modal Knowledge Distillation (Stage 2 - CMKD).

Responsible: Leader / Member 1 (Vision Lead & Multi-Modal Framework)
Specification reference: `specification/project_specification.pdf` (Sections 1.3.3 & 1.4) and `tmp_student.md`.

Training pipeline (Stage 2):
1. Load the converged Multi-Modal Teacher (`best_teacher.pth`) and COMPLETELY FREEZE it (`eval()`, `requires_grad=False`).
2. For each sample in `MultimodalDataset` (filtered to samples with real RGB images):
   - Teacher receives `(img, audio, sensor)` -> extracts `(logits_teacher, f_mm_teacher)` inside `torch.no_grad()`.
   - Student RECEIVES ONLY `img` -> outputs `(logits_student, f_img_student, f_proj_student)`.
3. Optimize Student using the 3-component composite distillation loss (`CrossModalDistillationLoss`):
   L_total = alpha * L_CE(y, z_student)
           + beta  * L_KD(z_student, z_teacher; T)
           + gamma * L_feat(phi(f_img_student), f_mm_teacher)
4. Two-phase training strategy to protect pretrained representations:
   - Phase 1 (Warm-up Freeze, default 3 epochs): freeze Student's VisionBackbone; train only
     ProjectionHead phi(.) and Classifier to align with Teacher's f_mm space.
   - Phase 2 (Joint Fine-tuning with Differential LR): unfreeze full Student network (Backbone LR 1e-4,
     Head LR 5e-4) with Cosine Annealing and Early Stopping.
5. During Validation and Test: Student runs completely standalone, USING ONLY RGB IMAGES (`ImageDataset`),
   requiring neither Teacher, microphone, nor sensors.

Usage (from project root):
    python workspaces/member1_vision_lead/04_train_teacher_and_student/train_student.py --smoke
    python workspaces/member1_vision_lead/04_train_teacher_and_student/train_student.py --eval-test
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, cast

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from torch.utils.data import ConcatDataset, DataLoader, Subset  # noqa: E402

from src.datasets.multimodal_dataset import build_multimodal_datasets, worker_init_fn  # noqa: E402
from src.datasets.sensor_dataset import SENSOR_PREPROCESSOR_PATH, SENSOR_RESULTS_DIR  # noqa: E402
from src.losses.classification_loss import compute_class_weights  # noqa: E402
from src.losses.distillation_loss import FEAT_LOSS_TYPES, KD_WEIGHTINGS, CrossModalDistillationLoss  # noqa: E402
from src.models.student_model import (  # noqa: E402
    StudentConfig,
    VisionStudent,
    resolve_student_backbone_name,
    student_checkpoint_dict,
)
from src.models.teacher_model import (  # noqa: E402
    MultiModalTeacher,
    TeacherConfig,
    build_teacher_from_checkpoint,
)
from src.utils.common import CLASS_ORDER, first_existing, resolve_path, set_seed  # noqa: E402

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from eval_student import (  # noqa: E402
    build_vision_eval_dataset,
    evaluate_student_split,
    generate_benchmark_comparison,
)

logger = logging.getLogger("train_student")

TEACHER_CKPT_CANDIDATES = [
    "checkpoints/teachermodal/best_teacher.pth",
    "checkpoints/teacher/best_teacher.pth",
]
VISION_CKPT_CANDIDATES = [
    "checkpoints/unimodal/vision/best_convnextv2_multiview.pth",
    "checkpoints/unimodal/vision/convnextv2_multiview_backbone_256d.pth",
    "checkpoints/unimodal/vision/best_vision_unimodal.pth",
    "checkpoints/unimodal/image/best_convnextv2_multiview.pth",
    "checkpoints/unimodal/image/convnextv2_multiview_backbone_256d.pth",
    "checkpoints/unimodal/image/best_vision_unimodal.pth",
]


# ------------------------------------------------------------------------------
# 1. Command-line arguments
# ------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Multi-View Vision-Only Student via CMKD (Stage 2)")
    # Data & Paths
    p.add_argument("--manifest", default="data/final/final_multimodal_manifest.csv")
    p.add_argument("--data-dir", default="data/final")
    p.add_argument("--sensor-cache-dir", default=str(SENSOR_RESULTS_DIR))
    p.add_argument("--sensor-preprocessor", default=str(SENSOR_PREPROCESSOR_PATH))
    p.add_argument("--teacher-ckpt", default="checkpoints/teachermodal/best_teacher.pth",
                   help="Path to best_teacher.pth checkpoint trained in Stage 1B")
    p.add_argument("--init-vision-ckpt", default=None,
                   help="Default: auto-search Multi-View ConvNeXtV2 checkpoint in checkpoints/unimodal/{vision,image}/")
    p.add_argument("--no-unimodal-init", action="store_true",
                   help="Do not auto-initialize Student from unimodal Multi-View ConvNeXtV2 checkpoint")
    p.add_argument("--save-path", default="checkpoints/studentmodal/best_student.pth")
    p.add_argument("--results-dir",
                   default="workspaces/member1_vision_lead/04_train_teacher_and_student/results_student")

    # Student architecture
    p.add_argument("--student-backbone", default="convnextv2_base",
                   help="Student backbone: convnextv2_base (default, Multi-View + 256-d Proj + Gated Attention), convnextv2_tiny...")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Disable loading ImageNet pretrained weights when initializing Student")
    p.add_argument("--drop-path", type=float, default=0.2)
    p.add_argument("--head-dropout", type=float, default=0.25)
    p.add_argument("--proj-hidden-dim", type=int, default=512)

    # Distillation Loss weights (Section 1.3.3)
    p.add_argument("--alpha", type=float, default=1.0, help="Weight for L_CE (hard labels)")
    p.add_argument("--beta", type=float, default=2.0, help="Weight for L_KD (KL Divergence soft logits)")
    p.add_argument("--gamma", type=float, default=1.0, help="Weight for L_feat (latent f_mm alignment)")
    p.add_argument("--temperature", type=float, default=3.0, help="Softmax temperature T for L_KD")
    p.add_argument("--feat-loss-type", default="cosine", choices=list(FEAT_LOSS_TYPES),
                   help="Distance function L_feat between phi(f_img) and the Teacher target feature "
                        "(relational = match in-batch similarity structure instead of absolute features)")
    p.add_argument("--feat-target", default="fused", choices=["fused", "vision"],
                   help="Teacher feature aligned by L_feat: 'fused' = f_mm (contains audio/sensor info), "
                        "'vision' = Teacher's own vision-branch embedding (learnable from RGB alone)")
    p.add_argument("--kd-weighting", default="none", choices=list(KD_WEIGHTINGS),
                   help="Per-sample L_KD weighting: none | teacher_correct | teacher_confidence")
    p.add_argument("--loss-type", default="cost_sensitive_focal",
                   choices=["cost_sensitive_focal", "focal", "ce"],
                   help="Student L_CE loss type")
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--cost-weight", type=float, default=0.5)
    p.add_argument("--no-class-weights", action="store_true")

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup-epochs", type=int, default=3,
                   help="Initial epochs to freeze VisionBackbone and stabilize ProjectionHead + Classifier")
    p.add_argument("--batch-size", type=int, default=4, help="Multi-view fruit-bag batch size")
    p.add_argument("--accum-steps", type=int, default=2, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=5e-4, help="LR for ProjectionHead + Classifier")
    p.add_argument("--lr-backbone", type=float, default=1.5e-5, help="LR for VisionBackbone in Phase 2")
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay-head", type=float, default=1e-2)
    p.add_argument("--weight-decay-backbone", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience in Phase 2")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--select-metric", default="fruit_macro_f1",
                   choices=["sample_macro_f1", "fruit_macro_f1", "val_loss", "last"],
                   help="'last' = fixed-epoch training: keep the final epoch (required when Val is in --train-splits)")
    p.add_argument("--train-splits", default="train",
                   help="Comma-separated splits used for distillation, e.g. 'train' or 'train,valid' "
                        "(Teacher was trained on Train only, so its Val soft labels are genuinely out-of-sample)")
    p.add_argument("--ema-decay", type=float, default=0.0,
                   help="If > 0, keep an EMA of Student weights (e.g. 0.99) and validate/save the EMA model")
    p.add_argument("--no-tta", action="store_true",
                   help="Disable 3-view TTA (orig + hflip + 95%% zoom) during validation and final evaluation")

    # Hardware & Utilities
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default=None, help="Path to last_student.pth to resume training")
    p.add_argument("--fresh", action="store_true", help="Train from scratch at Epoch 1, ignoring last_student.pth")
    p.add_argument("--eval-test", action="store_true", help="Evaluate on independent Test set after selecting best Student")
    p.add_argument("--eval-only", action="store_true", help="Run evaluation only from --save-path")
    p.add_argument("--skip-final-eval", action="store_true")
    p.add_argument("--smoke", "--smoke-test", dest="smoke", action="store_true",
                   help="Quick 2-epoch smoke run to verify the entire pipeline")
    p.add_argument("--max-batches", type=int, default=0)
    return p.parse_args()


# ------------------------------------------------------------------------------
# 2. Utilities
# ------------------------------------------------------------------------------
def setup_logging(results_dir: Path, smoke: bool, append: bool = False) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    mode = "a" if append else "w"
    log_name = "smoke_train_student.log" if smoke else "train_student.log"
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(results_dir / log_name, mode=mode, encoding="utf-8")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.propagate = False


def amp_settings(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return False, torch.float32
    return True, (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)


def filter_rgb_indices(multimodal_ds) -> List[int]:
    """Keep only samples with real RGB images on disk (drop image = 0 samples from RGB-missing fruits)."""
    indices = [
        i for i, s in enumerate(multimodal_ds.samples)
        if bool(s.get("img_paths")) or (s.get("img_path") is not None)
    ]
    if not indices:
        logger.warning(
            "No RGB image files found in data_dir (local/smoke run without images) -> using all manifest samples."
        )
        return list(range(len(multimodal_ds)))
    return indices


class ModelEMA:
    """Exponential Moving Average of Student weights (smoother, less val-noise-dependent checkpoints)."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for ema_v, v in zip(self.module.state_dict().values(), model.state_dict().values()):
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                ema_v.copy_(v)


def _atomic_torch_save(payload: Dict, target: Path) -> None:
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(target)


def load_frozen_teacher(args, device: torch.device) -> MultiModalTeacher:
    """Load the converged Stage 1B Teacher and freeze it completely."""
    search = ([args.teacher_ckpt] if args.teacher_ckpt else []) + list(TEACHER_CKPT_CANDIDATES)
    ckpt_path = first_existing(search)
    if ckpt_path is not None:
        try:
            logger.info(f"Loading Frozen Teacher from: {ckpt_path}")
            return build_teacher_from_checkpoint(ckpt_path, device=device, eval_mode=True)
        except Exception as exc:
            if not args.smoke:
                raise RuntimeError(
                    f"Teacher checkpoint at {ckpt_path} belongs to an older architecture ({exc}). "
                    "Please re-train the Multi-Modal Teacher first (`01_train_multimodal_teacher.ipynb` / `train_teacher.py`)."
                ) from exc
            logger.warning(f"--smoke mode: existing Teacher checkpoint incompatible ({type(exc).__name__}), using smoke Teacher.")

    if args.smoke:
        logger.warning("--smoke mode: best_teacher.pth not found, initializing lightweight Teacher for pipeline test.")
        t_cfg = TeacherConfig(
            vision_model_name="resnet18",
            vision_drop_path=0.0,
            vision_dim=256,
            audio_dim=512,
            sensor_input_dim=21,
            sensor_latent_dim=32,
            proj_dim=64,
            fusion_hidden=128,
            latent_dim_mm=256,
        )
        teacher = MultiModalTeacher(t_cfg).to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        return teacher

    raise FileNotFoundError(
        f"Teacher checkpoint not found at {search}. "
        "Please ensure checkpoints/teachermodal/best_teacher.pth exists before training Student."
    )


# ------------------------------------------------------------------------------
# 3. Training (Distillation) & Validation (Vision-Only) loops
# ------------------------------------------------------------------------------
def train_one_epoch(
    student: VisionStudent,
    teacher: MultiModalTeacher,
    loader: DataLoader,
    criterion: CrossModalDistillationLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_on: bool,
    amp_dtype: torch.dtype,
    accum_steps: int,
    grad_clip: float,
    max_batches: int,
    desc: str,
    feat_target: str = "fused",
    ema: Optional[ModelEMA] = None,
) -> Dict[str, float]:
    """One distillation epoch.

    feat_target: "fused" aligns phi(f_img) with Teacher f_mm; "vision" aligns it with the Teacher's
                 vision-branch embedding (details["features"]["vision"]).
    ema: if given, updated after every optimizer step.
    """
    student.train()
    teacher.eval()
    optimizer.zero_grad(set_to_none=True)

    tot_loss = tot_ce = tot_kd = tot_feat = 0.0
    correct, n_seen, t0 = 0, 0, time.time()
    n_batches = min(len(loader), max_batches) if max_batches > 0 else len(loader)

    for step, (img, audio, sensor, label, _fid, _extra) in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        img = img.to(device, non_blocking=True)
        audio = audio.to(device, non_blocking=True)
        sensor = sensor.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        ctx = torch.autocast(device.type, dtype=amp_dtype) if amp_on else nullcontext()
        with torch.no_grad(), ctx:
            if feat_target == "vision":
                logits_t, _f_mm_t, details_t = teacher(img, audio, sensor, return_details=True)
                feat_t = details_t["features"]["vision"]
            else:
                logits_t, feat_t = teacher(img, audio, sensor)

        with ctx:
            logits_s, _f_img_s, f_proj_s = student(img, return_features=True)
            loss, parts = criterion(logits_s, logits_t, f_proj_s, feat_t, label)

        scaler.scale(loss / accum_steps).backward()
        if (step + 1) % accum_steps == 0 or (step + 1) == n_batches:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(student)

        bs = label.size(0)
        tot_loss += parts["loss_total"] * bs
        tot_ce += parts["loss_ce"] * bs
        tot_kd += parts["loss_kd"] * bs
        tot_feat += parts["loss_feat"] * bs
        n_seen += bs
        correct += (logits_s.detach().argmax(1) == label).sum().item()

        if (step + 1) % 20 == 0 or (step + 1) == n_batches:
            logger.info(
                f"  {desc} | batch {step + 1}/{n_batches} | "
                f"loss {tot_loss / n_seen:.4f} (CE {tot_ce / n_seen:.3f}, "
                f"KD {tot_kd / n_seen:.3f}, Feat {tot_feat / n_seen:.3f}) | "
                f"{time.time() - t0:.0f}s"
            )

    denom = max(n_seen, 1)
    return {
        "train_loss": tot_loss / denom,
        "train_loss_ce": tot_ce / denom,
        "train_loss_kd": tot_kd / denom,
        "train_loss_feat": tot_feat / denom,
        "train_acc": correct / denom,
    }


@torch.no_grad()
def validate_student(
    student: VisionStudent,
    loader: DataLoader,
    device: torch.device,
    amp_on: bool,
    amp_dtype: torch.dtype,
    max_batches: int = 0,
    tta: bool = False,
) -> Dict[str, float]:
    """Evaluate Student on Val USING ONLY RGB IMAGES (no Teacher, Audio, or Sensor)."""
    student.eval()
    probs_all, y_all, fids = [], [], []
    for step, batch in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        img, label, fid = batch[0].to(device, non_blocking=True), batch[1], batch[2]
        ctx = torch.autocast(device.type, dtype=amp_dtype) if amp_on else nullcontext()
        with ctx:
            logits = student(img, return_features=False, tta=tta)
        probs_all.append(F.softmax(logits.float(), dim=-1).cpu())
        y_all.append(label)
        fids.extend(fid)

    probs = torch.cat(probs_all).numpy()
    y = torch.cat(y_all).numpy()
    pred = probs.argmax(1)
    val_loss = float(F.nll_loss(torch.log(torch.from_numpy(probs).clamp_min(1e-12)), torch.from_numpy(y)))

    df = pd.DataFrame(probs, columns=[0, 1, 2])
    df["fruit_id"], df["y"] = fids, y
    fr = df.groupby("fruit_id").agg({0: "mean", 1: "mean", 2: "mean", "y": "first"})
    fr_pred = fr[[0, 1, 2]].to_numpy().argmax(1)

    return {
        "val_loss": val_loss,
        "sample_acc": float(accuracy_score(y, pred)),
        "sample_macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "fruit_acc": float(accuracy_score(fr["y"], fr_pred)),
        "fruit_macro_f1": float(f1_score(fr["y"], fr_pred, average="macro", labels=[0, 1, 2], zero_division=0)),
    }


def parse_train_splits(value: str) -> tuple:
    """'train,valid' -> ('train', 'valid'); 'val' is normalized to 'valid'."""
    splits = []
    for s in value.split(","):
        s = s.strip().lower()
        if not s:
            continue
        s = "valid" if s == "val" else s
        if s not in ("train", "valid"):
            raise ValueError(f"--train-splits only supports 'train' and 'valid', got {s!r}")
        if s not in splits:
            splits.append(s)
    if "train" not in splits:
        raise ValueError("--train-splits must include 'train'")
    return tuple(splits)


def is_better(metrics: Dict[str, float], best: Dict[str, float], key: str) -> bool:
    if not best or key == "last":
        return True
    if key == "val_loss":
        return metrics["val_loss"] < best["val_loss"] - 1e-6
    if metrics[key] > best[key] + 1e-9:
        return True
    return abs(metrics[key] - best[key]) <= 1e-9 and metrics["val_loss"] < best["val_loss"] - 1e-6


def plot_student_history(history: pd.DataFrame, warmup_epochs: int, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    axes[0].plot(history["epoch"], history["train_loss"], "o-", label="train total")
    axes[0].plot(history["epoch"], history["val_loss"], "o-", label="val NLL")
    axes[0].set_title("Total Loss (Train vs Val)")

    axes[1].plot(history["epoch"], history["train_loss_ce"], "o-", label="L_CE (Hard)")
    axes[1].plot(history["epoch"], history["train_loss_kd"], "o-", label="L_KD (Soft Logits)")
    axes[1].plot(history["epoch"], history["train_loss_feat"], "o-", label="L_feat (Latent f_mm)")
    axes[1].set_title("Distillation Loss Components")

    axes[2].plot(history["epoch"], history["sample_macro_f1"], "o-", label="val sample Macro-F1")
    axes[2].plot(history["epoch"], history["fruit_macro_f1"], "o-", label="val fruit Macro-F1")
    axes[2].set_title("Validation Macro-F1 (RGB-Only)")

    for ax in axes:
        if warmup_epochs > 0:
            ax.axvline(warmup_epochs + 0.5, color="gray", ls="--", lw=1)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Vision-Only Student (CMKD Stage 2)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------
# 4. Main
# ------------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs, args.warmup_epochs = 2, 1
        args.max_batches = args.max_batches or 3
        args.save_path = str(Path(args.save_path).with_name("smoke_student.pth"))
        args.no_pretrained = True

    results_dir = resolve_path(args.results_dir)
    save_path = resolve_path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        last_name = "smoke_last_student.pth"
    elif save_path.name == "best_student.pth":
        last_name = "last_student.pth"
    else:  # e.g. student_trainval_s43.pth -> last_student_trainval_s43.pth (one resume file per run/seed)
        last_name = f"last_{save_path.stem}.pth"
    last_path = resolve_path(args.resume) if args.resume else save_path.with_name(last_name)
    resuming = (not args.fresh) and (not args.eval_only) and last_path.is_file()

    setup_logging(results_dir, args.smoke, append=resuming)
    set_seed(args.seed)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    amp_on, amp_dtype = amp_settings(device, not args.no_amp)

    logger.info("=" * 78)
    logger.info("TRAINING VISION-ONLY STUDENT VIA CMKD (STAGE 2)")
    logger.info(f"Device: {device} | AMP: {amp_on} ({amp_dtype}) | batch {args.batch_size} x accum {args.accum_steps}")
    logger.info(f"Student backbone : {resolve_student_backbone_name(args.student_backbone)}")
    logger.info(f"Distill weights  : alpha={args.alpha} (CE) | beta={args.beta} (KD, T={args.temperature}) | gamma={args.gamma} (Feat: {args.feat_loss_type})")
    logger.info(f"Best student ckpt: {save_path}")
    logger.info("=" * 78)

    # ---- Eval-only mode (--eval-only) ------------------------------------------
    if args.eval_only:
        res_val = evaluate_student_split(
            save_path, args.manifest, args.data_dir, "valid", results_dir,
            batch_size=args.batch_size, num_workers=args.num_workers,
            image_size=args.image_size, device=device, smoke=args.smoke, seed=args.seed,
            tta=not args.no_tta,
        )
        logger.info(f"VAL  | sample F1 {res_val['metrics']['sample_macro_f1']:.4f} | fruit F1 {res_val['metrics']['fruit_macro_f1']:.4f}")
        if args.eval_test:
            res_test = evaluate_student_split(
                save_path, args.manifest, args.data_dir, "test", results_dir,
                batch_size=args.batch_size, num_workers=args.num_workers,
                image_size=args.image_size, device=device, smoke=args.smoke, seed=args.seed,
                tta=not args.no_tta,
            )
            logger.info(f"TEST | sample F1 {res_test['metrics']['sample_macro_f1']:.4f} | fruit F1 {res_test['metrics']['fruit_macro_f1']:.4f}")
            if not args.smoke:
                generate_benchmark_comparison(results_dir)
        return

    # ---- 1. Load Frozen Teacher ------------------------------------------------
    teacher = load_frozen_teacher(args, device)
    # Student projection head output must match the Teacher feature used as L_feat target
    if args.feat_target == "vision":
        latent_dim_mm = int(teacher.config.vision_dim)
    else:
        latent_dim_mm = int(teacher.config.latent_dim_mm)
    logger.info(f"L_feat target: Teacher '{args.feat_target}' feature ({latent_dim_mm}-d) | "
                f"KD weighting: {args.kd_weighting} | EMA: {args.ema_decay or 'off'} | TTA: {not args.no_tta}")

    # ---- 2. Build Datasets -----------------------------------------------------
    # Train requires MultimodalDataset (so Teacher gets img + audio + sensor), filtered to samples with real RGB images
    train_splits = parse_train_splits(args.train_splits)
    mm_datasets = build_multimodal_datasets(
        resolve_path(args.manifest),
        resolve_path(args.data_dir),
        image_size=args.image_size,
        seed=args.seed,
        splits=train_splits,
        sensor_cache_dir=resolve_path(args.sensor_cache_dir),
        sensor_preprocessor_path=resolve_path(args.sensor_preprocessor),
        train_mode_splits=train_splits,
    )
    subsets, train_labels = [], []
    for sp in train_splits:
        ds_sp = mm_datasets[sp]
        idx = filter_rgb_indices(ds_sp)
        train_labels += [ds_sp.samples[i]["label"] for i in idx]
        subsets.append(Subset(ds_sp, idx))
    train_ds = subsets[0] if len(subsets) == 1 else ConcatDataset(subsets)

    # Val uses ImageDataset only (RGB-Only), matching Student's deployment conditions
    val_ds = build_vision_eval_dataset(args.manifest, args.data_dir, split="valid", image_size=args.image_size)
    logger.info(f"[data] Train splits {train_splits} -> RGB samples: {len(train_ds)} | Val RGB samples: {len(val_ds)}")
    if "valid" in train_splits:
        logger.warning(
            "Val fruits are part of the training set: validation metrics are in-sample (monitoring only). "
            "Use --select-metric last (fixed-epoch training) and judge the model on Test only."
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", worker_init_fn=worker_init_fn,
        persistent_workers=args.num_workers > 0, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", worker_init_fn=worker_init_fn,
        persistent_workers=args.num_workers > 0, drop_last=False,
    )

    # ---- 3. Initialize VisionStudent -------------------------------------------
    init_vision_ckpt = None
    if not args.no_unimodal_init and not args.smoke:
        init_vision_ckpt = first_existing(
            ([args.init_vision_ckpt] if args.init_vision_ckpt else []) + list(VISION_CKPT_CANDIDATES)
        )
    elif args.init_vision_ckpt:
        init_vision_ckpt = first_existing([args.init_vision_ckpt])

    s_cfg = StudentConfig(
        backbone_name=args.student_backbone,
        pretrained=not args.no_pretrained,
        drop_path_rate=args.drop_path,
        num_classes=len(CLASS_ORDER),
        proj_hidden_dim=args.proj_hidden_dim,
        latent_dim_mm=latent_dim_mm,
        head_dropout=args.head_dropout,
    )
    student = VisionStudent(s_cfg, vision_ckpt=init_vision_ckpt)
    student.to(device)
    if init_vision_ckpt is not None:
        logger.info(f"Initialized Student VisionBackbone from unimodal checkpoint: {init_vision_ckpt}")
    n_params = sum(p.numel() for p in student.parameters())
    logger.info(f"Student: {n_params / 1e6:.1f}M params | F_rgb = {student.feat_dim} -> f_proj = {latent_dim_mm}")

    # ---- 4. 3-Component Distillation Loss --------------------------------------
    class_w = None if args.no_class_weights else compute_class_weights(train_labels, len(CLASS_ORDER))
    if class_w is not None:
        logger.info(f"Class weights (Train RGB): {dict(zip(CLASS_ORDER, class_w.numpy().round(3)))}")
    criterion = CrossModalDistillationLoss(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        temperature=args.temperature,
        feat_loss_type=args.feat_loss_type,
        ce_loss_type=args.loss_type,
        class_weights=class_w,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        cost_weight=args.cost_weight,
        kd_weighting=args.kd_weighting,
    ).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on and amp_dtype == torch.float16)

    config_dump = {
        "model": "vision_student",
        **vars(args),
        "init_vision_ckpt": str(init_vision_ckpt) if init_vision_ckpt else None,
        "save_path": str(save_path),
        "last_path": str(last_path),
        "device_used": str(device),
        "amp": amp_on,
        "amp_dtype": str(amp_dtype),
        "student_config": s_cfg.__dict__,
        "class_weights": None if class_w is None else class_w.tolist(),
        "n_train_rgb_samples": len(train_ds),
        "n_val_rgb_samples": len(val_ds),
    }
    cfg_file = results_dir / ("smoke_config.json" if args.smoke else "config.json")
    cfg_file.write_text(json.dumps(config_dump, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    history, best, bad_epochs, start_epoch = [], {}, 0, 1
    resume_state = None
    ema_resume_state = None
    if resuming:
        try:
            cand_state = torch.load(last_path, map_location="cpu", weights_only=False)
            student.load_state_dict(cand_state["model_state_dict"], strict=True)
            student.to(device)
            if cand_state.get("scaler_state_dict") and scaler.is_enabled():
                scaler.load_state_dict(cand_state["scaler_state_dict"])
            resume_state = cand_state
            if args.ema_decay > 0 and cand_state.get("ema_state_dict"):
                ema_resume_state = cand_state["ema_state_dict"]
            history = list(resume_state.get("history", []))
            best = dict(resume_state.get("best", {}))
            bad_epochs = int(resume_state.get("bad_epochs", 0))
            start_epoch = int(resume_state.get("epoch", 0)) + 1
            logger.info(f"Resumed from {last_path} (finished epoch {start_epoch - 1}/{args.epochs})")
        except Exception as exc:
            logger.warning(
                f"Existing checkpoint {last_path} is incompatible with current Multi-View Student architecture "
                f"({type(exc).__name__}); starting fresh training from Epoch 1."
            )
            resuming = False
            resume_state = None

    ema = ModelEMA(student, args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None and ema_resume_state is not None:
        ema.module.load_state_dict(ema_resume_state, strict=True)
    use_tta = not args.no_tta

    def eval_model() -> VisionStudent:
        """Model that is validated / saved: the EMA copy when enabled, otherwise the live Student."""
        return cast(VisionStudent, ema.module) if ema is not None else student

    phase2_epochs = max(args.epochs - args.warmup_epochs, 0)
    prefix = "smoke_" if args.smoke else ""

    def run_epoch(epoch: int, phase: str, optimizer) -> Dict[str, float]:
        tr = train_one_epoch(
            student, teacher, train_loader, criterion, optimizer, scaler,
            device, amp_on, amp_dtype, args.accum_steps, args.grad_clip,
            args.max_batches, f"{phase} ep {epoch}",
            feat_target=args.feat_target, ema=ema,
        )
        va = validate_student(eval_model(), val_loader, device, amp_on, amp_dtype, args.max_batches, tta=use_tta)
        lrs = {f"lr_{g.get('name', i)}": g["lr"] for i, g in enumerate(optimizer.param_groups)}
        row = {"epoch": epoch, "phase": phase, **tr, **va, **lrs}
        history.append(row)
        logger.info(
            f"[{phase}] Epoch {epoch}/{args.epochs} | "
            f"train loss {tr['train_loss']:.4f} (CE {tr['train_loss_ce']:.3f}, KD {tr['train_loss_kd']:.3f}, Feat {tr['train_loss_feat']:.3f}) | "
            f"val loss {va['val_loss']:.4f} | val F1 sample {va['sample_macro_f1']:.4f} / fruit {va['fruit_macro_f1']:.4f}"
        )
        return row

    def maybe_save(row: Dict[str, float]) -> bool:
        nonlocal best
        if is_better(row, best, args.select_metric):
            best = dict(row)
            _atomic_torch_save(
                student_checkpoint_dict(
                    eval_model(),
                    epoch=row["epoch"],
                    phase=row["phase"],
                    val_metrics=row,
                    is_ema=ema is not None,
                    select_metric=args.select_metric,
                    train_args=vars(args),
                ),
                save_path,
            )
            if args.select_metric == "last":
                logger.info(f"  -> Saved Student (fixed-epoch mode, epoch {row['epoch']}): {save_path}")
            else:
                logger.info(f"  -> Saved best Student ({args.select_metric} = {row[args.select_metric]:.4f}): {save_path}")
            return True
        return False

    def save_last_state(row: Dict[str, float], optimizer, scheduler=None) -> None:
        _atomic_torch_save(
            student_checkpoint_dict(
                student,
                epoch=row["epoch"],
                phase=row["phase"],
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict() if scheduler is not None else None,
                scaler_state_dict=scaler.state_dict() if scaler.is_enabled() else None,
                ema_state_dict=ema.module.state_dict() if ema is not None else None,
                best=best,
                bad_epochs=bad_epochs,
                history=history,
                select_metric=args.select_metric,
                train_args=vars(args),
            ),
            last_path,
        )
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(results_dir / f"{prefix}history.csv", index=False)
        plot_student_history(hist_df, args.warmup_epochs, results_dir / f"{prefix}training_curves.png")

    # ---- PHASE 1: WARM-UP FREEZE -----------------------------------------------
    if args.warmup_epochs > 0 and start_epoch <= args.warmup_epochs:
        logger.info(f">>> PHASE 1 - WARM-UP FREEZE (epoch {start_epoch}..{args.warmup_epochs}): freezing VisionBackbone")
        student.freeze_backbone()
        opt1 = torch.optim.AdamW(student.param_groups(0.0, args.lr, head_wd=args.weight_decay_head))
        if resume_state and resume_state.get("phase") == "phase1" and resume_state.get("optimizer_state_dict"):
            opt1.load_state_dict(resume_state["optimizer_state_dict"])
        for epoch in range(start_epoch, args.warmup_epochs + 1):
            row = run_epoch(epoch, "phase1", opt1)
            maybe_save(row)
            save_last_state(row, opt1)
        del opt1

    # ---- PHASE 2: JOINT FINE-TUNING --------------------------------------------
    p2_start = max(start_epoch, args.warmup_epochs + 1)
    if phase2_epochs > 0 and p2_start <= args.epochs and bad_epochs < args.patience:
        logger.info(
            f">>> PHASE 2 - JOINT DISTILLATION (epoch {p2_start}..{args.epochs}): "
            f"backbone LR {args.lr_backbone}, head LR {args.lr}"
        )
        student.unfreeze_all()
        opt2 = torch.optim.AdamW(
            student.param_groups(
                args.lr_backbone,
                args.lr,
                backbone_wd=args.weight_decay_backbone,
                head_wd=args.weight_decay_head,
            )
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=phase2_epochs, eta_min=args.min_lr)
        if resume_state and resume_state.get("phase") == "phase2":
            if resume_state.get("optimizer_state_dict"):
                opt2.load_state_dict(resume_state["optimizer_state_dict"])
            if resume_state.get("scheduler_state_dict"):
                sched.load_state_dict(resume_state["scheduler_state_dict"])
        for epoch in range(p2_start, args.epochs + 1):
            row = run_epoch(epoch, "phase2", opt2)
            sched.step()
            if maybe_save(row):
                bad_epochs = 0
            else:
                bad_epochs += 1
            save_last_state(row, opt2, sched)
            if bad_epochs >= args.patience:
                logger.info(f"Early stopping after {args.patience} epochs without improvement.")
                break

    if history:
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(results_dir / f"{prefix}history.csv", index=False)
        plot_student_history(hist_df, args.warmup_epochs, results_dir / f"{prefix}training_curves.png")
    if best:
        logger.info(
            f"Best epoch {best['epoch']} ({best['phase']}): "
            f"val sample F1 {best['sample_macro_f1']:.4f} | "
            f"fruit F1 {best['fruit_macro_f1']:.4f} | val loss {best['val_loss']:.4f}"
        )

    # ---- Final evaluation (RGB-Only) -------------------------------------------
    if not args.skip_final_eval:
        del student, teacher, train_loader, val_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info(">>> Independent evaluation of best Student on Val (RGB-only) ...")
        res_val = evaluate_student_split(
            save_path, args.manifest, args.data_dir, "valid", results_dir,
            batch_size=args.batch_size, num_workers=args.num_workers,
            image_size=args.image_size, device=device, smoke=args.smoke, seed=args.seed,
            tta=not args.no_tta,
        )
        logger.info(
            f"VAL  | sample Acc {res_val['metrics']['sample_accuracy']:.4f} | "
            f"sample F1 {res_val['metrics']['sample_macro_f1']:.4f} | "
            f"fruit Acc {res_val['metrics']['fruit_accuracy']:.4f} | "
            f"fruit F1 {res_val['metrics']['fruit_macro_f1']:.4f}"
        )
        if args.eval_test:
            logger.info(">>> Independent evaluation of best Student on Test (RGB-only) ...")
            res_test = evaluate_student_split(
                save_path, args.manifest, args.data_dir, "test", results_dir,
                batch_size=args.batch_size, num_workers=args.num_workers,
                image_size=args.image_size, device=device, smoke=args.smoke, seed=args.seed,
                tta=not args.no_tta,
            )
            logger.info(
                f"TEST | sample Acc {res_test['metrics']['sample_accuracy']:.4f} | "
                f"sample F1 {res_test['metrics']['sample_macro_f1']:.4f} | "
                f"fruit Acc {res_test['metrics']['fruit_accuracy']:.4f} | "
                f"fruit F1 {res_test['metrics']['fruit_macro_f1']:.4f}"
            )
            if not args.smoke:
                comp_df = generate_benchmark_comparison(results_dir)
                if comp_df is not None:
                    logger.info(f"Exported benchmark comparison table: {results_dir / 'benchmark_comparison.md'}")

    logger.info("VISION-ONLY STUDENT TRAINING COMPLETE.")


if __name__ == "__main__":
    main()

