"""Huấn luyện Vision-Only Student bằng Chưng cất tri thức Liên phương thức (Giai đoạn 2 - CMKD).

Phụ trách: Leader / Member 1 (Vision Lead & Multi-Modal Framework)
Tham chiếu đặc tả: `specification/Đặc tả đề tài.pdf` (Mục 1.3.3 & Mục 1.4) và `tmp_student.md`.

Cơ chế huấn luyện nâng cao (Giai đoạn 2):
1. Mô hình Multi-Modal Teacher (`best_teacher.pth`) được nạp và ĐÓNG BĂNG HOÀN TOÀN (`eval()`, `requires_grad=False`).
2. Với mỗi mẫu thuộc `MultimodalDataset` (lọc các mẫu có ảnh RGB thật):
   - Teacher nhận `(img, audio, sensor)` -> trích xuất `(logits_teacher, f_mm_teacher)` trong `torch.no_grad()`.
   - Student CHỈ NHẬN `img` -> sinh ra `(logits_student, f_img_student, f_proj_student)`.
3. Tối ưu hóa Student bằng hàm mất mát 3 thành phần có Trọng số Độ tin cậy & Phương thức (`CrossModalDistillationLoss`):
   - Tự động giảm trọng số L_KD và L_feat trên các mẫu mà Teacher bị khuyết Audio/Sensor hoặc dự đoán sai nhãn thật.
4. Chiến lược 2 pha chống Học vẹt danh tính quả (Fruit-Identity Overfitting) và chống sốc Gradient:
   - Pha 1 (Warm-up Freeze, mặc định 3 epoch): đóng băng VisionBackbone của Student, chỉ huấn luyện
     ProjectionHead phi(.) và Classifier.
   - Pha 2 (Joint Fine-tuning với Differential LR + Smooth Unfreeze Warmup): mở khóa toàn mạng Student
     (Backbone LR 3e-5 có ramp-up mượt 2 epoch đầu, Head LR 3e-4, DropPath 0.35, Weight Decay 1e-2).
5. Khi Validate và Test: Student hoàn toàn độc lập, CHỈ DÙNG ẢNH RGB (`ImageDataset`), kết hợp
   gộp xác suất cấp quả theo trọng số góc chụp (`angle_weighted_prob`).

Cách chạy (từ thư mục gốc dự án):
    python workspaces/member1_vision_lead/04_train_teacher_and_student/train_student.py --smoke
    python workspaces/member1_vision_lead/04_train_teacher_and_student/train_student.py --eval-test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

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
from torch.utils.data import DataLoader, Subset  # noqa: E402

from src.datasets.multimodal_dataset import build_multimodal_datasets, worker_init_fn  # noqa: E402
from src.datasets.sensor_dataset import SENSOR_PREPROCESSOR_PATH, SENSOR_RESULTS_DIR  # noqa: E402
from src.losses.classification_loss import compute_class_weights  # noqa: E402
from src.losses.distillation_loss import CrossModalDistillationLoss  # noqa: E402
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
    compute_angle_weighted_fruit_probs,
    evaluate_student_split,
    generate_benchmark_comparison,
)

logger = logging.getLogger("train_student")

TEACHER_CKPT_CANDIDATES = [
    "checkpoints/teachermodal/best_teacher.pth",
    "checkpoints/teacher/best_teacher.pth",
]


# ------------------------------------------------------------------------------
# 1. Tham số dòng lệnh
# ------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Huấn luyện Vision-Only Student bằng CMKD (Giai đoạn 2)")
    # Dữ liệu & Đường dẫn
    p.add_argument("--manifest", default="data/final/final_multimodal_manifest.csv")
    p.add_argument("--data-dir", default="data/final")
    p.add_argument("--sensor-cache-dir", default=str(SENSOR_RESULTS_DIR))
    p.add_argument("--sensor-preprocessor", default=str(SENSOR_PREPROCESSOR_PATH))
    p.add_argument("--teacher-ckpt", default="checkpoints/teachermodal/best_teacher.pth",
                   help="Đường dẫn checkpoint best_teacher.pth đã huấn luyện ở Giai đoạn 1B")
    p.add_argument("--init-vision-ckpt", default=None,
                   help="Tùy chọn: nạp khởi tạo từ best_vision_unimodal.pth nếu dùng cùng kiến trúc")
    p.add_argument("--save-path", default="checkpoints/studentmodal/best_student.pth")
    p.add_argument("--results-dir",
                   default="workspaces/member1_vision_lead/04_train_teacher_and_student/results_student")

    # Kiến trúc Student & Regularization chống Overfitting danh tính quả
    p.add_argument("--student-backbone", default="convnextv2_tiny",
                   help="Backbone của Student: convnextv2_tiny (mặc định, ~28M), convnextv2_nano, convnextv2_base, resnet18...")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Tắt tải trọng số ImageNet pretrained khi khởi tạo Student")
    p.add_argument("--drop-path", type=float, default=0.35,
                   help="Stochastic depth drop_path_rate cho VisionBackbone (tăng lên 0.35 để chống overfit 120 quả)")
    p.add_argument("--head-dropout", type=float, default=0.3)
    p.add_argument("--proj-hidden-dim", type=int, default=512)

    # Trọng số Distillation Loss & Confidence/Modality Weighting (Mục 1.3.3)
    p.add_argument("--alpha", type=float, default=1.0, help="Trọng số L_CE (nhãn thật)")
    p.add_argument("--beta", type=float, default=2.0, help="Trọng số L_KD (KL Divergence soft logits)")
    p.add_argument("--gamma", type=float, default=1.0, help="Trọng số L_feat (căn chỉnh đặc trưng ẩn f_mm)")
    p.add_argument("--temperature", type=float, default=2.5, help="Nhiệt độ làm mềm T cho L_KD")
    p.add_argument("--feat-loss-type", default="cosine", choices=["cosine", "mse", "cosine_mse"],
                   help="Hàm đo khoảng cách L_feat giữa phi(f_img) và f_mm")
    p.add_argument("--loss-type", default="cost_sensitive_focal",
                   choices=["cost_sensitive_focal", "focal", "ce"],
                   help="Loại hàm L_CE của Student")
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--cost-weight", type=float, default=0.2,
                   help="Trọng số phạt thứ tự Ordinal trong L_CE (giảm còn 0.2 để tránh dồn vào lớp giữa Ripe)")
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--no-confidence-weighting", action="store_true",
                   help="Tắt cơ chế giảm trọng số L_KD/L_feat khi Teacher khuyết phương thức hoặc đoán sai")
    p.add_argument("--missing-modality-scale", type=float, default=0.45,
                   help="Hệ số nhân giảm L_KD/L_feat khi mẫu huấn luyện bị khuyết Audio hoặc Sensor")
    p.add_argument("--wrong-teacher-scale", type=float, default=0.30,
                   help="Hệ số nhân giảm L_KD/L_feat khi Teacher dự đoán sai nhãn thật")

    # Siêu tham số huấn luyện
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup-epochs", type=int, default=3,
                   help="Số epoch đầu đóng băng backbone để ổn định ProjectionHead + Classifier")
    p.add_argument("--phase2-warmup-epochs", type=int, default=2,
                   help="Số epoch đầu của Pha 2 tăng dần LR backbone để tránh sốc gradient khi unfreeze")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4, help="LR cho ProjectionHead + Classifier")
    p.add_argument("--lr-backbone", type=float, default=3e-5, help="LR cho VisionBackbone ở Pha 2 (3e-5 chống overfit)")
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay-head", type=float, default=1e-2)
    p.add_argument("--weight-decay-backbone", type=float, default=1e-2,
                   help="Weight decay L2 cho VisionBackbone ở Pha 2 (1e-2 ngăn học vẹt danh tính quả)")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience ở Pha 2")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--select-metric", default="fruit_macro_f1",
                   choices=["sample_macro_f1", "fruit_macro_f1", "val_loss"])
    p.add_argument("--fruit-agg", default="angle_weighted_prob",
                   choices=["angle_weighted_prob", "mean_prob", "majority_vote"],
                   help="Chiến lược tổng hợp xác suất cấp quả theo góc chụp")

    # Phần cứng & Tiện ích
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default=None, help="Đường dẫn last_student.pth để tiếp tục")
    p.add_argument("--fresh", action="store_true", help="Huấn luyện mới từ Epoch 1, bỏ qua last_student.pth")
    p.add_argument("--eval-test", action="store_true", help="Chấm tập Test độc lập sau khi chốt best Student")
    p.add_argument("--eval-only", action="store_true", help="Chỉ chạy đánh giá từ --save-path")
    p.add_argument("--skip-final-eval", action="store_true")
    p.add_argument("--smoke", "--smoke-test", dest="smoke", action="store_true",
                   help="Chạy thử nhanh 2 epoch để kiểm tra toàn bộ luồng code")
    p.add_argument("--max-batches", type=int, default=0)
    return p.parse_args()


# ------------------------------------------------------------------------------
# 2. Tiện ích
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
    """Chỉ giữ các mẫu có ảnh RGB thật trên đĩa (loại bỏ các mẫu ảnh = 0 do quả thiếu RGB)."""
    indices = [i for i, s in enumerate(multimodal_ds.samples) if s.get("img_path") is not None]
    if not indices:
        logger.warning(
            "Không tìm thấy file ảnh RGB trong data_dir (chạy local/smoke chưa tải ảnh) -> dùng toàn bộ mẫu manifest."
        )
        return list(range(len(multimodal_ds)))
    return indices


def _atomic_torch_save(payload: Dict, target: Path) -> None:
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(target)


def load_frozen_teacher(args, device: torch.device) -> MultiModalTeacher:
    """Nạp Teacher đã hội tụ ở Giai đoạn 1B và đóng băng hoàn toàn."""
    search = ([args.teacher_ckpt] if args.teacher_ckpt else []) + list(TEACHER_CKPT_CANDIDATES)
    ckpt_path = first_existing(search)
    if ckpt_path is not None:
        logger.info(f"Nạp Frozen Teacher từ: {ckpt_path}")
        return build_teacher_from_checkpoint(ckpt_path, device=device, eval_mode=True)

    if args.smoke:
        logger.warning("Chế độ --smoke: không tìm thấy best_teacher.pth, khởi tạo Teacher nhẹ để kiểm thử luồng.")
        t_cfg = TeacherConfig(
            vision_model_name="resnet18",
            vision_drop_path=0.0,
            vision_dim=512,
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
        f"Không tìm thấy checkpoint Teacher tại {search}. "
        "Hãy đảm bảo đã có checkpoints/teachermodal/best_teacher.pth trước khi huấn luyện Student."
    )


# ------------------------------------------------------------------------------
# 3. Vòng lặp Train (Distillation) & Validate (Vision-Only)
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
) -> Dict[str, float]:
    student.train()
    teacher.eval()
    optimizer.zero_grad(set_to_none=True)

    tot_loss = tot_ce = tot_kd = tot_feat = tot_w = 0.0
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
            logits_t, f_mm_t = teacher(img, audio, sensor)

        with ctx:
            logits_s, _f_img_s, f_proj_s = student(img, return_features=True)
            loss, parts = criterion(
                logits_s,
                logits_t,
                f_proj_s,
                f_mm_t,
                label,
                audio=audio,
                sensor=sensor,
            )

        scaler.scale(loss / accum_steps).backward()
        if (step + 1) % accum_steps == 0 or (step + 1) == n_batches:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        bs = label.size(0)
        tot_loss += parts["loss_total"] * bs
        tot_ce += parts["loss_ce"] * bs
        tot_kd += parts["loss_kd"] * bs
        tot_feat += parts["loss_feat"] * bs
        tot_w += parts.get("distill_weight_mean", 1.0) * bs
        n_seen += bs
        correct += (logits_s.detach().argmax(1) == label).sum().item()

        if (step + 1) % 20 == 0 or (step + 1) == n_batches:
            logger.info(
                f"  {desc} | batch {step + 1}/{n_batches} | "
                f"loss {tot_loss / n_seen:.4f} (CE {tot_ce / n_seen:.3f}, "
                f"KD {tot_kd / n_seen:.3f}, Feat {tot_feat / n_seen:.3f}, w {tot_w / n_seen:.2f}) | "
                f"{time.time() - t0:.0f}s"
            )

    denom = max(n_seen, 1)
    return {
        "train_loss": tot_loss / denom,
        "train_loss_ce": tot_ce / denom,
        "train_loss_kd": tot_kd / denom,
        "train_loss_feat": tot_feat / denom,
        "train_distill_weight": tot_w / denom,
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
    fruit_agg: str = "angle_weighted_prob",
) -> Dict[str, float]:
    """Đánh giá Student trên tập Val CHỈ VỚI ẢNH RGB (không dùng Teacher, Audio hay Sensor)."""
    student.eval()
    probs_all, y_all, fids, angles = [], [], [], []
    for step, batch in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        img, label, fid = batch[0].to(device, non_blocking=True), batch[1], batch[2]
        extra = batch[3] if len(batch) > 3 and isinstance(batch[3], dict) else {}
        batch_angles = extra.get("camera_angle", [""] * len(fid))
        ctx = torch.autocast(device.type, dtype=amp_dtype) if amp_on else nullcontext()
        with ctx:
            logits = student(img, return_features=False)
        probs_all.append(F.softmax(logits.float(), dim=-1).cpu())
        y_all.append(label)
        fids.extend(fid)
        angles.extend([str(a) for a in batch_angles])

    probs = torch.cat(probs_all).numpy()
    y = torch.cat(y_all).numpy()
    pred = probs.argmax(1)
    val_loss = float(F.nll_loss(torch.log(torch.from_numpy(probs).clamp_min(1e-12)), torch.from_numpy(y)))

    df = pd.DataFrame(probs, columns=[0, 1, 2])
    df["fruit_id"], df["y"], df["camera_angle"] = fids, y, angles

    fr_y, fr_pred = [], []
    for _fid, g in df.groupby("fruit_id", sort=True):
        fr_y.append(int(g["y"].iloc[0]))
        g_probs = g[[0, 1, 2]].to_numpy(dtype=np.float64)
        if fruit_agg == "angle_weighted_prob":
            wp = compute_angle_weighted_fruit_probs(g_probs, g["camera_angle"].tolist())
        else:
            wp = g_probs.mean(axis=0)
        fr_pred.append(int(wp.argmax()))

    return {
        "val_loss": val_loss,
        "sample_acc": float(accuracy_score(y, pred)),
        "sample_macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "fruit_acc": float(accuracy_score(fr_y, fr_pred)),
        "fruit_macro_f1": float(f1_score(fr_y, fr_pred, average="macro", labels=[0, 1, 2], zero_division=0)),
    }


def is_better(metrics: Dict[str, float], best: Dict[str, float], key: str) -> bool:
    if not best:
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
    axes[0].set_title("Tổng hợp Loss (Train vs Val)")

    axes[1].plot(history["epoch"], history["train_loss_ce"], "o-", label="L_CE (Hard)")
    axes[1].plot(history["epoch"], history["train_loss_kd"], "o-", label="L_KD (Soft Logits)")
    axes[1].plot(history["epoch"], history["train_loss_feat"], "o-", label="L_feat (Latent f_mm)")
    axes[1].set_title("Các thành phần Distillation Loss")

    axes[2].plot(history["epoch"], history["sample_macro_f1"], "o-", label="val sample Macro-F1")
    axes[2].plot(history["epoch"], history["fruit_macro_f1"], "o-", label="val fruit Macro-F1")
    axes[2].set_title("Macro-F1 trên Validation (RGB-Only)")

    for ax in axes:
        if warmup_epochs > 0:
            ax.axvline(warmup_epochs + 0.5, color="gray", ls="--", lw=1)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Vision-Only Student (CMKD Giai đoạn 2)")
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
    last_name = "smoke_last_student.pth" if args.smoke else "last_student.pth"
    last_path = resolve_path(args.resume) if args.resume else save_path.with_name(last_name)
    resuming = (not args.fresh) and (not args.eval_only) and last_path.is_file()

    setup_logging(results_dir, args.smoke, append=resuming)
    set_seed(args.seed)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    amp_on, amp_dtype = amp_settings(device, not args.no_amp)

    logger.info("=" * 78)
    logger.info("HUẤN LUYỆN VISION-ONLY STUDENT BẰNG CMKD (GIAI ĐOẠN 2)")
    logger.info(f"Thiết bị: {device} | AMP: {amp_on} ({amp_dtype}) | batch {args.batch_size} x accum {args.accum_steps}")
    logger.info(f"Student backbone : {resolve_student_backbone_name(args.student_backbone)} (drop_path={args.drop_path})")
    logger.info(
        f"Distill weights  : alpha={args.alpha} (CE) | beta={args.beta} (KD, T={args.temperature}) | "
        f"gamma={args.gamma} (Feat: {args.feat_loss_type}) | conf_weight={not args.no_confidence_weighting}"
    )
    logger.info(f"Fruit aggregation: {args.fruit_agg} | Best student ckpt: {save_path}")
    logger.info("=" * 78)

    # ---- Chế độ chỉ đánh giá (--eval-only) -------------------------------------
    if args.eval_only:
        res_val = evaluate_student_split(
            save_path, args.manifest, args.data_dir, "valid", results_dir,
            batch_size=args.batch_size, num_workers=args.num_workers,
            image_size=args.image_size, device=device, smoke=args.smoke, seed=args.seed,
            fruit_aggregation_strategy=args.fruit_agg,
        )
        logger.info(f"VAL  | sample F1 {res_val['metrics']['sample_macro_f1']:.4f} | fruit F1 {res_val['metrics']['fruit_macro_f1']:.4f}")
        if args.eval_test:
            res_test = evaluate_student_split(
                save_path, args.manifest, args.data_dir, "test", results_dir,
                batch_size=args.batch_size, num_workers=args.num_workers,
                image_size=args.image_size, device=device, smoke=args.smoke, seed=args.seed,
                fruit_aggregation_strategy=args.fruit_agg,
            )
            logger.info(f"TEST | sample F1 {res_test['metrics']['sample_macro_f1']:.4f} | fruit F1 {res_test['metrics']['fruit_macro_f1']:.4f}")
            if not args.smoke:
                generate_benchmark_comparison(results_dir)
        return

    # ---- 1. Nạp Frozen Teacher -------------------------------------------------
    teacher = load_frozen_teacher(args, device)
    latent_dim_mm = int(teacher.config.latent_dim_mm)

    # ---- 2. Xây dựng Datasets --------------------------------------------------
    mm_datasets = build_multimodal_datasets(
        resolve_path(args.manifest),
        resolve_path(args.data_dir),
        image_size=args.image_size,
        seed=args.seed,
        splits=("train",),
        sensor_cache_dir=resolve_path(args.sensor_cache_dir),
        sensor_preprocessor_path=resolve_path(args.sensor_preprocessor),
    )
    full_train_ds = mm_datasets["train"]
    rgb_train_indices = filter_rgb_indices(full_train_ds)
    train_labels = [full_train_ds.samples[i]["label"] for i in rgb_train_indices]
    train_ds = Subset(full_train_ds, rgb_train_indices)

    val_ds = build_vision_eval_dataset(args.manifest, args.data_dir, split="valid", image_size=args.image_size)
    logger.info(f"[data] Train RGB samples: {len(train_ds)} | Val RGB samples: {len(val_ds)}")

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

    # ---- 3. Khởi tạo VisionStudent ---------------------------------------------
    s_cfg = StudentConfig(
        backbone_name=args.student_backbone,
        pretrained=not args.no_pretrained,
        drop_path_rate=args.drop_path,
        num_classes=len(CLASS_ORDER),
        proj_hidden_dim=args.proj_hidden_dim,
        latent_dim_mm=latent_dim_mm,
        head_dropout=args.head_dropout,
    )
    student = VisionStudent(s_cfg, vision_ckpt=resolve_path(args.init_vision_ckpt) if args.init_vision_ckpt else None)
    student.to(device)
    n_params = sum(p.numel() for p in student.parameters())
    logger.info(f"Student: {n_params / 1e6:.1f}M tham số | f_img = {student.feat_dim} -> f_proj = {latent_dim_mm}")

    # ---- 4. Hàm mất mát Chưng cất 3 thành phần ---------------------------------
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
        confidence_weighting=not args.no_confidence_weighting,
        missing_modality_scale=args.missing_modality_scale,
        wrong_teacher_scale=args.wrong_teacher_scale,
    ).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on and amp_dtype == torch.float16)

    config_dump = {
        "model": "vision_student",
        **vars(args),
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
    if resuming:
        resume_state = torch.load(last_path, map_location="cpu", weights_only=False)
        student.load_state_dict(resume_state["model_state_dict"], strict=True)
        student.to(device)
        if resume_state.get("scaler_state_dict") and scaler.is_enabled():
            scaler.load_state_dict(resume_state["scaler_state_dict"])
        history = list(resume_state.get("history", []))
        best = dict(resume_state.get("best", {}))
        bad_epochs = int(resume_state.get("bad_epochs", 0))
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        logger.info(f"Đã khôi phục từ {last_path} (xong epoch {start_epoch - 1}/{args.epochs})")

    phase2_epochs = max(args.epochs - args.warmup_epochs, 0)
    prefix = "smoke_" if args.smoke else ""

    def run_epoch(epoch: int, phase: str, optimizer) -> Dict[str, float]:
        tr = train_one_epoch(
            student, teacher, train_loader, criterion, optimizer, scaler,
            device, amp_on, amp_dtype, args.accum_steps, args.grad_clip,
            args.max_batches, f"{phase} ep {epoch}",
        )
        va = validate_student(
            student, val_loader, device, amp_on, amp_dtype,
            args.max_batches, fruit_agg=args.fruit_agg,
        )
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
                    student,
                    epoch=row["epoch"],
                    phase=row["phase"],
                    val_metrics=row,
                    select_metric=args.select_metric,
                    train_args=vars(args),
                ),
                save_path,
            )
            logger.info(f"  -> Lưu best Student ({args.select_metric} = {row[args.select_metric]:.4f}): {save_path}")
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

    # ---- PHA 1: WARM-UP FREEZE -------------------------------------------------
    if args.warmup_epochs > 0 and start_epoch <= args.warmup_epochs:
        logger.info(f">>> PHA 1 - WARM-UP FREEZE (epoch {start_epoch}..{args.warmup_epochs}): đóng băng VisionBackbone")
        student.freeze_backbone()
        opt1 = torch.optim.AdamW(student.param_groups(0.0, args.lr, head_wd=args.weight_decay_head))
        if resume_state and resume_state.get("phase") == "phase1" and resume_state.get("optimizer_state_dict"):
            opt1.load_state_dict(resume_state["optimizer_state_dict"])
        for epoch in range(start_epoch, args.warmup_epochs + 1):
            row = run_epoch(epoch, "phase1", opt1)
            maybe_save(row)
            save_last_state(row, opt1)
        del opt1

    # ---- PHA 2: JOINT FINE-TUNING (kèm Smooth Unfreeze Warmup) -----------------
    p2_start = max(start_epoch, args.warmup_epochs + 1)
    if phase2_epochs > 0 and p2_start <= args.epochs and bad_epochs < args.patience:
        logger.info(
            f">>> PHA 2 - JOINT DISTILLATION (epoch {p2_start}..{args.epochs}): "
            f"backbone LR {args.lr_backbone} (wd={args.weight_decay_backbone}), head LR {args.lr}"
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
        p2_warmup = min(max(int(args.phase2_warmup_epochs), 0), max(phase2_epochs - 1, 0))
        if p2_warmup > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                opt2, start_factor=0.25, end_factor=1.0, total_iters=p2_warmup
            )
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt2, T_max=max(phase2_epochs - p2_warmup, 1), eta_min=args.min_lr
            )
            sched = torch.optim.lr_scheduler.SequentialLR(
                opt2, schedulers=[warmup_sched, cosine_sched], milestones=[p2_warmup]
            )
        else:
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
                logger.info(f"Early stopping sau {args.patience} epoch không cải thiện.")
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

    # ---- Đánh giá cuối (RGB-Only) ----------------------------------------------
    if not args.skip_final_eval:
        del student, teacher, train_loader, val_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info(">>> Đánh giá độc lập best Student trên Val (chỉ dùng ảnh RGB) ...")
        res_val = evaluate_student_split(
            save_path, args.manifest, args.data_dir, "valid", results_dir,
            batch_size=args.batch_size, num_workers=args.num_workers,
            image_size=args.image_size, device=device, smoke=args.smoke, seed=args.seed,
            fruit_aggregation_strategy=args.fruit_agg,
        )
        logger.info(
            f"VAL  | sample Acc {res_val['metrics']['sample_accuracy']:.4f} | "
            f"sample F1 {res_val['metrics']['sample_macro_f1']:.4f} | "
            f"fruit Acc {res_val['metrics']['fruit_accuracy']:.4f} | "
            f"fruit F1 {res_val['metrics']['fruit_macro_f1']:.4f}"
        )
        if args.eval_test:
            logger.info(">>> Đánh giá độc lập best Student trên Test (chỉ dùng ảnh RGB) ...")
            res_test = evaluate_student_split(
                save_path, args.manifest, args.data_dir, "test", results_dir,
                batch_size=args.batch_size, num_workers=args.num_workers,
                image_size=args.image_size, device=device, smoke=args.smoke, seed=args.seed,
                fruit_aggregation_strategy=args.fruit_agg,
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
                    logger.info(f"Đã xuất bảng đối sánh: {results_dir / 'benchmark_comparison.md'}")

    logger.info("HOÀN TẤT HUẤN LUYỆN VISION-ONLY STUDENT.")


if __name__ == "__main__":
    main()
