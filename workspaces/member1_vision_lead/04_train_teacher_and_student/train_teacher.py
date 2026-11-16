"""Huấn luyện Multi-Modal Teacher (Giai đoạn 1B của CMKD) - chiến lược 2 pha.

Phụ trách: Leader / Member 1 (Vision Lead & Multi-Modal Framework)
Tham chiếu đặc tả: Mục 1.3.2.

Pha 1 - Warm-up Freeze (mặc định 5 epoch):
    Đóng băng 3 backbone (đã hội tụ ở Giai đoạn 1A), chỉ học Fusion + Classifier (LR 1e-3).
Pha 2 - Joint Fine-tuning với Differential LR:
    Mở khóa toàn mạng. Backbone LR 1e-5, Fusion + Head LR 5e-4, Cosine Annealing, Early Stopping.

Loss: Class-Weighted Cross-Entropy + Label Smoothing (src/losses/classification_loss.py).
Checkpoint unimodal GỐC chỉ được ĐỌC, không bị ghi đè. Teacher (3 backbone đã tinh chỉnh + Fusion + Head)
được lưu riêng vào checkpoints/teachermodal/best_teacher.pth.

Cách chạy (LUÔN từ thư mục gốc repo):
    python workspaces/member1_vision_lead/04_train_teacher_and_student/train_teacher.py --smoke     # chạy thử
    python workspaces/member1_vision_lead/04_train_teacher_and_student/train_teacher.py            # train + chấm Val
    python workspaces/member1_vision_lead/04_train_teacher_and_student/train_teacher.py --eval-test # + chấm Test 1 lần

Đầu ra:
    checkpoints/teachermodal/best_teacher.pth
    workspaces/member1_vision_lead/04_train_teacher_and_student/results/
        config.json, train.log, history.csv, training_curves.png,
        val_* / test_* (sinh bởi eval_teacher.py của Member 2: metrics, predictions, confusion matrix, ablation)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional

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
from src.losses.classification_loss import DurianClassificationLoss, compute_class_weights  # noqa: E402
from src.models.teacher_model import (  # noqa: E402
    MODALITIES,
    MultiModalTeacher,
    TeacherConfig,
    build_teacher_from_checkpoint,
    teacher_checkpoint_dict,
)
from src.utils.common import CLASS_ORDER, first_existing, resolve_path, set_seed  # noqa: E402

logger = logging.getLogger("train_teacher")

VISION_CKPT_CANDIDATES = [
    "checkpoints/unimodal/vision/best_vision_unimodal.pth",  # tên thư mục trên Google Drive
    "checkpoints/unimodal/image/best_vision_unimodal.pth",   # tên thư mục local cũ
]
AUDIO_CKPT_CANDIDATES = [
    "checkpoints/unimodal/audio/audio_backbone_teacher.pth",
    "checkpoints/unimodal/audio/best_audio_unimodal.pth",
]
SENSOR_CKPT_CANDIDATES = ["checkpoints/unimodal/sensor/best_sensor_unimodal.pth"]


# ------------------------------------------------------------------------------
# 1. Tham số dòng lệnh
# ------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Huấn luyện Multi-Modal Teacher (2 pha)")
    # Dữ liệu & checkpoint
    p.add_argument("--manifest", default="data/final/final_multimodal_manifest.csv")
    p.add_argument("--data-dir", default="data/final")
    p.add_argument("--sensor-cache-dir", default=str(SENSOR_RESULTS_DIR),
                   help="Thư mục chứa sensor_features_{split}.csv của Member 3")
    p.add_argument("--sensor-preprocessor", default=str(SENSOR_PREPROCESSOR_PATH))
    p.add_argument("--vision-ckpt", default=None, help="Mặc định: tự tìm trong checkpoints/unimodal/{vision,image}/")
    p.add_argument("--audio-ckpt", default=None)
    p.add_argument("--sensor-ckpt", default=None)
    p.add_argument("--save-path", default="checkpoints/teachermodal/best_teacher.pth")
    p.add_argument("--results-dir", default=str(THIS_DIR / "results"))
    # Lịch huấn luyện
    p.add_argument("--epochs", type=int, default=30, help="Tổng số epoch (Pha 1 + Pha 2)")
    p.add_argument("--warmup-epochs", type=int, default=5, help="Số epoch Pha 1 (đóng băng backbone)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accum-steps", type=int, default=2, help="Gradient accumulation (batch hiệu dụng = bs x accum)")
    p.add_argument("--lr-head-warmup", type=float, default=1e-3, help="LR Fusion + Head ở Pha 1")
    p.add_argument("--lr-backbone", type=float, default=1e-5, help="LR 3 backbone ở Pha 2")
    p.add_argument("--lr-head", type=float, default=5e-4, help="LR Fusion + Head ở Pha 2")
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay-head", type=float, default=1e-2)
    p.add_argument("--weight-decay-backbone", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8, help="Early stopping (chỉ tính ở Pha 2)")
    p.add_argument("--grad-clip", type=float, default=1.0)
    # Loss & regularization
    p.add_argument("--loss-type", default="cost_sensitive_focal",
                   choices=["cost_sensitive_focal", "focal", "ce"],
                   help="Hàm mất mát: cost_sensitive_focal (mặc định), focal, hoặc ce")
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--focal-gamma", type=float, default=2.0, help="Số mũ gamma cho Focal Loss (mặc định 2.0)")
    p.add_argument("--cost-weight", type=float, default=0.5,
                   help="Trọng số phạt chi phí sai lầm thứ tự (Ordinal Cost Weight)")
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--modality-dropout", type=float, default=0.15)
    p.add_argument("--fusion-dropout", type=float, default=0.3)
    p.add_argument("--fusion-type", default="cross_attention",
                   choices=["cross_attention", "gated_concat"],
                   help="Cơ chế hợp nhất: cross_attention (Transformer) hoặc gated_concat")
    p.add_argument("--num-attn-heads", type=int, default=4, help="Số attention heads cho Cross-Attention")
    p.add_argument("--num-attn-layers", type=int, default=1, help="Số layer Transformer Cross-Attention")
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--latent-dim-mm", type=int, default=256)
    # Chọn mô hình tốt nhất
    p.add_argument("--select-metric", default="sample_macro_f1",
                   choices=["sample_macro_f1", "fruit_macro_f1", "val_loss"])
    # Phần cứng
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--seed", type=int, default=42)
    # Đánh giá & chạy thử
    p.add_argument("--resume", default=None, help="Đường dẫn last_teacher.pth để tiếp tục (mặc định: tự phát hiện)")
    p.add_argument("--fresh", action="store_true", help="Huấn luyện lại từ Epoch 1, bỏ qua checkpoint dở dang")
    p.add_argument("--eval-test", action="store_true", help="Chấm tập Test 1 lần sau khi chốt mô hình theo Val")
    p.add_argument("--eval-only", action="store_true", help="Chỉ chấm điểm từ checkpoint --save-path (bỏ qua train)")
    p.add_argument("--skip-final-eval", action="store_true")
    p.add_argument("--smoke", "--smoke-test", dest="smoke", action="store_true",
                   help="Chạy thử: 2 epoch, vài batch, không ghi đè checkpoint thật")
    p.add_argument("--max-batches", type=int, default=0, help="Giới hạn số batch / epoch (0 = không giới hạn)")
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
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(results_dir / ("smoke_train.log" if smoke else "train.log"), mode=mode,
                                  encoding="utf-8")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.propagate = False


def resolve_ckpt(user_path: Optional[str], candidates, name: str) -> Path:
    search_list = ([user_path] if user_path else []) + list(candidates)
    path = first_existing(search_list)
    if path is None:
        raise FileNotFoundError(f"Không tìm thấy checkpoint {name}. Đã thử: {search_list}")
    return path


def amp_settings(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return False, torch.float32
    return True, (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)


def make_loader(ds, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=device.type == "cuda", worker_init_fn=worker_init_fn,
                      persistent_workers=num_workers > 0, drop_last=False)


# ------------------------------------------------------------------------------
# 3. Vòng lặp train / validate
# ------------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp_on, amp_dtype,
                    accum_steps: int, grad_clip: float, max_batches: int, desc: str) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, n_seen, correct = 0.0, 0, 0
    n_batches = len(loader) if not max_batches else min(len(loader), max_batches)
    t0 = time.time()
    for step, (img, audio, sensor, label, _fid, _extra) in enumerate(loader):
        if step >= n_batches:
            break
        img, audio = img.to(device, non_blocking=True), audio.to(device, non_blocking=True)
        sensor, label = sensor.to(device, non_blocking=True), label.to(device, non_blocking=True)
        ctx = torch.autocast(device.type, dtype=amp_dtype) if amp_on else nullcontext()
        with ctx:
            logits, _ = model(img, audio, sensor)
        loss = criterion(logits.float(), label)
        scaler.scale(loss / accum_steps).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == n_batches:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        bs = label.size(0)
        total_loss += loss.item() * bs
        n_seen += bs
        correct += (logits.argmax(1) == label).sum().item()
        if (step + 1) % 20 == 0 or (step + 1) == n_batches:
            logger.info(f"  {desc} | batch {step + 1}/{n_batches} | loss {total_loss / n_seen:.4f} | "
                        f"{time.time() - t0:.0f}s")
    return {"train_loss": total_loss / max(n_seen, 1), "train_acc": correct / max(n_seen, 1)}


@torch.no_grad()
def validate(model, loader, device, amp_on, amp_dtype, max_batches: int = 0) -> Dict[str, float]:
    model.eval()
    probs_all, y_all, fids, gates = [], [], [], []
    for step, (img, audio, sensor, label, fid, _extra) in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        img, audio, sensor = img.to(device), audio.to(device), sensor.to(device)
        ctx = torch.autocast(device.type, dtype=amp_dtype) if amp_on else nullcontext()
        with ctx:
            logits, _ = model(img, audio, sensor)
        probs_all.append(F.softmax(logits.float(), dim=-1).cpu())
        y_all.append(label)
        fids.extend(fid)
        gates.append(model.last_gate_weights.float().cpu())
    probs = torch.cat(probs_all).numpy()
    y = torch.cat(y_all).numpy()
    pred = probs.argmax(1)
    val_loss = float(F.nll_loss(torch.log(torch.from_numpy(probs).clamp_min(1e-12)), torch.from_numpy(y)))

    df = pd.DataFrame(probs, columns=[0, 1, 2])
    df["fruit_id"], df["y"] = fids, y
    fr = df.groupby("fruit_id").agg({0: "mean", 1: "mean", 2: "mean", "y": "first"})
    fr_pred = fr[[0, 1, 2]].to_numpy().argmax(1)
    gate_mean = torch.cat(gates).mean(0).tolist()
    return {
        "val_loss": val_loss,
        "sample_acc": accuracy_score(y, pred),
        "sample_macro_f1": f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0),
        "fruit_acc": accuracy_score(fr["y"], fr_pred),
        "fruit_macro_f1": f1_score(fr["y"], fr_pred, average="macro", labels=[0, 1, 2], zero_division=0),
        **{f"gate_{m}": g for m, g in zip(MODALITIES, gate_mean)},
    }


def is_better(metrics: Dict[str, float], best: Dict[str, float], key: str) -> bool:
    if not best:
        return True
    if key == "val_loss":
        return metrics["val_loss"] < best["val_loss"] - 1e-6
    if metrics[key] > best[key] + 1e-9:
        return True
    return abs(metrics[key] - best[key]) <= 1e-9 and metrics["val_loss"] < best["val_loss"] - 1e-6


def plot_history(history: pd.DataFrame, warmup_epochs: int, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    axes[0].plot(history["epoch"], history["train_loss"], "o-", label="train")
    axes[0].plot(history["epoch"], history["val_loss"], "o-", label="val")
    axes[0].set_title("Loss")
    axes[1].plot(history["epoch"], history["sample_macro_f1"], "o-", label="val sample Macro-F1")
    axes[1].plot(history["epoch"], history["fruit_macro_f1"], "o-", label="val fruit Macro-F1")
    axes[1].set_title("Macro-F1 (Validation)")
    for m in MODALITIES:
        axes[2].plot(history["epoch"], history[f"gate_{m}"], "o-", label=m)
    axes[2].set_title("Trọng số gate trung bình (Val)")
    for ax in axes:
        ax.axvline(warmup_epochs + 0.5, color="gray", ls="--", lw=1)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Multi-Modal Teacher - Pha 1 (freeze) | Pha 2 (joint fine-tuning)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------
# 4. Đánh giá cuối bằng eval_teacher.py của Member 2
# ------------------------------------------------------------------------------
def final_evaluation(ckpt_path: Path, datasets: Dict, split: str, results_dir: Path, args, device) -> Dict:
    sys.path.insert(0, str(PROJECT_ROOT / "workspaces" / "member2_audio" / "03_test_eval_teacher_and_gradcam"))
    from eval_teacher import TeacherEvaluator, TeacherEvaluatorConfig  # noqa: E402

    ds = datasets[split]
    if args.smoke:
        ds = Subset(ds, list(range(min(len(ds), args.batch_size * 2))))
    loader = make_loader(ds, args.batch_size, False, args.num_workers, device)
    model = build_teacher_from_checkpoint(ckpt_path, device=device)
    cfg = TeacherEvaluatorConfig(
        teacher_checkpoint_path=str(ckpt_path), manifest_path=str(resolve_path(args.manifest)),
        test_data_dir=str(resolve_path(args.data_dir) / ("valid" if split != "test" else "test")),
        output_dir=str(results_dir), batch_size=args.batch_size, device=str(device),
        input_mode="multimodal", split_name=("smoke_" if args.smoke else "") + ("val" if split != "test" else "test"),
        model_name="multimodal_teacher", seed=args.seed,
    )
    return TeacherEvaluator(cfg, model=model, test_loader=loader).run_full_evaluation_pipeline()


def _atomic_torch_save(payload: Dict, target: Path) -> None:
    """Lưu file .pth thông qua file tạm (.tmp) rồi đổi tên nguyên tử để chống hỏng file khi rớt mạng."""
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(target)


# ------------------------------------------------------------------------------
# 5. Main
# ------------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs, args.warmup_epochs = 2, 1
        args.max_batches = args.max_batches or 3
        args.save_path = str(Path(args.save_path).with_name("smoke_teacher.pth"))
    results_dir = resolve_path(args.results_dir)
    save_path = resolve_path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    last_ckpt_name = "smoke_last_teacher.pth" if args.smoke else "last_teacher.pth"
    last_path = resolve_path(args.resume) if args.resume else (save_path.parent / last_ckpt_name)
    resuming = (not args.fresh) and (not args.eval_only) and last_path.is_file()

    setup_logging(results_dir, args.smoke, append=resuming)
    set_seed(args.seed)

    device = torch.device(args.device)
    amp_on, amp_dtype = amp_settings(device, not args.no_amp)

    vision_ckpt = resolve_ckpt(args.vision_ckpt, VISION_CKPT_CANDIDATES, "vision")
    audio_ckpt = resolve_ckpt(args.audio_ckpt, AUDIO_CKPT_CANDIDATES, "audio")
    sensor_ckpt = resolve_ckpt(args.sensor_ckpt, SENSOR_CKPT_CANDIDATES, "sensor")

    logger.info("=" * 78)
    logger.info("HUẤN LUYỆN MULTI-MODAL TEACHER (GIAI ĐOẠN 1B)")
    logger.info(f"Thiết bị: {device} | AMP: {amp_on} ({amp_dtype}) | batch {args.batch_size} x accum {args.accum_steps}")
    logger.info(f"Vision ckpt : {vision_ckpt}")
    logger.info(f"Audio ckpt  : {audio_ckpt}")
    logger.info(f"Sensor ckpt : {sensor_ckpt}")
    logger.info(f"Best ckpt   : {save_path}")
    logger.info(f"Resume ckpt : {last_path} ({'SẼ TIẾP TỤC TỪ CHECKPOINT CŨ' if resuming else 'train mới từ Epoch 1'})")
    logger.info("=" * 78)

    # ---- Dữ liệu ---------------------------------------------------------------
    splits = ("train", "valid", "test") if (args.eval_test or args.eval_only) else ("train", "valid")
    datasets = build_multimodal_datasets(
        resolve_path(args.manifest), resolve_path(args.data_dir), splits=splits, image_size=args.image_size,
        sensor_cache_dir=resolve_path(args.sensor_cache_dir),
        sensor_preprocessor_path=resolve_path(args.sensor_preprocessor), seed=args.seed,
    )
    if args.eval_only:
        logger.info(">>> CHẾ ĐỘ EVAL-ONLY: Chấm trực tiếp từ checkpoint đã lưu ...")
        res_val = final_evaluation(save_path, datasets, "valid", results_dir, args, device)
        logger.info(f"VAL  | sample F1 {res_val['metrics']['sample_macro_f1']:.4f} | "
                    f"fruit F1 {res_val['metrics']['fruit_macro_f1']:.4f}")
        res_test = final_evaluation(save_path, datasets, "test", results_dir, args, device)
        logger.info(f"TEST | sample F1 {res_test['metrics']['sample_macro_f1']:.4f} | "
                    f"fruit F1 {res_test['metrics']['fruit_macro_f1']:.4f}")
        return
    train_ds, val_ds = datasets["train"], datasets["valid"]
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, device)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers, device)

    # ---- Mô hình ---------------------------------------------------------------
    sensor_meta = torch.load(sensor_ckpt, map_location="cpu", weights_only=False)
    cfg = TeacherConfig(
        proj_dim=args.proj_dim, latent_dim_mm=args.latent_dim_mm, fusion_dropout=args.fusion_dropout,
        modality_dropout=args.modality_dropout,
        fusion_type=args.fusion_type,
        num_attn_heads=args.num_attn_heads,
        num_attn_layers=args.num_attn_layers,
        sensor_input_dim=int(sensor_meta.get("input_dim", 21)),
        sensor_hidden_dims=list(sensor_meta.get("hidden_dims", [64, 48])),
        sensor_latent_dim=int(sensor_meta.get("latent_dim", 32)),
        sensor_dropout=float(sensor_meta.get("dropout", 0.3)),
    )
    del sensor_meta
    model = MultiModalTeacher(cfg)
    model.load_unimodal_backbones(vision_ckpt, audio_ckpt, sensor_ckpt)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Teacher: {n_params / 1e6:.1f}M tham số ({args.fusion_type}) | f_mm = {cfg.latent_dim_mm} chiều")

    # ---- Loss ------------------------------------------------------------------
    class_w = None if args.no_class_weights else compute_class_weights(train_ds.labels, len(CLASS_ORDER))
    if class_w is not None:
        logger.info(f"Class weights (theo số mẫu Train): {dict(zip(CLASS_ORDER, class_w.numpy().round(3)))}")
    logger.info(f"Loss Function: {args.loss_type} (focal_gamma={args.focal_gamma}, cost_weight={args.cost_weight})")
    criterion = DurianClassificationLoss(
        class_weights=class_w,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        loss_type=args.loss_type,
        cost_weight=args.cost_weight,
    ).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on and amp_dtype == torch.float16)

    config_dump = {**vars(args), "vision_ckpt": str(vision_ckpt), "audio_ckpt": str(audio_ckpt),
                   "sensor_ckpt": str(sensor_ckpt), "save_path": str(save_path), "last_path": str(last_path),
                   "device_used": str(device), "amp": amp_on, "amp_dtype": str(amp_dtype),
                   "teacher_config": cfg.__dict__, "class_weights": None if class_w is None else class_w.tolist(),
                   "data_stats": {k: v.stats for k, v in datasets.items()}}
    with open(results_dir / ("smoke_config.json" if args.smoke else "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dump, f, ensure_ascii=False, indent=2, default=str)

    history, best, bad_epochs, start_epoch = [], {}, 0, 1
    resume_state = None
    if resuming:
        resume_state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_state["model_state_dict"], strict=True)
        model.to(device)
        if resume_state.get("scaler_state_dict") and scaler.is_enabled():
            scaler.load_state_dict(resume_state["scaler_state_dict"])
        history = list(resume_state.get("history", []))
        best = dict(resume_state.get("best", {}))
        bad_epochs = int(resume_state.get("bad_epochs", 0))
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        logger.info(f"🔄 Đã khôi phục trạng thái từ {last_path} (đã xong epoch {start_epoch - 1}/{args.epochs}, "
                    f"best {args.select_metric} = {best.get(args.select_metric, float('nan')):.4f})")

    phase2_epochs = max(args.epochs - args.warmup_epochs, 0)
    prefix = "smoke_" if args.smoke else ""

    def run_epoch(epoch: int, phase: str, optimizer) -> Dict[str, float]:
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, amp_on, amp_dtype,
                             args.accum_steps, args.grad_clip, args.max_batches, f"{phase} ep {epoch}")
        va = validate(model, val_loader, device, amp_on, amp_dtype, args.max_batches)
        lrs = {f"lr_{g.get('name', i)}": g["lr"] for i, g in enumerate(optimizer.param_groups)}
        row = {"epoch": epoch, "phase": phase, **tr, **va, **lrs}
        history.append(row)
        logger.info(f"[{phase}] Epoch {epoch}/{args.epochs} | train loss {tr['train_loss']:.4f} | "
                    f"val loss {va['val_loss']:.4f} | val F1 sample {va['sample_macro_f1']:.4f} / "
                    f"fruit {va['fruit_macro_f1']:.4f} | gate V/A/S "
                    f"{va['gate_vision']:.2f}/{va['gate_audio']:.2f}/{va['gate_sensor']:.2f}")
        return row

    def maybe_save(row: Dict[str, float]) -> bool:
        nonlocal best
        if is_better(row, best, args.select_metric):
            best = dict(row)
            _atomic_torch_save(teacher_checkpoint_dict(
                model, epoch=row["epoch"], phase=row["phase"], val_loss=row["val_loss"],
                val_f1=row["sample_macro_f1"], val_fruit_f1=row["fruit_macro_f1"],
                select_metric=args.select_metric, unimodal_sources={
                    "vision": str(vision_ckpt), "audio": str(audio_ckpt), "sensor": str(sensor_ckpt)},
                train_args=vars(args)), save_path)
            logger.info(f"  -> Lưu best Teacher ({args.select_metric} = {row[args.select_metric]:.4f}): {save_path}")
            return True
        return False

    def save_last_state(row: Dict[str, float], optimizer, scheduler=None) -> None:
        """Lưu trạng thái cuối mỗi epoch vào last_teacher.pth + cập nhật ngay CSV/biểu đồ."""
        _atomic_torch_save(teacher_checkpoint_dict(
            model,
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
        ), last_path)
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(results_dir / f"{prefix}history.csv", index=False)
        plot_history(hist_df, args.warmup_epochs, results_dir / f"{prefix}training_curves.png")

    # ---- PHA 1: WARM-UP FREEZE ------------------------------------------------
    if args.warmup_epochs > 0 and start_epoch <= args.warmup_epochs:
        logger.info(f">>> PHA 1 - WARM-UP FREEZE (epoch {start_epoch}..{args.warmup_epochs}): đóng băng 3 backbone")
        model.freeze_backbones()
        opt1 = torch.optim.AdamW(model.param_groups(0.0, args.lr_head_warmup, head_wd=args.weight_decay_head))
        if resume_state and resume_state.get("phase") == "phase1" and resume_state.get("optimizer_state_dict"):
            opt1.load_state_dict(resume_state["optimizer_state_dict"])
        for epoch in range(start_epoch, args.warmup_epochs + 1):
            row = run_epoch(epoch, "phase1", opt1)
            maybe_save(row)
            save_last_state(row, opt1)
        del opt1

    # ---- PHA 2: JOINT FINE-TUNING ---------------------------------------------
    p2_start = max(start_epoch, args.warmup_epochs + 1)
    if phase2_epochs > 0 and p2_start <= args.epochs and bad_epochs < args.patience:
        logger.info(f">>> PHA 2 - JOINT FINE-TUNING (epoch {p2_start}..{args.epochs}): backbone LR {args.lr_backbone}, "
                    f"fusion/head LR {args.lr_head}")
        model.unfreeze_all()
        opt2 = torch.optim.AdamW(model.param_groups(args.lr_backbone, args.lr_head,
                                                    backbone_wd=args.weight_decay_backbone,
                                                    head_wd=args.weight_decay_head))
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
        plot_history(hist_df, args.warmup_epochs, results_dir / f"{prefix}training_curves.png")
    if best:
        logger.info(f"Best epoch {best['epoch']} ({best['phase']}): val sample F1 {best['sample_macro_f1']:.4f} | "
                    f"fruit F1 {best['fruit_macro_f1']:.4f} | val loss {best['val_loss']:.4f}")

    # ---- Đánh giá cuối ---------------------------------------------------------
    if not args.skip_final_eval:
        del model, train_loader, val_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info(">>> Đánh giá best Teacher trên Val bằng eval_teacher.py (kèm Modality Ablation) ...")
        res = final_evaluation(save_path, datasets, "valid", results_dir, args, device)
        logger.info(f"VAL  | sample F1 {res['metrics']['sample_macro_f1']:.4f} | "
                    f"fruit F1 {res['metrics']['fruit_macro_f1']:.4f}")
        if args.eval_test:
            logger.info(">>> Chấm tập Test (1 lần, sau khi đã chốt mô hình theo Val) ...")
            res = final_evaluation(save_path, datasets, "test", results_dir, args, device)
            logger.info(f"TEST | sample F1 {res['metrics']['sample_macro_f1']:.4f} | "
                        f"fruit F1 {res['metrics']['fruit_macro_f1']:.4f}")
    logger.info("HOÀN TẤT HUẤN LUYỆN TEACHER.")


if __name__ == "__main__":
    main()

