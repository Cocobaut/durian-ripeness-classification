"""Huấn luyện nhánh Âm thanh đơn phương thức (Unimodal Audio Baseline) - Tuần 2.

Phụ trách: Member 2 (Nhánh Âm thanh)
Mô hình đã chọn ở Tuần 1: **ResNet-18 train từ đầu (timm, in_chans=1) trên log-Mel [1, 64, 306]**
(6 tiếng gõ x 0.5 s, xem notebook 01_eda_and_preprocess_audio, mục 9: Macro-F1 0.824 ± 0.018,
ổn định nhất giữa các fold, nhẹ, dễ tách vector đặc trưng 512 chiều cho Teacher).

Dữ liệu: data/final/{train,valid,test}/sound (238 file WAV mono 48 kHz, nhãn = Ripeness của quả).
Tiền xử lý: AudioPreprocessor của Tuần 1 (không viết lại).

Chống overfit đặc thù âm thanh:
- Mỗi epoch chọn ngẫu nhiên K tiếng gõ + dịch thời gian (AudioPreprocessor, train=True)
- SpecAugment: Time Masking + Frequency Masking (AudioPreprocessor.spec_augment)
- Mixup trên log-Mel
- Dropout ở head, Weight Decay (AdamW), Label Smoothing, Weighted Cross-Entropy (lệch lớp)
- Early stopping theo Macro-F1 trên tập Validation

Đầu ra:
- checkpoints/unimodal/best_audio_unimodal.pth   (model đầy đủ + head tách riêng + cấu hình)
- checkpoints/unimodal/audio_backbone_teacher.pth (chỉ backbone, để Tuần 3 rút vector f_audio 512 chiều)
- workspaces/member2_audio/02_train_unimodal_audio/results/
    config.json, train.log, history.csv, training_curves.png
    val_* và test_* (sinh bởi eval_teacher.py: metrics.csv/json, predictions, confusion matrix)
    audio_baseline_summary.csv (bảng gọn để ghép với nhánh ảnh và cảm biến)

Cách chạy (thư mục gốc repo, đã kích hoạt .venv):
    python workspaces/member2_audio/02_train_unimodal_audio/train_audio.py               # train + chấm Val
    python workspaces/member2_audio/02_train_unimodal_audio/train_audio.py --eval-test   # + chấm Test 1 lần
    python workspaces/member2_audio/02_train_unimodal_audio/train_audio.py --smoke       # chạy thử vài batch

Lưu ý: tập Test âm thanh đã được dùng 1 lần ở Tuần 1. Không chỉnh tham số dựa trên Test;
chỉ chấm Test một lần sau khi đã chốt mô hình theo Validation.

Dùng lại mô hình ở Tuần 3:
    from train_audio import AudioResNet18
    model = AudioResNet18()
    model.backbone.load_state_dict(torch.load("checkpoints/unimodal/audio_backbone_teacher.pth")["backbone_state_dict"])
    f_audio = model.forward_features(mel)   # [B, 512]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
MEMBER2_DIR = THIS_DIR.parent
PROJECT_ROOT = MEMBER2_DIR.parent.parent
sys.path.insert(0, str(MEMBER2_DIR / "01_eda_and_preprocess_audio"))
sys.path.insert(0, str(MEMBER2_DIR / "03_test_eval_teacher_and_gradcam"))
from preprocess_audio import AudioPreprocessor  # noqa: E402
from eval_teacher import evaluate  # noqa: E402

MODEL_NAME = "audio_resnet18_scratch"  # Tên mô hình đã chọn (ghi vào bảng kết quả)
CLASS_ORDER = ["Unripe", "Ripe", "Overripe"]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}
SPLITS = {"train": "train", "val": "valid", "test": "test"}  # tên split -> tên thư mục trong data/final
FILE_PATTERN = re.compile(r"^durian_(\d+)_(android|iphone)_(Unripe|Ripe|Overripe)_knock(?:_\d+)?\.wav$", re.I)

logger = logging.getLogger("train_audio")


# ------------------------------------------------------------------------------
# 1. Dữ liệu
# ------------------------------------------------------------------------------
def list_audio_files(data_dir: Path, split: str, manifest: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Liệt kê file âm thanh của một split và đọc nhãn từ tên file (đối chiếu manifest nếu có)."""
    folder = data_dir / SPLITS[split] / "sound"
    rows = []
    for path in sorted(folder.glob("*.wav")):
        m = FILE_PATTERN.match(path.name)
        if not m:
            logger.warning(f"Bỏ qua file không đúng định dạng tên: {path.name}")
            continue
        label = next(c for c in CLASS_ORDER if c.lower() == m.group(3).lower())
        rows.append({"file": path.name, "path": str(path), "split": split,
                     "fruit_id": f"durian_{int(m.group(1)):03d}", "device": m.group(2).lower(),
                     "label": label, "y": LABEL_TO_IDX[label]})
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"Không có file âm thanh hợp lệ trong {folder}")
    if manifest is not None:
        ripeness = dict(zip(manifest["fruit_id"], manifest["Ripeness"]))
        mismatch = df[df["fruit_id"].map(ripeness) != df["label"]]
        if len(mismatch):
            raise ValueError(f"{len(mismatch)} file có nhãn khác manifest, vd: {mismatch['file'].tolist()[:5]}")
    return df


class KnockDataset(Dataset):
    """Dataset âm thanh gõ: mỗi phần tử là dict {x [1, 64, T], label, fruit_id, device, file}.

    Các đoạn tiếng gõ của mỗi file được cắt sẵn một lần (AudioPreprocessor.load_knocks) và giữ trong RAM.
    Train: mỗi lần lấy mẫu chọn ngẫu nhiên K tiếng gõ + dịch thời gian + SpecAugment.
    Val/Test: K tiếng gõ cách đều, tất định.
    `repeats` > 1 cho phép mỗi epoch duyệt mỗi file nhiều lần với tổ hợp tiếng gõ khác nhau.
    """

    def __init__(self, files: pd.DataFrame, preprocessor: AudioPreprocessor, train: bool,
                 repeats: int = 1, seed: int = 42):
        self.files = files.reset_index(drop=True)
        self.pre = preprocessor
        self.train = train
        self.repeats = max(1, int(repeats)) if train else 1
        self.rng = np.random.default_rng(seed)
        self.knocks = [preprocessor.load_knocks(p) for p in self.files["path"]]

    def __len__(self) -> int:
        return len(self.files) * self.repeats

    def __getitem__(self, idx: int) -> Dict:
        i = idx % len(self.files)
        row = self.files.iloc[i]
        x = self.pre.process_knocks(self.knocks[i], train=self.train, rng=self.rng)
        return {"x": x, "label": int(row["y"]), "fruit_id": row["fruit_id"],
                "device": row["device"], "file": row["file"]}


# ------------------------------------------------------------------------------
# 2. Mô hình
# ------------------------------------------------------------------------------
class AudioResNet18(nn.Module):
    """ResNet-18 (timm, in_chans=1) trên log-Mel, tách riêng backbone và classification head.

    - `forward_features(x)`: vector đặc trưng f_audio [B, 512] (dùng cho Teacher ở Tuần 3).
    - `forward(x)`: (logits [B, 3], features [B, 512]) — cùng quy ước với nhánh ảnh của Lead.
    """

    def __init__(self, num_classes: int = 3, head_dropout: float = 0.3, drop_rate: float = 0.0,
                 pretrained: bool = False):
        super().__init__()
        self.backbone = timm.create_model("resnet18", pretrained=pretrained, in_chans=1, num_classes=0,
                                          drop_rate=drop_rate)
        self.num_features = self.backbone.num_features  # 512
        self.head = nn.Sequential(nn.Dropout(p=head_dropout), nn.Linear(self.num_features, num_classes))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(x)
        return self.head(features), features


# ------------------------------------------------------------------------------
# 3. Tiện ích huấn luyện
# ------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float, rng: np.random.Generator):
    lam = float(rng.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], y, y[perm], lam


def class_weights(labels: np.ndarray) -> torch.Tensor:
    """Trọng số nghịch tần suất, chuẩn hóa để trung bình có trọng số = 1."""
    counts = np.bincount(labels, minlength=len(CLASS_ORDER)).astype(float)
    w = np.where(counts > 0, counts.sum() / (len(CLASS_ORDER) * np.maximum(counts, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32)


def warn_gpu_memory(min_free_gb: float = 8.0) -> None:
    """Cảnh báo nếu GPU đang bị app khác chiếm nhiều VRAM (vd LM Studio nạp Qwen)."""
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    free_gb, total_gb = free / 1024 ** 3, total / 1024 ** 3
    logger.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM trống {free_gb:.1f}/{total_gb:.1f} GB")
    if free_gb < min_free_gb:
        logger.warning(f"VRAM trống chỉ {free_gb:.1f} GB. Có thể LM Studio đang nạp model (Qwen). "
                       f"Hãy Eject model hoặc chạy `lms unload --all` để train nhanh hơn.")


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, amp=False,
              mixup_alpha=0.0, mixup_prob=0.0, rng=None, max_batches=None) -> Dict[str, float]:
    train = optimizer is not None
    model.train(train)
    total_loss, n, preds, targets = 0.0, 0, [], []
    with torch.set_grad_enabled(train):
        for b, batch in enumerate(loader):
            if max_batches is not None and b >= max_batches:
                break
            x = batch["x"].to(device, non_blocking=True).float()
            y = batch["label"].to(device, non_blocking=True).long()
            use_mix = train and mixup_alpha > 0 and rng is not None and rng.random() < mixup_prob
            with torch.autocast(device_type=device.type,
                                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16, enabled=amp):
                if use_mix:
                    xm, ya, yb, lam = mixup(x, y, mixup_alpha, rng)
                    logits, _ = model(xm)
                    loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
                else:
                    logits, _ = model(x)
                    loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            preds.append(logits.argmax(1).detach().cpu().numpy())
            targets.append(y.cpu().numpy())
    preds, targets = np.concatenate(preds), np.concatenate(targets)
    return {"loss": total_loss / max(n, 1), "acc": float((preds == targets).mean()),
            "macro_f1": float(f1_score(targets, preds, labels=[0, 1, 2], average="macro", zero_division=0))}


def save_curves(history: pd.DataFrame, best_epoch: int, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, key, title in zip(axes, ["loss", "acc", "macro_f1"], ["Loss", "Accuracy", "Macro-F1"]):
        ax.plot(history["epoch"], history[f"train_{key}"], label="Train")
        ax.plot(history["epoch"], history[f"val_{key}"], label="Val")
        ax.axvline(best_epoch, color="green", ls="--", alpha=0.7, label=f"Best ep {best_epoch}")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"{MODEL_NAME}: learning curves (train có augmentation, val tất định)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_summary(results: Dict[str, Dict], path: Path) -> pd.DataFrame:
    """Bảng gọn để ghép với nhánh ảnh/cảm biến: mỗi dòng = (split, level)."""
    keys = ["n_samples", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "f1_unripe", "f1_ripe",
            "f1_overripe", "roc_auc_ovr_macro", "cohen_kappa", "mcc", "log_loss", "ece"]
    rows = []
    for split, res in results.items():
        for level in ("sample", "fruit"):
            row = {"modality": "audio", "model": MODEL_NAME, "split": split, "level": level}
            row.update({k: res["metrics"].get(f"{level}_{k}") for k in keys})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


# ------------------------------------------------------------------------------
# 4. Main
# ------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train nhánh âm thanh (ResNet-18 trên log-Mel)")
    p.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "final"))
    p.add_argument("--results-dir", default=str(THIS_DIR / "results"))
    p.add_argument("--ckpt-dir", default=str(PROJECT_ROOT / "checkpoints" / "unimodal"))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=15, help="Early stopping theo Val Macro-F1")
    p.add_argument("--repeats", type=int, default=4, help="Số lần duyệt mỗi file train trong 1 epoch")
    p.add_argument("--mixup-alpha", type=float, default=0.4)
    p.add_argument("--mixup-prob", type=float, default=0.5)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--head-dropout", type=float, default=0.3)
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--amp", dest="amp", action="store_true", default=None, help="Bật mixed precision (mặc định: bật nếu có CUDA)")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-test", action="store_true", help="Chấm tập Test (chỉ làm 1 lần sau khi chốt mô hình)")
    p.add_argument("--smoke", action="store_true", help="Chạy thử nhanh: 2 epoch x 2 batch")
    return p.parse_args(argv)


def main(argv=None) -> Dict:
    args = parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # tránh lỗi in tiếng Việt trên console Windows
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    results_dir, ckpt_dir, data_dir = Path(args.results_dir), Path(args.ckpt_dir), Path(args.data_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(results_dir / ("smoke_train.log" if args.smoke else "train.log"), "w", "utf-8")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.propagate = False

    set_seed(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    amp = (device.type == "cuda") if args.amp is None else (args.amp and device.type == "cuda")
    logger.info(f"Model: {MODEL_NAME} | device={device} | AMP={amp} | torch {torch.__version__}")
    warn_gpu_memory()

    manifest_path = data_dir / "final_multimodal_manifest.csv"
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else None
    files = {s: list_audio_files(data_dir, s, manifest) for s in ("train", "val", "test")}
    for s, df in files.items():
        logger.info(f"{s}: {len(df)} file, {df['fruit_id'].nunique()} quả, nhãn {df['label'].value_counts().to_dict()}")
    overlap = set(files["train"]["fruit_id"]) & (set(files["val"]["fruit_id"]) | set(files["test"]["fruit_id"]))
    if overlap:
        raise ValueError(f"Rò rỉ: quả xuất hiện ở cả train và val/test: {sorted(overlap)[:5]}")

    pre = AudioPreprocessor(random_state=args.seed)
    logger.info(f"Tiền xử lý: AudioPreprocessor, output {pre.output_shape}. Đang cắt tiếng gõ...")
    t0 = time.time()
    train_ds = KnockDataset(files["train"], pre, train=True, repeats=args.repeats, seed=args.seed)
    val_ds = KnockDataset(files["val"], pre, train=False)
    logger.info(f"Cắt tiếng gõ xong trong {time.time() - t0:.1f}s (file ít hơn K tiếng gõ: {len(pre.short_files)})")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=pin, drop_last=len(train_ds) > args.batch_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=pin)

    model = AudioResNet18(head_dropout=args.head_dropout).to(device)
    logger.info(f"Tham số: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | feature dim {model.num_features}")

    weights = None if args.no_class_weights else class_weights(files["train"]["y"].to_numpy())
    if weights is not None:
        logger.info(f"Class weights: {dict(zip(CLASS_ORDER, weights.numpy().round(3)))}")
    criterion = nn.CrossEntropyLoss(weight=None if weights is None else weights.to(device),
                                    label_smoothing=args.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss(weight=None if weights is None else weights.to(device))

    epochs = 2 if args.smoke else args.epochs
    max_batches = 2 if args.smoke else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(1, min(args.warmup_epochs, epochs - 1)) if epochs > 1 else 1
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup),
         torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup), eta_min=1e-6)],
        milestones=[warmup])
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    rng = np.random.default_rng(args.seed + 1)

    config = {**vars(args), "model_name": MODEL_NAME, "device_used": str(device), "amp_used": amp,
              "class_order": CLASS_ORDER, "input_shape": list(pre.output_shape),
              "preprocessor": {k: getattr(pre, k) for k in ("target_sr", "n_knocks", "knock_duration", "n_mels",
                                                            "n_fft", "win_length", "hop_length", "fmin", "fmax")},
              "class_weights": None if weights is None else weights.tolist(),
              "n_files": {s: len(df) for s, df in files.items()}}
    with open(results_dir / ("smoke_config.json" if args.smoke else "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=str)

    best_path = ckpt_dir / ("smoke_best_audio_unimodal.pth" if args.smoke else "best_audio_unimodal.pth")
    backbone_path = ckpt_dir / ("smoke_audio_backbone_teacher.pth" if args.smoke else "audio_backbone_teacher.pth")
    best = {"f1": -1.0, "loss": float("inf"), "epoch": 0}
    history, bad_epochs = [], 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, device, criterion, optimizer, scaler, amp,
                       args.mixup_alpha, args.mixup_prob, rng, max_batches)
        va = run_epoch(model, val_loader, device, eval_criterion, amp=amp, max_batches=max_batches)
        scheduler.step()
        history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                        **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}})
        improved = va["macro_f1"] > best["f1"] + 1e-4 or (abs(va["macro_f1"] - best["f1"]) <= 1e-4
                                                          and va["loss"] < best["loss"])
        if improved:
            best = {"f1": va["macro_f1"], "loss": va["loss"], "epoch": epoch}
            bad_epochs = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "backbone_state_dict": model.backbone.state_dict(),
                "head_state_dict": model.head.state_dict(),
                "best_val_f1": va["macro_f1"],
                "best_val_loss": va["loss"],
                "model_architecture": "resnet18",
                "model_name": MODEL_NAME,
                "feature_dim": model.num_features,
                "class_order": CLASS_ORDER,
                "config": config,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, best_path)
        else:
            bad_epochs += 1
        logger.info(f"Ep {epoch:03d} | train loss {tr['loss']:.4f} acc {tr['acc']:.3f} f1 {tr['macro_f1']:.3f} | "
                    f"val loss {va['loss']:.4f} acc {va['acc']:.3f} f1 {va['macro_f1']:.3f} | "
                    f"{time.time() - t0:.1f}s{' *' if improved else ''}")
        if bad_epochs >= args.patience:
            logger.info(f"Early stopping: Val Macro-F1 không cải thiện sau {args.patience} epoch.")
            break

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(results_dir / ("smoke_history.csv" if args.smoke else "history.csv"), index=False)
    save_curves(hist_df, best["epoch"], results_dir / ("smoke_training_curves.png" if args.smoke else "training_curves.png"))
    logger.info(f"Best epoch {best['epoch']}: Val Macro-F1 {best['f1']:.4f}, Val loss {best['loss']:.4f}")

    # Nạp mô hình tốt nhất, lưu backbone riêng cho Tuần 3
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    torch.save({"backbone_state_dict": model.backbone.state_dict(), "model_architecture": "resnet18",
                "timm_kwargs": {"in_chans": 1, "num_classes": 0}, "feature_dim": model.num_features,
                "input_shape": list(pre.output_shape), "source_checkpoint": best_path.name,
                "best_epoch": best["epoch"]}, backbone_path)
    logger.info(f"Đã lưu: {best_path} và {backbone_path}")

    # Chấm điểm bằng eval_teacher.py
    eval_dir = results_dir / "smoke_eval" if args.smoke else results_dir
    results = {"val": evaluate(model, val_loader, eval_dir, input_mode="audio", split_name="val",
                               device=str(device), model_name=MODEL_NAME)}
    if args.eval_test or args.smoke:
        test_ds = KnockDataset(files["test"], pre, train=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        results["test"] = evaluate(model, test_loader, eval_dir, input_mode="audio", split_name="test",
                                   device=str(device), model_name=MODEL_NAME)
    summary = build_summary(results, eval_dir / "audio_baseline_summary.csv")
    logger.info("Bảng baseline:\n" + summary[["split", "level", "n_samples", "accuracy", "macro_f1",
                                               "f1_unripe", "f1_ripe", "f1_overripe"]].to_string(index=False))
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)
    return {"best": best, "results": results, "summary": summary}


if __name__ == "__main__":
    main()
