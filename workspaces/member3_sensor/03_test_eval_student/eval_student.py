"""Module kiểm thử, đánh giá mô hình (Vision-Only Student / Baseline / Teacher / Sensor) và đo hiệu năng triển khai.

Phụ trách: Member 3 (Gia Bảo)
Dự án: Nhận diện độ chín sầu riêng bằng Chưng cất tri thức Đa phương thức (CMKD)

Script nhận vào MÔ HÌNH + DATALOADER và tự động xuất: bảng metrics, ma trận nhầm lẫn, log, CSV dự đoán.

Các chức năng chính
-------------------
1. StudentEvaluator.evaluate_and_report(model, loader)     -> metrics + confusion matrix + log (dùng cho mọi mô hình)
2. StudentEvaluator.evaluate_vision_model(...)              -> bảng dự đoán từng mẫu / từng góc ảnh
3. StudentEvaluator.evaluate_fruit_level_aggregation(...)   -> gộp 6 góc về cấp độ quả + Angle Consistency
4. StudentEvaluator.compute_metrics(...)                    -> Acc, Macro-F1, per-class P/R/F1/FPR, ROC-AUC, PR-AUC, ECE,
                                                               MCC, Kappa, Top-2, LogLoss, Brier
5. StudentEvaluator.compute_distillation_alignment(...)     -> KL-Divergence và Cosine so với Teacher
6. StudentEvaluator.compare_tri_benchmark(...)              -> Baseline vs Student vs Teacher, Distillation Gain, Retention
7. StudentDeploymentBenchmark                               -> Latency (mean/p95/p99), FPS, tham số, FLOPs, dung lượng, VRAM
8. StudentDistillationAblationEvaluator                     -> lưới ablation (T, alpha/beta/gamma, ...)
9. StudentBaselineCAMComparer                               -> Grad-CAM Baseline vs Student (CNN)

QUY ƯỚC DATALOADER (mặc định, có thể ghi đè bằng unpack_fn / forward_fn):
    batch = (images, labels, fruit_ids[, camera_angles])      # mô hình ảnh
    batch = (x_sensor, labels, fruit_ids)                      # mô hình sensor (SensorDataset)
    Teacher đa phương thức (img, audio, sensor, label, fruit_id): truyền unpack_fn và forward_fn riêng (xem docstring).

KIỂM TRA NHANH (không cần dữ liệu/checkpoint thật):
    python eval_student.py --selftest
"""

from __future__ import annotations

import copy
import io
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eval_student")

# ------------------------------------------------------------------------------
# 1. HẰNG SỐ CHUẨN TOÀN DỰ ÁN
# ------------------------------------------------------------------------------
CLASS_ORDER = ["Unripe", "Ripe", "Overripe"]  # 0: Chưa chín, 1: Chín, 2: Quá chín
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}
IDX_TO_LABEL = {i: c for i, c in enumerate(CLASS_ORDER)}
NUM_CLASSES = len(CLASS_ORDER)
PROB_COLS = [f"prob_{c.lower()}" for c in CLASS_ORDER]


# ------------------------------------------------------------------------------
# 2. TIỆN ÍCH CHUNG
# ------------------------------------------------------------------------------
def df_to_markdown(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Bảng Markdown không cần thư viện tabulate."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _default_unpack(batch) -> Tuple[Any, Any, Any, Any]:
    """Tách batch thành (đầu_vào, nhãn, fruit_id, góc_chụp). Hỗ trợ tuple/list và dict."""
    if isinstance(batch, dict):
        x = next((batch[k] for k in ("image", "img", "x", "input") if k in batch), None)
        y = next((batch[k] for k in ("label", "y", "target") if k in batch), None)
        fid = batch.get("fruit_id")
        ang = next((batch[k] for k in ("camera_angle", "angle") if k in batch), None)
        return x, y, fid, ang
    x, y = batch[0], batch[1]
    fid = batch[2] if len(batch) > 2 else None
    ang = batch[3] if len(batch) > 3 else None
    return x, y, fid, ang


def _default_forward(model: nn.Module, x: Any, device: torch.device):
    return model(x.to(device))


def _as_str_list(v: Any, n: int, prefix: str, offset: int) -> List[str]:
    if v is None:
        return [f"{prefix}_{offset + i}" for i in range(n)]
    if isinstance(v, torch.Tensor):
        v = v.tolist()
    return [str(a) for a in v]


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


# ------------------------------------------------------------------------------
# 3. CẤU HÌNH
# ------------------------------------------------------------------------------
@dataclass
class StudentEvaluatorConfig:
    """Cấu hình đánh giá. Các đường dẫn là mặc định, có thể ghi đè bằng tham số dòng lệnh."""
    student_checkpoint_path: str = "checkpoints/student_distilled_best.pth"
    baseline_checkpoint_path: str = "checkpoints/unimodal/best_vision_unimodal.pth"
    teacher_checkpoint_path: str = "checkpoints/teacher_best.pth"
    test_rgb_dir: str = "data/final/test/rgb"
    manifest_path: str = "data/final/final_multimodal_manifest.csv"
    output_dir: str = "workspaces/member3_sensor/03_test_eval_student/results"

    batch_size: int = 16
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    latency_warmup_runs: int = 50
    latency_test_runs: int = 200
    input_image_size: Tuple[int, int] = (224, 224)

    camera_angles: List[str] = field(default_factory=lambda: ["TOP", "BOT", "SA", "SB", "SC", "SD"])
    fruit_aggregation_strategy: str = "mean_prob"  # 'mean_prob' hoặc 'majority_vote'
    distill_temperature: float = 1.0


# ------------------------------------------------------------------------------
# 4. BỘ ĐÁNH GIÁ CHÍNH
# ------------------------------------------------------------------------------
class StudentEvaluator:
    """Bộ đánh giá toàn diện: nạp mô hình, suy luận, metrics, so sánh 3 mốc, độ lệch so với Teacher."""

    def __init__(
        self,
        config: StudentEvaluatorConfig,
        student_model: Optional[nn.Module] = None,
        baseline_model: Optional[nn.Module] = None,
        teacher_model: Optional[nn.Module] = None,
        test_loader: Optional[DataLoader] = None,
        model_builders: Optional[Dict[str, Callable[[dict], nn.Module]]] = None,
    ) -> None:
        """
        Args:
            model_builders: {'student': fn, 'baseline': fn, 'teacher': fn}; mỗi fn nhận dict checkpoint
                và trả về nn.Module chưa nạp trọng số. Cần khi checkpoint chỉ chứa state_dict và
                không có khóa 'arch' (tên kiến trúc timm).
        """
        self.config = config
        self.device = torch.device(config.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.student_model = student_model
        self.baseline_model = baseline_model
        self.teacher_model = teacher_model
        self.test_loader = test_loader
        self.model_builders = model_builders or {}
        self.last_outputs: Dict[str, Dict[str, Any]] = {}

        log_path = self.output_dir / "eval_student.log"
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path.resolve()
                   for h in logger.handlers):
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s]: %(message)s"))
            logger.addHandler(fh)
        logger.info(f"Khởi tạo StudentEvaluator trên thiết bị: {self.device} | kết quả: {self.output_dir}")

    # ---------------------------------------------------------------- nạp mô hình
    def load_model_from_checkpoint(self, checkpoint_path: str, architecture_type: str = "student") -> nn.Module:
        """Nạp mô hình từ .pth.

        Thứ tự xử lý:
          1. Checkpoint chứa nguyên đối tượng nn.Module (khóa 'model') -> dùng luôn.
          2. Dựng kiến trúc bằng model_builders[architecture_type](ckpt), hoặc timm.create_model(ckpt['arch']).
          3. Lấy state_dict từ 'model_state_dict' / 'state_dict' / 'full_state_dict' / 'backbone_state_dict'
             hoặc chính checkpoint; bỏ prefix 'module.'; bỏ các khóa Projection Head (proj/phi) khi kiểm thử.
        """
        logger.info(f"Đang nạp mô hình [{architecture_type}] từ: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if isinstance(ckpt, nn.Module):
            return ckpt.to(self.device).eval()
        if isinstance(ckpt, dict) and isinstance(ckpt.get("model"), nn.Module):
            return ckpt["model"].to(self.device).eval()

        if architecture_type in self.model_builders:
            model = self.model_builders[architecture_type](ckpt)
        elif isinstance(ckpt, dict) and "arch" in ckpt:
            import timm
            model = timm.create_model(ckpt["arch"], pretrained=False, num_classes=NUM_CLASSES)
        else:
            raise ValueError(
                f"Không biết dựng kiến trúc cho [{architecture_type}]. Hãy truyền "
                f"model_builders={{'{architecture_type}': lambda ckpt: MyModel(...)}} hoặc lưu khóa 'arch' trong checkpoint."
            )

        state = ckpt
        if isinstance(ckpt, dict):
            for k in ("model_state_dict", "state_dict", "full_state_dict", "backbone_state_dict"):
                if k in ckpt and isinstance(ckpt[k], dict):
                    state = ckpt[k]
                    break
        state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
        skip = ("projection", "proj_head", "phi.")
        state = {k: v for k, v in state.items() if not k.startswith(skip)}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning(f"[{architecture_type}] thiếu {len(missing)} khóa, ví dụ: {missing[:3]}")
        if unexpected:
            logger.warning(f"[{architecture_type}] thừa {len(unexpected)} khóa, ví dụ: {unexpected[:3]}")
        return model.to(self.device).eval()

    # ---------------------------------------------------------------- suy luận
    @torch.no_grad()
    def extract_outputs(
        self,
        model: nn.Module,
        loader: DataLoader,
        forward_fn: Callable = _default_forward,
        unpack_fn: Callable = _default_unpack,
    ) -> Dict[str, Any]:
        """Chạy mô hình trên loader, trả về logits, (tùy chọn) đặc trưng, nhãn, fruit_id, góc chụp.

        forward_fn(model, x, device) có thể trả về logits hoặc (logits, feats). Với SensorBackbone:
            forward_fn=lambda m, x, d: m(x.to(d), return_latent=True)
        Với Teacher đa phương thức: unpack_fn=lambda b: (b, b[3], b[4], None) và
            forward_fn=lambda m, b, d: m(b[0].to(d), b[1].to(d), b[2].to(d)).
        """
        model.eval().to(self.device)
        logits_l, feats_l, labels_l, fids, angs = [], [], [], [], []
        seen = 0
        for batch in loader:
            x, y, fid, ang = unpack_fn(batch)
            out = forward_fn(model, x, self.device)
            logits, feat = (out[0], out[1] if len(out) > 1 else None) if isinstance(out, (tuple, list)) else (out, None)
            n = logits.shape[0]
            logits_l.append(logits.detach().float().cpu())
            if feat is not None:
                feats_l.append(feat.detach().float().cpu())
            labels_l.append(torch.as_tensor(y).long().cpu())
            fids += _as_str_list(fid, n, "sample", seen)
            angs += _as_str_list(ang, n, "NA", seen) if ang is not None else ["NA"] * n
            seen += n
        return {
            "logits": torch.cat(logits_l),
            "feats": torch.cat(feats_l) if feats_l else None,
            "labels": torch.cat(labels_l).numpy(),
            "fruit_ids": fids,
            "angles": angs,
        }

    def evaluate_vision_model(
        self,
        model: nn.Module,
        test_loader: Optional[DataLoader] = None,
        model_name: str = "Student",
        forward_fn: Callable = _default_forward,
        unpack_fn: Callable = _default_unpack,
    ) -> pd.DataFrame:
        """Suy luận trên từng mẫu/góc ảnh. Trả về DataFrame:
        fruit_id, camera_angle, true_label, pred_label, prob_unripe, prob_ripe, prob_overripe, is_correct."""
        loader = test_loader or self.test_loader
        if loader is None:
            raise ValueError("Cần truyền test_loader hoặc gán StudentEvaluator.test_loader.")
        logger.info(f"Đánh giá mô hình [{model_name}] ở cấp độ góc nhìn (View-level)...")
        out = self.extract_outputs(model, loader, forward_fn, unpack_fn)
        self.last_outputs[model_name] = out
        probs = F.softmax(out["logits"], dim=-1).numpy()
        pred = probs.argmax(1)
        df = pd.DataFrame({
            "fruit_id": out["fruit_ids"],
            "camera_angle": out["angles"],
            "true_label": out["labels"],
            "pred_label": pred,
        })
        for i, col in enumerate(PROB_COLS):
            df[col] = probs[:, i]
        df["is_correct"] = (df["true_label"] == df["pred_label"]).astype(int)
        return df

    def evaluate_fruit_level_aggregation(self, sample_df: pd.DataFrame, strategy: Optional[str] = None) -> pd.DataFrame:
        """Gộp các góc chụp về cấp độ quả. Thêm Angle Consistency = tỷ lệ góc cùng dự đoán với nhãn quả."""
        strat = strategy or self.config.fruit_aggregation_strategy
        logger.info(f"Tổng hợp cấp độ quả theo chiến lược: {strat}")
        rows = []
        for fid, g in sample_df.groupby("fruit_id", sort=True):
            mean_p = g[PROB_COLS].mean().values
            if strat == "majority_vote":
                counts = np.bincount(g["pred_label"].values, minlength=NUM_CLASSES)
                top = np.flatnonzero(counts == counts.max())
                pred = int(top[np.argmax(mean_p[top])]) if len(top) > 1 else int(top[0])  # hòa phiếu -> theo xác suất
            elif strat == "mean_prob":
                pred = int(mean_p.argmax())
            else:
                raise ValueError("strategy phải là 'mean_prob' hoặc 'majority_vote'")
            true = int(g["true_label"].iloc[0])
            row = {"fruit_id": fid, "n_views": len(g), "true_label": true, "pred_label": pred,
                   "angle_consistency": float((g["pred_label"] == pred).mean()),
                   "is_correct": int(pred == true)}
            row.update({c: float(mean_p[i]) for i, c in enumerate(PROB_COLS)})
            rows.append(row)
        return pd.DataFrame(rows)

    # ---------------------------------------------------------------- metrics
    def compute_metrics(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        y_probs: Optional[np.ndarray] = None,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Metrics phân loại toàn diện (xem docstring đầu file). Metric cần xác suất chỉ tính khi có y_probs."""
        y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
        labels = list(range(NUM_CLASSES))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        pr, rc, f1c, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        m: Dict[str, float] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "cohen_kappa": cohen_kappa_score(y_true, y_pred, labels=labels),
        }
        for i, c in enumerate(CLASS_ORDER):
            fp = cm[:, i].sum() - cm[i, i]
            tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
            m[f"precision_{c}"], m[f"recall_{c}"], m[f"f1_{c}"] = pr[i], rc[i], f1c[i]
            m[f"fpr_{c}"] = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
        if y_probs is not None:
            p = np.asarray(y_probs, dtype=np.float64)
            p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)  # chuẩn hóa lại sai số float32
            onehot = np.eye(NUM_CLASSES)[y_true]
            m["top2_accuracy"] = float(np.mean([t in np.argsort(-r)[:2] for t, r in zip(y_true, p)]))
            m["ece"] = expected_calibration_error(y_true, p)
            m["brier"] = float(np.mean(np.sum((p - onehot) ** 2, axis=1)))
            try:
                m["log_loss"] = log_loss(y_true, p, labels=labels)
            except ValueError:
                m["log_loss"] = float("nan")
            try:
                m["roc_auc_macro"] = roc_auc_score(y_true, p, multi_class="ovr", average="macro", labels=labels)
                m["pr_auc_macro"] = float(np.mean([average_precision_score(onehot[:, i], p[:, i]) for i in labels]))
            except ValueError:  # thiếu lớp trong y_true
                m["roc_auc_macro"] = m["pr_auc_macro"] = float("nan")
        return {f"{prefix}{k}": float(v) for k, v in m.items()}

    def compute_distillation_alignment(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_feats: Optional[torch.Tensor] = None,
        teacher_feats: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        projection: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """Độ lệch của Student so với Teacher trên CÙNG thứ tự mẫu (loader không shuffle).

        - kl_teacher_to_student: KL(softmax(z_t/T) || softmax(z_s/T)), dạng KD chuẩn (batchmean).
        - kl_student_to_teacher: KL(softmax(z_s/T) || softmax(z_t/T)), đúng công thức L_KD trong đặc tả đề tài.
        - prediction_agreement: tỷ lệ mẫu Student và Teacher dự đoán trùng nhau.
        - cosine_similarity_mean/std: giữa phi(f_img^(S)) và f_mm. Nếu số chiều khác nhau phải truyền projection.
        """
        T = float(temperature or self.config.distill_temperature)
        s = torch.as_tensor(student_logits).float()
        t = torch.as_tensor(teacher_logits).float()
        if s.shape != t.shape:
            raise ValueError(f"Logits Student {tuple(s.shape)} và Teacher {tuple(t.shape)} không cùng kích thước "
                             "(kiểm tra thứ tự/ số mẫu của hai loader).")
        eps = 1e-12
        ps, pt = F.softmax(s / T, dim=-1), F.softmax(t / T, dim=-1)
        kl_ts = (pt * (torch.log(pt + eps) - torch.log(ps + eps))).sum(-1).mean().item()
        kl_st = (ps * (torch.log(ps + eps) - torch.log(pt + eps))).sum(-1).mean().item()
        res = {
            "temperature": T,
            "kl_teacher_to_student": kl_ts,
            "kl_student_to_teacher": kl_st,
            "prediction_agreement": float((s.argmax(1) == t.argmax(1)).float().mean().item()),
        }
        if student_feats is not None and teacher_feats is not None:
            sf = torch.as_tensor(student_feats).float()
            tf = torch.as_tensor(teacher_feats).float()
            if projection is not None:
                with torch.no_grad():
                    sf = projection.cpu().eval()(sf)
            if sf.shape[-1] != tf.shape[-1]:
                raise ValueError(f"Số chiều đặc trưng khác nhau (Student {sf.shape[-1]} vs Teacher {tf.shape[-1]}): "
                                 "hãy truyền projection (Projection Head phi) để đưa về cùng chiều.")
            cos = F.cosine_similarity(sf, tf, dim=-1)
            res["cosine_similarity_mean"] = float(cos.mean())
            res["cosine_similarity_std"] = float(cos.std(unbiased=False))
        return res

    # ---------------------------------------------------------------- so sánh 3 mốc
    def compare_tri_benchmark(
        self,
        baseline_metrics: Dict[str, float],
        student_metrics: Dict[str, float],
        teacher_metrics: Optional[Dict[str, float]] = None,
        save_name: str = "tri_benchmark",
    ) -> pd.DataFrame:
        """Bảng Baseline | Student | Gain (Δ) | Teacher | Retention (%). Gain = Student - Baseline
        (với ECE/LogLoss, giá trị càng THẤP càng tốt nên Δ âm là cải thiện)."""
        keys = ["accuracy", "balanced_accuracy", "macro_f1", "f1_Unripe", "f1_Ripe", "f1_Overripe", "ece", "log_loss"]
        higher_better = {"ece": False, "log_loss": False}
        rows = []
        for k in keys:
            if k not in baseline_metrics or k not in student_metrics:
                continue
            b, s = baseline_metrics[k], student_metrics[k]
            t = teacher_metrics.get(k) if teacher_metrics else None
            ret = (s / t * 100.0) if (t not in (None, 0) and higher_better.get(k, True)) else np.nan
            rows.append({
                "Metric": k, "Baseline (No KD)": b, "Student (CMKD)": s,
                "Distillation Gain (Δ)": s - b,
                "Teacher (Upper Bound)": t if t is not None else np.nan,
                "Retention (%)": ret,
            })
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / f"{save_name}.csv", index=False)
        (self.output_dir / f"{save_name}.md").write_text(df_to_markdown(df), encoding="utf-8")
        logger.info(f"Đã lưu bảng so sánh 3 mốc: {save_name}.csv / .md")
        return df

    # ---------------------------------------------------------------- vẽ
    def plot_confusion_matrix(self, y_true, y_pred, title: str, save_name: str) -> str:
        labels = list(range(NUM_CLASSES))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0
        fig, ax = plt.subplots(figsize=(5, 4.4))
        self._draw_cm(ax, cm, pct, title)
        fig.tight_layout()
        path = self.output_dir / save_name
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return str(path)

    @staticmethod
    def _draw_cm(ax, cm, pct, title):
        labels = list(range(NUM_CLASSES))
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(labels)
        ax.set_yticks(labels)
        ax.set_xticklabels(CLASS_ORDER)
        ax.set_yticklabels(CLASS_ORDER)
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")
        ax.set_title(title, fontsize=10)
        for i in labels:
            for j in labels:
                ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.0f}%)", ha="center", va="center",
                        color="white" if pct[i, j] > 55 else "black", fontsize=9)

    def plot_comparative_confusion_matrices(
        self,
        baseline_y_true: Sequence[int],
        baseline_y_pred: Sequence[int],
        student_y_true: Sequence[int],
        student_y_pred: Sequence[int],
        save_name: str = "comparative_confusion_matrix.png",
    ) -> str:
        """Hai ma trận nhầm lẫn song song (chuẩn hóa theo hàng, kèm số lượng)."""
        labels = list(range(NUM_CLASSES))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        for ax, (yt, yp, name) in zip(axes, [(baseline_y_true, baseline_y_pred, "Vision Baseline"),
                                              (student_y_true, student_y_pred, "Student (CMKD)")]):
            cm = confusion_matrix(yt, yp, labels=labels)
            self._draw_cm(ax, cm, cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0, name)
        fig.tight_layout()
        path = self.output_dir / save_name
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return str(path)

    # ---------------------------------------------------------------- đánh giá một mô hình (model + loader)
    def evaluate_and_report(
        self,
        model: nn.Module,
        loader: DataLoader,
        model_name: str = "model",
        forward_fn: Callable = _default_forward,
        unpack_fn: Callable = _default_unpack,
        aggregate_by_fruit: bool = False,
    ) -> Dict[str, float]:
        """Điểm vào chính: model + dataloader -> bảng metrics (CSV/MD), confusion matrix (PNG), dự đoán (CSV), log."""
        df = self.evaluate_vision_model(model, loader, model_name, forward_fn, unpack_fn)
        if aggregate_by_fruit:
            df = self.evaluate_fruit_level_aggregation(df)
        met = self.compute_metrics(df["true_label"], df["pred_label"], df[PROB_COLS].values)
        tag = f"{model_name}{'_fruit' if aggregate_by_fruit else ''}"
        df.to_csv(self.output_dir / f"predictions_{tag}.csv", index=False)
        mt = pd.DataFrame({"metric": list(met), "value": list(met.values())})
        mt.to_csv(self.output_dir / f"metrics_{tag}.csv", index=False)
        (self.output_dir / f"metrics_{tag}.md").write_text(df_to_markdown(mt), encoding="utf-8")
        self.plot_confusion_matrix(df["true_label"], df["pred_label"], f"{model_name} ({'fruit' if aggregate_by_fruit else 'view'})",
                                   f"confusion_matrix_{tag}.png")
        logger.info(f"[{model_name}] n={len(df)} | acc={met['accuracy']:.4f} | macro-F1={met['macro_f1']:.4f} | "
                    f"ECE={met.get('ece', float('nan')):.4f}")
        return met

    # ---------------------------------------------------------------- chạy toàn bộ so sánh 3 mốc
    def run_full_evaluation(
        self,
        test_loader: Optional[DataLoader] = None,
        teacher_loader: Optional[DataLoader] = None,
        teacher_forward_fn: Callable = _default_forward,
        teacher_unpack_fn: Callable = _default_unpack,
        student_forward_fn: Callable = _default_forward,
        student_projection: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        """Baseline vs Student (vs Teacher): metrics cấp góc ảnh và cấp quả, Distillation Gain, KL/Cosine, ma trận nhầm lẫn."""
        loader = test_loader or self.test_loader
        models = {"baseline": self.baseline_model, "student": self.student_model}
        for name, path in (("baseline", self.config.baseline_checkpoint_path), ("student", self.config.student_checkpoint_path)):
            if models[name] is None and Path(path).exists():
                models[name] = self.load_model_from_checkpoint(path, name)
        if self.teacher_model is None and Path(self.config.teacher_checkpoint_path).exists() and teacher_loader is not None:
            self.teacher_model = self.load_model_from_checkpoint(self.config.teacher_checkpoint_path, "teacher")

        results: Dict[str, Any] = {}
        view_df, fruit_df, met_view, met_fruit = {}, {}, {}, {}
        for name, mdl in models.items():
            if mdl is None:
                logger.warning(f"Bỏ qua [{name}]: chưa có mô hình/checkpoint.")
                continue
            view_df[name] = self.evaluate_vision_model(mdl, loader, name, student_forward_fn)
            fruit_df[name] = self.evaluate_fruit_level_aggregation(view_df[name])
            for tag, d, store in (("view", view_df[name], met_view), ("fruit", fruit_df[name], met_fruit)):
                store[name] = self.compute_metrics(d["true_label"], d["pred_label"], d[PROB_COLS].values)
                d.to_csv(self.output_dir / f"predictions_{name}_{tag}.csv", index=False)
                self.plot_confusion_matrix(d["true_label"], d["pred_label"], f"{name} ({tag})",
                                           f"confusion_matrix_{name}_{tag}.png")

        t_view = t_fruit = None
        if self.teacher_model is not None and teacher_loader is not None:
            tdf = self.evaluate_vision_model(self.teacher_model, teacher_loader, "teacher", teacher_forward_fn, teacher_unpack_fn)
            t_view = self.compute_metrics(tdf["true_label"], tdf["pred_label"], tdf[PROB_COLS].values)
            t_fruit = self.compute_metrics(*(lambda d: (d["true_label"], d["pred_label"], d[PROB_COLS].values))(
                self.evaluate_fruit_level_aggregation(tdf)))
            if "student" in self.last_outputs and "teacher" in self.last_outputs:
                so, to = self.last_outputs["student"], self.last_outputs["teacher"]
                results["alignment"] = self.compute_distillation_alignment(
                    so["logits"], to["logits"], so["feats"], to["feats"], projection=student_projection)
                pd.DataFrame([results["alignment"]]).to_csv(self.output_dir / "distillation_alignment.csv", index=False)

        if "baseline" in met_view and "student" in met_view:
            results["tri_view"] = self.compare_tri_benchmark(met_view["baseline"], met_view["student"], t_view, "tri_benchmark_view")
            results["tri_fruit"] = self.compare_tri_benchmark(met_fruit["baseline"], met_fruit["student"], t_fruit, "tri_benchmark_fruit")
            self.plot_comparative_confusion_matrices(
                view_df["baseline"]["true_label"], view_df["baseline"]["pred_label"],
                view_df["student"]["true_label"], view_df["student"]["pred_label"])
        (self.output_dir / "metrics_summary.json").write_text(
            json.dumps({"view": met_view, "fruit": met_fruit, "teacher_view": t_view, "teacher_fruit": t_fruit,
                        "alignment": results.get("alignment")}, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
        return results


# ------------------------------------------------------------------------------
# 5. ĐO LƯỜNG HIỆU NĂNG TRIỂN KHAI
# ------------------------------------------------------------------------------
class StudentDeploymentBenchmark:
    """Latency, FPS, tham số, FLOPs, dung lượng, VRAM đỉnh."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def measure_latency_and_throughput(
        self,
        model: nn.Module,
        input_size: Tuple[int, ...] = (1, 3, 224, 224),
        device: str = "cpu",
        warmup_runs: int = 50,
        test_runs: int = 200,
    ) -> Dict[str, float]:
        """Latency (ms) mean/std/median/min/max/p95/p99; FPS = batch*1000/mean; VRAM đỉnh (MB) nếu dùng CUDA.
        input_size có thể là (B,C,H,W) hoặc (B, số_đặc_trưng) cho mô hình sensor."""
        logger.info(f"Đo Latency & Throughput trên [{device}] với input {input_size}...")
        dev = torch.device(device)
        mdl = copy.deepcopy(model).to(dev).eval()
        x = torch.randn(*input_size, device=dev)
        use_cuda = dev.type == "cuda"
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(dev)
        for _ in range(warmup_runs):
            mdl(x)
        if use_cuda:
            torch.cuda.synchronize()
        times = []
        for _ in range(test_runs):
            if use_cuda:
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                mdl(x)
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
            else:
                t0 = time.perf_counter()
                mdl(x)
                times.append((time.perf_counter() - t0) * 1000.0)
        t = np.asarray(times)
        res = {
            "device": device, "batch_size": int(input_size[0]),
            "latency_ms_mean": float(t.mean()), "latency_ms_std": float(t.std()), "latency_ms_median": float(np.median(t)),
            "latency_ms_min": float(t.min()), "latency_ms_max": float(t.max()),
            "latency_ms_p95": float(np.percentile(t, 95)), "latency_ms_p99": float(np.percentile(t, 99)),
            "fps": float(input_size[0] * 1000.0 / t.mean()),
        }
        if use_cuda:
            res["vram_peak_mb"] = float(torch.cuda.max_memory_allocated(dev) / 1024 ** 2)
        return res

    def profile_model_complexity(self, model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 224, 224)) -> Dict[str, Any]:
        """Tham số (tổng/huấn luyện được), dung lượng (MB, đo bằng state_dict tuần tự hóa), FLOPs/MACs."""
        logger.info("Tính độ phức tạp mô hình (Parameters, FLOPs, Size)...")
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        res: Dict[str, Any] = {
            "params_total": int(total), "params_trainable": int(trainable), "params_millions": total / 1e6,
            "model_size_mb": buf.getbuffer().nbytes / 1024 ** 2,
        }
        try:
            from torch.utils.flop_counter import FlopCounterMode
            mdl = copy.deepcopy(model).cpu().eval()
            with torch.no_grad(), FlopCounterMode(display=False) as fc:
                mdl(torch.randn(*input_size))
            flops = fc.get_total_flops()
            res.update(flops=int(flops), gflops=flops / 1e9, macs=int(flops // 2), gmacs=flops / 2e9)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Không tính được FLOPs: {e}")
            res.update(flops=np.nan, gflops=np.nan, macs=np.nan, gmacs=np.nan)
        return res

    def run_and_save(self, models: Dict[str, nn.Module], input_size: Tuple[int, ...] = (1, 3, 224, 224),
                     devices: Sequence[str] = ("cpu",), save_name: str = "deployment_benchmark") -> pd.DataFrame:
        """Đo nhiều mô hình (ví dụ Student, Baseline, Teacher) và lưu bảng so sánh."""
        rows = []
        for name, mdl in models.items():
            cx = self.profile_model_complexity(mdl, input_size)
            for dv in devices:
                if dv.startswith("cuda") and not torch.cuda.is_available():
                    continue
                rows.append({"model": name, **cx, **self.measure_latency_and_throughput(mdl, input_size, dv)})
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / f"{save_name}.csv", index=False)
        (self.output_dir / f"{save_name}.md").write_text(df_to_markdown(df), encoding="utf-8")
        return df


# ------------------------------------------------------------------------------
# 6. ABLATION
# ------------------------------------------------------------------------------
class StudentDistillationAblationEvaluator:
    """Đánh giá lưới ablation: T, alpha/beta, gamma, Cosine vs MSE..."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_ablation_grid(
        self,
        checkpoints_dict: Dict[str, str],
        test_loader: DataLoader,
        evaluator: StudentEvaluator,
    ) -> pd.DataFrame:
        """checkpoints_dict = {'T4_a1_b1_g0.5': 'path.pth', ...}. Dùng evaluator.load_model_from_checkpoint."""
        rows = []
        for name, path in checkpoints_dict.items():
            mdl = evaluator.load_model_from_checkpoint(path, "student")
            v = evaluator.evaluate_vision_model(mdl, test_loader, f"ablation_{name}")
            f = evaluator.evaluate_fruit_level_aggregation(v)
            mv = evaluator.compute_metrics(v["true_label"], v["pred_label"], v[PROB_COLS].values)
            mf = evaluator.compute_metrics(f["true_label"], f["pred_label"], f[PROB_COLS].values)
            rows.append({"config": name,
                         **{f"view_{k}": mv[k] for k in ("accuracy", "macro_f1", "f1_Unripe", "f1_Ripe", "f1_Overripe", "ece")},
                         "fruit_accuracy": mf["accuracy"], "fruit_macro_f1": mf["macro_f1"]})
        df = pd.DataFrame(rows).sort_values("view_macro_f1", ascending=False)
        df.to_csv(self.output_dir / "ablation_results.csv", index=False)
        (self.output_dir / "ablation_results.md").write_text(df_to_markdown(df), encoding="utf-8")
        fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(df)), 4))
        ax.bar(df["config"], df["view_macro_f1"], color="#4C78A8")
        ax.set_ylabel("Macro-F1 (view)")
        ax.set_ylim(0, 1)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(self.output_dir / "ablation_macro_f1.png", dpi=160)
        plt.close(fig)
        return df


# ------------------------------------------------------------------------------
# 7. GRAD-CAM: BASELINE VS STUDENT (CNN)
# ------------------------------------------------------------------------------
class _GradCAM:
    """Grad-CAM cho mạng tích chập: hook vào tầng target_layer."""

    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model
        self.acts: Optional[torch.Tensor] = None
        self.grads: Optional[torch.Tensor] = None
        self._h = layer.register_forward_hook(self._fwd)

    def _fwd(self, _m, _i, out):
        self.acts = out
        out.register_hook(lambda g: setattr(self, "grads", g))

    def close(self):
        self._h.remove()

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None):
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            out = self.model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            probs = F.softmax(logits.detach(), dim=-1)[0]
            idx = int(class_idx if class_idx is not None else probs.argmax())
            logits[0, idx].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy(), probs.cpu().numpy(), idx


class StudentBaselineCAMComparer:
    """Đặt Grad-CAM của Baseline và Student cạnh nhau trên cùng một ảnh."""

    def __init__(self, baseline_model: nn.Module, student_model: nn.Module, target_layer_name: str,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        self.baseline_model = baseline_model
        self.student_model = student_model
        self.target_layer_name = target_layer_name
        self.device = torch.device(device)

    def _layer(self, model: nn.Module) -> nn.Module:
        mods = dict(model.named_modules())
        if self.target_layer_name not in mods:
            raise KeyError(f"Không có tầng '{self.target_layer_name}'. Ví dụ tên hợp lệ: {list(mods)[-5:]}")
        return mods[self.target_layer_name]

    def generate_side_by_side_cam(
        self,
        rgb_tensor: torch.Tensor,
        raw_rgb_image: np.ndarray,
        target_class: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """[Ảnh gốc] | [Baseline CAM] | [Student CAM]. rgb_tensor: (1,3,H,W) đã chuẩn hóa; raw_rgb_image: HxWx3 uint8."""
        x = (rgb_tensor if rgb_tensor.dim() == 4 else rgb_tensor.unsqueeze(0)).to(self.device)
        cams, infos = [], []
        for mdl in (self.baseline_model, self.student_model):
            mdl.to(self.device).eval()
            g = _GradCAM(mdl, self._layer(mdl))
            try:
                cam, probs, idx = g(x, target_class)
            finally:
                g.close()
            cams.append(cam)
            infos.append((idx, float(probs[idx])))
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        axes[0].imshow(raw_rgb_image)
        axes[0].set_title("Ảnh gốc RGB")
        for ax, cam, (idx, p), name in zip(axes[1:], cams, infos, ("Baseline (trước KD)", "Student (sau KD)")):
            cam_r = F.interpolate(torch.tensor(cam)[None, None], size=raw_rgb_image.shape[:2], mode="bilinear",
                                  align_corners=False)[0, 0].numpy()
            ax.imshow(raw_rgb_image)
            ax.imshow(cam_r, cmap="jet", alpha=0.45)
            ax.set_title(f"{name}\n{CLASS_ORDER[idx]} ({p:.2f})")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=160)
        plt.close(fig)
        return cams[0], cams[1]


# ------------------------------------------------------------------------------
# 8. TỰ KIỂM TRA & CLI
# ------------------------------------------------------------------------------
def _selftest(output_dir: str) -> None:
    """Chạy toàn bộ pipeline trên mô hình/dữ liệu ngẫu nhiên để xác nhận mọi hàm hoạt động."""
    torch.manual_seed(0)

    class Tiny(nn.Module):
        def __init__(self, seed=0, feat=8):
            super().__init__()
            torch.manual_seed(seed)
            self.conv = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, feat, 3, padding=1), nn.ReLU())
            self.fc = nn.Linear(feat, NUM_CLASSES)

        def forward(self, x, return_feat=False):
            f = self.conv(x).mean(dim=(2, 3))
            return (self.fc(f), f) if return_feat else self.fc(f)

    n_fruit, angles = 9, ["TOP", "BOT", "SA"]
    N = n_fruit * len(angles)
    X = torch.randn(N, 3, 32, 32)
    y = torch.tensor([(i // len(angles)) % 3 for i in range(N)])
    fids = [f"durian_{i // len(angles):03d}" for i in range(N)]
    angs = [angles[i % len(angles)] for i in range(N)]
    ds = list(zip(X, y, fids, angs))
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    cfg = StudentEvaluatorConfig(output_dir=output_dir, device="cpu")
    base, stud, teach = Tiny(1), Tiny(2), Tiny(3, feat=16)
    ev = StudentEvaluator(cfg, student_model=stud, baseline_model=base, teacher_model=teach, test_loader=loader)

    fwd_feat = lambda m, x, d: m(x.to(d), return_feat=True)  # noqa: E731
    res = ev.run_full_evaluation(teacher_loader=loader, teacher_forward_fn=fwd_feat, student_forward_fn=fwd_feat,
                                 student_projection=nn.Linear(8, 16))
    ev.evaluate_and_report(stud, loader, "student_report")
    assert "alignment" in res and "tri_view" in res and "tri_fruit" in res

    db = StudentDeploymentBenchmark(output_dir)
    db.run_and_save({"student": stud, "baseline": base}, (1, 3, 32, 32), ("cpu",))
    # mô hình sensor (đầu vào 1 chiều)
    sens = nn.Sequential(nn.Linear(21, 32), nn.ReLU(), nn.Linear(32, 3))
    print("sensor:", db.profile_model_complexity(sens, (1, 21))["params_total"], "params")

    ab = StudentDistillationAblationEvaluator(output_dir)
    ck = Path(output_dir) / "tiny.pth"
    torch.save({"model_state_dict": {"module." + k: v for k, v in stud.state_dict().items()}}, ck)
    ev.model_builders["student"] = lambda c: Tiny(2)
    ab.evaluate_ablation_grid({"cfgA": str(ck), "cfgB": str(ck)}, loader, ev)

    cam = StudentBaselineCAMComparer(base, stud, "conv.2", "cpu")
    raw = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    cam.generate_side_by_side_cam(X[:1], raw, save_path=str(Path(output_dir) / "cam_demo.png"))
    print(f"[SELFTEST OK] Kết quả: {output_dir}")


def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Đánh giá Student / Baseline / Teacher và đo hiệu năng triển khai (CMKD)")
    p.add_argument("--student-checkpoint", default="checkpoints/student_distilled_best.pth")
    p.add_argument("--baseline-checkpoint", default="checkpoints/unimodal/best_vision_unimodal.pth")
    p.add_argument("--teacher-checkpoint", default="checkpoints/teacher_best.pth")
    p.add_argument("--output-dir", default="workspaces/member3_sensor/03_test_eval_student/results")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--selftest", action="store_true", help="Chạy thử toàn bộ pipeline bằng dữ liệu ngẫu nhiên")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.selftest:
        _selftest(args.output_dir)
    else:
        cfg = StudentEvaluatorConfig(
            student_checkpoint_path=args.student_checkpoint, baseline_checkpoint_path=args.baseline_checkpoint,
            teacher_checkpoint_path=args.teacher_checkpoint, output_dir=args.output_dir,
            batch_size=args.batch_size, device=args.device)
        ev = StudentEvaluator(cfg)
        print("Đã khởi tạo StudentEvaluator. Dùng như thư viện:")
        print("  ev = StudentEvaluator(cfg, model_builders={'student': build_student, 'baseline': build_baseline})")
        print("  ev.evaluate_and_report(model, loader, 'ten_mo_hinh')    # model + dataloader -> metrics, confusion matrix, log")
        print("  ev.run_full_evaluation(test_loader=loader)              # Baseline vs Student (vs Teacher)")
        print("Chạy thử không cần dữ liệu: python eval_student.py --selftest")