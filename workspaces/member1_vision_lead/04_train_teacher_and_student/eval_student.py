"""Đánh giá độc lập mô hình Vision-Only Student (Giai đoạn 2 CMKD) ở cấp Sample & Fruit.

Dự án: Phân loại độ chín sầu riêng bằng Chưng cất tri thức Đa phương thức (CMKD)
Tham chiếu đặc tả: Mục 1.3.3 & Mục 1.4.

Đặc điểm quan trọng:
- Khi đánh giá / suy luận thực tế, Student CHỈ NHẬN ẢNH RGB (`ImageDataset`, `input_mode="vision"`).
- Không cần micro thu âm, không cần cảm biến và không cần mô hình Teacher.
- Tích hợp chiến lược gộp cấp quả có trọng số theo góc chụp (`angle_weighted_prob`):
  Ưu tiên góc chụp cuống quả (`TOP`) và các mặt bên (`SD`, `SA`, `SB`), giảm trọng số góc đáy (`BOT`)
  vốn chứa ít thông tin thị giác phân biệt độ chín.
- Tự động tổng hợp bảng đối sánh `benchmark_comparison.csv` & `benchmark_comparison.md` giữa:
  Vision Unimodal vs Audio Unimodal vs Sensor Unimodal vs Multi-Modal Teacher vs Vision-Only Student.

Cách chạy (từ thư mục gốc dự án):
    python workspaces/member1_vision_lead/04_train_teacher_and_student/eval_student.py \
        --checkpoint checkpoints/studentmodal/best_student.pth \
        --split test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from src.datasets.image_dataset import IMAGE_SIZE, ImageDataset, get_eval_transforms, list_rgb_images  # noqa: E402
from src.datasets.multimodal_dataset import split_dir_name, worker_init_fn  # noqa: E402
from src.models.student_model import VisionStudent, build_student_from_checkpoint  # noqa: E402
from src.utils.common import resolve_path  # noqa: E402

EVAL_TEACHER_DIR = PROJECT_ROOT / "workspaces" / "member2_audio" / "03_test_eval_teacher_and_gradcam"
if str(EVAL_TEACHER_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_TEACHER_DIR))
from eval_teacher import (  # noqa: E402
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    TeacherEvaluator,
    TeacherEvaluatorConfig,
)

logger = logging.getLogger("eval_student")

# Trọng số mặc định theo góc chụp camera cho bài toán phân loại độ chín bằng ảnh RGB:
# - TOP (cuống quả) và SD/SA/SB (các góc mặt bên chính) mang nhiều dấu hiệu độ chín nhất.
# - BOT (đáy quả) có cụm gai chụm ít biến đổi theo độ chín nên được giảm trọng số.
DEFAULT_CAMERA_ANGLE_WEIGHTS: Dict[str, float] = {
    "TOP": 1.35,
    "SD": 1.25,
    "SA": 1.15,
    "SB": 1.10,
    "SC": 0.90,
    "BOT": 0.45,
}


def compute_angle_weighted_fruit_probs(
    probs: np.ndarray,
    camera_angles: Sequence[str],
    angle_weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Tính vector xác suất cấp quả [3] có trọng số theo góc chụp `camera_angle` và độ tự tin mẫu."""
    w_map = DEFAULT_CAMERA_ANGLE_WEIGHTS if angle_weights is None else angle_weights
    angle_w = np.array([float(w_map.get(str(a).upper(), 1.0)) for a in camera_angles], dtype=np.float64)
    # Kết hợp nhẹ với độ tự tin của từng góc chụp (0.75 + 0.25 * max_prob) để giảm nhiễu từ góc bị mờ/lóa
    max_prob = probs.max(axis=1)
    sample_w = (angle_w * (0.75 + 0.25 * max_prob))[:, None]
    weighted_probs = (probs * sample_w).sum(axis=0) / max(float(sample_w.sum()), 1e-12)
    return weighted_probs / max(float(weighted_probs.sum()), 1e-12)


class StudentEvaluator(TeacherEvaluator):
    """Bộ đánh giá mở rộng hỗ trợ `angle_weighted_prob` khi tổng hợp dự đoán cấp quả cho Student."""

    def __init__(
        self,
        config: TeacherEvaluatorConfig,
        model: Optional[torch.nn.Module] = None,
        test_loader: Optional[DataLoader] = None,
        angle_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__(config, model=model, test_loader=test_loader)
        self.angle_weights = DEFAULT_CAMERA_ANGLE_WEIGHTS if angle_weights is None else dict(angle_weights)

    def evaluate_fruit_level(
        self,
        sample_df: pd.DataFrame,
        strategy: Optional[str] = None,
    ) -> pd.DataFrame:
        agg_strategy = strategy or self.config.fruit_aggregation_strategy
        if agg_strategy != "angle_weighted_prob":
            return super().evaluate_fruit_level(sample_df, strategy=agg_strategy)

        logger.info("Tổng hợp kết quả cấp độ quả (fruit_id) theo chiến lược: angle_weighted_prob...")
        prob_cols = ["prob_unripe", "prob_ripe", "prob_overripe"]
        rows = []
        for fruit_id, group in sample_df.groupby("fruit_id", sort=True):
            true_label = group["true_label"].mode().iloc[0]
            probs = group[prob_cols].to_numpy(dtype=np.float64)
            angles = group["camera_angle"].astype(str).tolist() if "camera_angle" in group.columns else [""] * len(group)
            mean_probs = compute_angle_weighted_fruit_probs(probs, angles, self.angle_weights)
            pred_idx = int(mean_probs.argmax())
            rows.append({
                "fruit_id": fruit_id,
                "true_label": true_label,
                "pred_label": IDX_TO_LABEL[pred_idx],
                "prob_unripe_mean": float(mean_probs[0]),
                "prob_ripe_mean": float(mean_probs[1]),
                "prob_overripe_mean": float(mean_probs[2]),
                "is_correct": bool(true_label == IDX_TO_LABEL[pred_idx]),
                "n_samples": int(len(group)),
                "y_true": LABEL_TO_IDX[true_label],
                "y_pred": pred_idx,
            })
        fruit_df = pd.DataFrame(rows)
        logger.info(f"Đã tổng hợp {len(fruit_df)} quả, fruit accuracy = {fruit_df['is_correct'].mean():.4f}")
        return fruit_df


class _ManifestFallbackImageDataset(torch.utils.data.Dataset):
    """Dataset dự phòng khi chạy smoke/test trên máy local chưa tải thư mục ảnh RGB."""

    def __init__(self, fruit_ids, labels, image_size: int = IMAGE_SIZE):
        self.fruit_ids = list(fruit_ids)
        self.labels = [int(y) for y in labels]
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.fruit_ids)

    def __getitem__(self, idx: int):
        img = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
        return img, self.labels[idx], self.fruit_ids[idx], {"camera_angle": "NONE"}


def build_vision_eval_dataset(
    manifest_path: Union[str, Path],
    data_dir: Union[str, Path],
    split: str,
    image_size: int = IMAGE_SIZE,
):
    """Tạo `ImageDataset` chỉ chứa các mẫu có ảnh RGB thật cho tập `valid` hoặc `test`."""
    resolved_split = split_dir_name(split)
    manifest = pd.read_csv(resolve_path(manifest_path))
    fruits = manifest[manifest["Split"] == resolved_split].sort_values("fruit_id")
    label_map = dict(zip(fruits["fruit_id"], fruits["label_idx"].astype(int)))
    rgb_dir = resolve_path(data_dir) / resolved_split / "rgb"
    records = list_rgb_images(rgb_dir)
    ds = ImageDataset(records, label_map, transform=get_eval_transforms(image_size))
    if len(ds) == 0:
        logger.warning(f"Không tìm thấy ảnh RGB tại {rgb_dir}; dùng fallback từ manifest ({len(fruits)} quả).")
        return _ManifestFallbackImageDataset(fruits["fruit_id"], fruits["label_idx"], image_size=image_size)
    return ds


def evaluate_student_split(
    ckpt_path: Union[str, Path],
    manifest_path: Union[str, Path],
    data_dir: Union[str, Path],
    split: str,
    output_dir: Union[str, Path],
    batch_size: int = 16,
    num_workers: int = 0,
    image_size: int = IMAGE_SIZE,
    device: Union[str, torch.device] = "cpu",
    smoke: bool = False,
    seed: int = 42,
    model: Optional[VisionStudent] = None,
    fruit_aggregation_strategy: str = "angle_weighted_prob",
) -> Dict:
    """Chấm điểm mô hình VisionStudent trên tập `valid` hoặc `test` chỉ với ảnh RGB."""
    ckpt_path = resolve_path(ckpt_path)
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)

    ds = build_vision_eval_dataset(manifest_path, data_dir, split=split, image_size=image_size)
    if smoke:
        ds = Subset(ds, list(range(min(len(ds), batch_size * 2))))

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device_obj.type == "cuda",
        worker_init_fn=worker_init_fn,
    )
    if model is None:
        model = build_student_from_checkpoint(ckpt_path, device=device_obj, eval_mode=True)
    else:
        model.to(device_obj).eval()

    resolved_split = split_dir_name(split)
    split_prefix = ("smoke_" if smoke else "") + ("val" if resolved_split != "test" else "test")
    cfg = TeacherEvaluatorConfig(
        teacher_checkpoint_path=str(ckpt_path),
        manifest_path=str(resolve_path(manifest_path)),
        test_data_dir=str(resolve_path(data_dir) / resolved_split),
        output_dir=str(output_dir),
        batch_size=batch_size,
        device=str(device_obj),
        input_mode="vision",
        split_name=split_prefix,
        model_name="vision_student",
        fruit_aggregation_strategy=fruit_aggregation_strategy,
        seed=seed,
    )
    evaluator = StudentEvaluator(cfg, model=model, test_loader=loader)
    return evaluator.run_full_evaluation_pipeline(load_checkpoint=False)


def generate_benchmark_comparison(
    student_results_dir: Union[str, Path],
    project_root: Path = PROJECT_ROOT,
) -> Optional[pd.DataFrame]:
    """Tạo bảng đối sánh toàn diện giữa Student, Teacher và 3 Unimodal Baselines."""
    student_results_dir = resolve_path(student_results_dir)
    rows = []

    # 1. Vision Unimodal Baseline
    v_csv = project_root / "workspaces" / "member1_vision_lead" / "02_train_unimodal_vision" / "results" / "test_predictions_detailed.csv"
    if v_csv.is_file():
        v_df = pd.read_csv(v_csv)
        v_s_acc = float(v_df["is_correct"].mean())
        v_s_f1 = float(f1_score(v_df["true_label"], v_df["pred_label"], average="macro"))
        v_fr = v_df.groupby("fruit_id").agg({
            "prob_Unripe": "mean",
            "prob_Ripe": "mean",
            "prob_Overripe": "mean",
            "true_label": "first",
        })
        v_fr["pred"] = v_fr[["prob_Unripe", "prob_Ripe", "prob_Overripe"]].idxmax(axis=1).str.replace("prob_", "")
        v_f_acc = float(accuracy_score(v_fr["true_label"], v_fr["pred"]))
        v_f_f1 = float(f1_score(v_fr["true_label"], v_fr["pred"], average="macro"))
        rows.append({
            "Model": "1. Vision Unimodal (ConvNeXt-Base)",
            "Test Input": "RGB Only",
            "Sample Acc": v_s_acc,
            "Sample Macro-F1": v_s_f1,
            "Fruit Acc": v_f_acc,
            "Fruit Macro-F1": v_f_f1,
        })

    # 2. Sensor Unimodal
    s_json = project_root / "workspaces" / "member3_sensor" / "02_train_unimodal_sensor" / "results" / "official_summary.json"
    if s_json.is_file():
        s_data = json.loads(s_json.read_text(encoding="utf-8")).get("test", {})
        rows.append({
            "Model": "2. Sensor Unimodal (Integrated MLP)",
            "Test Input": "Sensor (19 fts)",
            "Sample Acc": s_data.get("accuracy"),
            "Sample Macro-F1": s_data.get("macro_f1"),
            "Fruit Acc": s_data.get("accuracy"),
            "Fruit Macro-F1": s_data.get("macro_f1"),
        })

    # 3. Audio Unimodal
    a_json = project_root / "workspaces" / "member2_audio" / "02_train_unimodal_audio" / "results" / "test_metrics.json"
    if a_json.is_file():
        a_m = json.loads(a_json.read_text(encoding="utf-8")).get("metrics", {})
        rows.append({
            "Model": "3. Audio Unimodal (ResNet-18, 20 fruits)",
            "Test Input": "Audio Only",
            "Sample Acc": a_m.get("sample_accuracy"),
            "Sample Macro-F1": a_m.get("sample_macro_f1"),
            "Fruit Acc": a_m.get("fruit_accuracy"),
            "Fruit Macro-F1": a_m.get("fruit_macro_f1"),
        })

    # 4. Multi-Modal Teacher
    t_json = project_root / "workspaces" / "member1_vision_lead" / "04_train_teacher_and_student" / "results" / "test_metrics.json"
    if t_json.is_file():
        t_m = json.loads(t_json.read_text(encoding="utf-8")).get("metrics", {})
        rows.append({
            "Model": "4. Multi-Modal Teacher (Cross-Attention)",
            "Test Input": "RGB + Audio + Sensor",
            "Sample Acc": t_m.get("sample_accuracy"),
            "Sample Macro-F1": t_m.get("sample_macro_f1"),
            "Fruit Acc": t_m.get("fruit_accuracy"),
            "Fruit Macro-F1": t_m.get("fruit_macro_f1"),
        })

    # 5. Vision-Only Student (CMKD)
    st_json = student_results_dir / "test_metrics.json"
    if st_json.is_file():
        st_m = json.loads(st_json.read_text(encoding="utf-8")).get("metrics", {})
        rows.append({
            "Model": "5. Vision-Only Student (CMKD Distilled)",
            "Test Input": "RGB Only",
            "Sample Acc": st_m.get("sample_accuracy"),
            "Sample Macro-F1": st_m.get("sample_macro_f1"),
            "Fruit Acc": st_m.get("fruit_accuracy"),
            "Fruit Macro-F1": st_m.get("fruit_macro_f1"),
        })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df.to_csv(student_results_dir / "benchmark_comparison.csv", index=False)

    md_lines = [
        "# Bảng đối sánh hiệu năng CMKD (Test Set)",
        "",
        "| Mô hình | Đầu vào khi Test | Sample Acc | Sample Macro-F1 | Fruit Acc | Fruit Macro-F1 |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        s_acc = f"{r['Sample Acc'] * 100:.2f}%" if r["Sample Acc"] is not None else "—"
        s_f1 = f"{r['Sample Macro-F1'] * 100:.2f}%" if r["Sample Macro-F1"] is not None else "—"
        f_acc = f"{r['Fruit Acc'] * 100:.2f}%" if r["Fruit Acc"] is not None else "—"
        f_f1 = f"{r['Fruit Macro-F1'] * 100:.2f}%" if r["Fruit Macro-F1"] is not None else "—"
        md_lines.append(
            f"| **{r['Model']}** | {r['Test Input']} | {s_acc} | {s_f1} | {f_acc} | {f_f1} |"
        )
    (student_results_dir / "benchmark_comparison.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Đánh giá độc lập mô hình Vision-Only Student (Giai đoạn 2)")
    p.add_argument("--checkpoint", default="checkpoints/studentmodal/best_student.pth")
    p.add_argument("--manifest", default="data/final/final_multimodal_manifest.csv")
    p.add_argument("--data-dir", default="data/final")
    p.add_argument("--split", default="test", choices=["val", "valid", "test"])
    p.add_argument("--output-dir", default="workspaces/member1_vision_lead/04_train_teacher_and_student/results_student")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    p.add_argument("--fruit-agg", default="angle_weighted_prob",
                   choices=["angle_weighted_prob", "mean_prob", "majority_vote"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    res = evaluate_student_split(
        ckpt_path=args.checkpoint,
        manifest_path=args.manifest,
        data_dir=args.data_dir,
        split=args.split,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        device=args.device,
        smoke=args.smoke,
        seed=args.seed,
        fruit_aggregation_strategy=args.fruit_agg,
    )
    m = res["metrics"]
    print(
        f"[{args.split.upper()}] Student | Sample Acc: {m['sample_accuracy']:.4f} | "
        f"Sample Macro-F1: {m['sample_macro_f1']:.4f} | "
        f"Fruit Acc: {m['fruit_accuracy']:.4f} | Fruit Macro-F1: {m['fruit_macro_f1']:.4f}"
    )
    if args.split == "test" and not args.smoke:
        generate_benchmark_comparison(args.output_dir)


if __name__ == "__main__":
    main()
