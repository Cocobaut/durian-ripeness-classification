"""Independent evaluation of Vision-Only Student (CMKD Stage 2) at Sample & Fruit levels.

Project: Durian Ripeness Classification via Cross-Modal Knowledge Distillation (CMKD)
Specification reference: Sections 1.3.3 & 1.4.

Key characteristics:
- During evaluation / deployment inference, Student ONLY RECEIVES RGB IMAGES (`ImageDataset`, `input_mode="vision"`).
- Does not require a microphone, physical sensors, or the Teacher model.
- Automatically builds `benchmark_comparison.csv` & `benchmark_comparison.md` comparing:
  Vision Unimodal vs Audio Unimodal vs Sensor Unimodal vs Multi-Modal Teacher vs Vision-Only Student.

Usage (from project root):
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
from typing import Dict, List, Optional, Sequence, Union

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from eval_teacher import TeacherEvaluator, TeacherEvaluatorConfig  # noqa: E402

logger = logging.getLogger("eval_student")


class StudentEnsemble(torch.nn.Module):
    """Average the softmax probabilities of several Students (e.g. different seeds).

    Returns log(mean probs) so downstream code that applies softmax recovers the averaged probabilities.
    """

    def __init__(self, members: List[VisionStudent]) -> None:
        super().__init__()
        self.members = torch.nn.ModuleList(members)

    @property
    def tta_enabled(self) -> bool:
        return all(m.tta_enabled for m in self.members)

    @tta_enabled.setter
    def tta_enabled(self, value: bool) -> None:
        for m in self.members:
            m.tta_enabled = bool(value)

    def forward(self, img: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        probs = torch.stack([torch.softmax(m(img, *args, **kwargs).float(), dim=-1) for m in self.members])
        return torch.log(probs.mean(dim=0).clamp_min(1e-12))


def load_students(
    ckpt_paths: Union[str, Path, Sequence[Union[str, Path]]],
    device: torch.device,
) -> torch.nn.Module:
    """One checkpoint -> VisionStudent; several -> StudentEnsemble."""
    paths = [ckpt_paths] if isinstance(ckpt_paths, (str, Path)) else list(ckpt_paths)
    models = [build_student_from_checkpoint(resolve_path(p), device=device, eval_mode=True) for p in paths]
    if len(models) == 1:
        return models[0]
    logger.info(f"Ensembling {len(models)} Students: {[str(p) for p in paths]}")
    return StudentEnsemble(models).to(device).eval()


class _ManifestFallbackImageDataset(torch.utils.data.Dataset):
    """Fallback multi-view dataset for smoke/test runs on local machines without downloaded RGB images."""

    def __init__(self, fruit_ids, labels, image_size: int = IMAGE_SIZE):
        self.fruit_ids = list(fruit_ids)
        self.labels = [int(y) for y in labels]
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.fruit_ids)

    def __getitem__(self, idx: int):
        img = torch.zeros(1, 3, self.image_size, self.image_size, dtype=torch.float32)
        return img, self.labels[idx], self.fruit_ids[idx], {"camera_angle": "MULTI_VIEW"}


def build_vision_eval_dataset(
    manifest_path: Union[str, Path],
    data_dir: Union[str, Path],
    split: str,
    image_size: int = IMAGE_SIZE,
):
    """Build Multi-View `ImageDataset` grouping all views of each fruit for `valid` or `test`."""
    resolved_split = split_dir_name(split)
    manifest = pd.read_csv(resolve_path(manifest_path))
    fruits = manifest[manifest["Split"] == resolved_split].sort_values("fruit_id")
    label_map = dict(zip(fruits["fruit_id"], fruits["label_idx"].astype(int)))
    rgb_dir = resolve_path(data_dir) / resolved_split / "rgb"
    records = list_rgb_images(rgb_dir)
    ds = ImageDataset(
        records,
        label_map,
        transform=get_eval_transforms(image_size),
        multiview=True,
        is_train=False,
        image_size=image_size,
    )
    if len(ds) == 0:
        logger.warning(f"No RGB images found at {rgb_dir}; using fallback from manifest ({len(fruits)} fruits).")
        return _ManifestFallbackImageDataset(fruits["fruit_id"], fruits["label_idx"], image_size=image_size)
    return ds


def evaluate_student_split(
    ckpt_path: Union[str, Path, Sequence[Union[str, Path]]],
    manifest_path: Union[str, Path],
    data_dir: Union[str, Path],
    split: str,
    output_dir: Union[str, Path],
    batch_size: int = 4,
    num_workers: int = 0,
    image_size: int = IMAGE_SIZE,
    device: Union[str, torch.device] = "cpu",
    smoke: bool = False,
    seed: int = 42,
    model: Optional[torch.nn.Module] = None,
    tta: bool = True,
) -> Dict:
    """Evaluate Multi-View VisionStudent on `valid` or `test` split using RGB image bags only.

    `tta=True` applies the same per-view 3-crop/flip TTA as the unimodal Vision baseline evaluation
    (0.5 * original + 0.3 * hflip + 0.2 * 95% zoom), so Student and baseline are compared fairly.
    """
    ckpt_list = [ckpt_path] if isinstance(ckpt_path, (str, Path)) else list(ckpt_path)
    ckpt_path = resolve_path(ckpt_list[0])
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
        model = load_students(ckpt_list, device_obj)
    else:
        model.to(device_obj).eval()
    model.tta_enabled = bool(tta)
    logger.info(f"Student evaluation on '{split}' | TTA: {bool(tta)}")

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
        seed=seed,
    )
    evaluator = TeacherEvaluator(cfg, model=model, test_loader=loader)
    return evaluator.run_full_evaluation_pipeline(load_checkpoint=False)


def generate_benchmark_comparison(
    student_results_dir: Union[str, Path],
    project_root: Path = PROJECT_ROOT,
) -> Optional[pd.DataFrame]:
    """Generate comprehensive comparison table across Student, Teacher, and 3 Unimodal Baselines."""
    student_results_dir = resolve_path(student_results_dir)
    rows = []

    # 1. Vision Unimodal Baseline (prefers Multi-View ConvNeXtV2-Base Gated Attention summary if present)
    v_dir = project_root / "workspaces" / "member1_vision_lead" / "02_train_unimodal_vision" / "results"
    v_mv_json = v_dir / "test_metrics_summary_multiview.json"
    v_csv = v_dir / "test_predictions_detailed.csv"
    if v_mv_json.is_file():
        v_data = json.loads(v_mv_json.read_text(encoding="utf-8"))
        rows.append({
            "Model": "1. Vision Unimodal (ConvNeXtV2-Base Multi-View)",
            "Test Input": "Multi-View RGB",
            "Sample Acc": v_data.get("image_level_accuracy"),
            "Sample Macro-F1": v_data.get("image_level_macro_f1"),
            "Fruit Acc": v_data.get("fruit_level_accuracy"),
            "Fruit Macro-F1": v_data.get("fruit_level_macro_f1"),
        })
    elif v_csv.is_file():
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
    if not t_json.is_file():
        t_json = project_root / "workspaces" / "member1_vision_lead" / "04_train_teacher_and_student" / "results_teacher" / "test_metrics.json"
    if t_json.is_file():
        t_m = json.loads(t_json.read_text(encoding="utf-8")).get("metrics", {})
        rows.append({
            "Model": "4. Multi-Modal Teacher (Cross-Attention)",
            "Test Input": "Multi-View RGB + Audio + Sensor",
            "Sample Acc": t_m.get("sample_accuracy"),
            "Sample Macro-F1": t_m.get("sample_macro_f1"),
            "Fruit Acc": t_m.get("fruit_accuracy"),
            "Fruit Macro-F1": t_m.get("fruit_macro_f1"),
        })

        # 4b. Teacher restricted to the fruits the Student is evaluated on (fruits with RGB images),
        #     so Teacher vs Student is compared on the same fruit set.
        t_fruit_csv = t_json.parent / "test_fruit_predictions.csv"
        s_fruit_csv = student_results_dir / "test_fruit_predictions.csv"
        if t_fruit_csv.is_file() and s_fruit_csv.is_file():
            t_fr = pd.read_csv(t_fruit_csv)
            s_ids = set(pd.read_csv(s_fruit_csv)["fruit_id"])
            t_common = t_fr[t_fr["fruit_id"].isin(list(s_ids))]
            if len(t_common) > 0 and len(t_common) < len(t_fr):
                rows.append({
                    "Model": f"4b. Multi-Modal Teacher (same {len(t_common)} RGB fruits as Student)",
                    "Test Input": "Multi-View RGB + Audio + Sensor",
                    "Sample Acc": None,
                    "Sample Macro-F1": None,
                    "Fruit Acc": float(accuracy_score(t_common["y_true"], t_common["y_pred"])),
                    "Fruit Macro-F1": float(f1_score(t_common["y_true"], t_common["y_pred"], average="macro",
                                                     labels=[0, 1, 2], zero_division=0)),
                })

    # 5. Vision-Only Student (CMKD)
    st_json = student_results_dir / "test_metrics.json"
    if st_json.is_file():
        st_m = json.loads(st_json.read_text(encoding="utf-8")).get("metrics", {})
        rows.append({
            "Model": "5. Vision-Only Student (CMKD Multi-View)",
            "Test Input": "Multi-View RGB Only",
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
        "# CMKD Performance Benchmark Comparison (Test Set)",
        "",
        "| Model | Test Input | Sample Acc | Sample Macro-F1 | Fruit Acc | Fruit Macro-F1 |",
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
    p = argparse.ArgumentParser(description="Independent evaluation of Multi-View Vision-Only Student model (Stage 2)")
    p.add_argument("--checkpoint", nargs="+", default=["checkpoints/studentmodal/best_student.pth"],
                   help="One Student checkpoint, or several to evaluate their probability-averaged ensemble")
    p.add_argument("--manifest", default="data/final/final_multimodal_manifest.csv")
    p.add_argument("--data-dir", default="data/final")
    p.add_argument("--split", default="test", choices=["val", "valid", "test"])
    p.add_argument("--output-dir", default="workspaces/member1_vision_lead/04_train_teacher_and_student/results_student")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-tta", action="store_true", help="Disable 3-view TTA (orig + hflip + 95%% zoom)")
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
        tta=not args.no_tta,
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

