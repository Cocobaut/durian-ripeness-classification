"""Cầu nối đánh giá mô hình Multi-Modal Teacher trong `src/evaluators/`.

Tái xuất (re-export) `TeacherEvaluator`, `TeacherEvaluatorConfig`, `evaluate` từ
`workspaces/member2_audio/03_test_eval_teacher_and_gradcam/eval_teacher.py`
đồng thời hỗ trợ chạy CLI trực tiếp với kiến trúc `MultiModalTeacher` và `MultimodalDataset`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from torch.utils.data import DataLoader

from src.datasets.multimodal_dataset import build_multimodal_datasets, worker_init_fn
from src.models.teacher_model import build_teacher_from_checkpoint
from src.utils.common import PROJECT_ROOT, add_to_sys_path, resolve_path

add_to_sys_path(PROJECT_ROOT / "workspaces" / "member2_audio" / "03_test_eval_teacher_and_gradcam")
from eval_teacher import (  # noqa: E402, F401
    CLASS_ORDER,
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    NUM_CLASSES,
    TeacherErrorAnalyzer,
    TeacherEvaluator,
    TeacherEvaluatorConfig,
    TeacherGradCAMVisualizer,
    evaluate,
    parse_args,
)


def run_cli() -> dict:
    """Chạy đánh giá Teacher hoàn chỉnh từ dòng lệnh."""
    args = parse_args()
    ckpt_path = resolve_path(args.checkpoint)
    manifest_path = resolve_path(args.manifest)
    data_dir = resolve_path(args.test_dir).parent
    split = "valid" if args.split_name in ("val", "valid") else "test"

    datasets = build_multimodal_datasets(manifest_path, data_dir, splits=(split,))
    loader = DataLoader(
        datasets[split],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=worker_init_fn,
    )
    model = build_teacher_from_checkpoint(ckpt_path, device=args.device, eval_mode=True)
    config = TeacherEvaluatorConfig(
        teacher_checkpoint_path=str(ckpt_path),
        manifest_path=str(manifest_path),
        test_data_dir=str(resolve_path(args.test_dir)),
        output_dir=str(resolve_path(args.output_dir)),
        batch_size=args.batch_size,
        device=args.device,
        input_mode=args.input_mode,
        split_name=args.split_name,
        model_name=args.model_name,
    )
    evaluator = TeacherEvaluator(config=config, model=model, test_loader=loader)
    return evaluator.run_full_evaluation_pipeline(load_checkpoint=False)


if __name__ == "__main__":
    run_cli()

