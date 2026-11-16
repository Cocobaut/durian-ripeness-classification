"""Evaluation module for the Multi-Modal Teacher and explainability visualization (Grad-CAM / XAI).

Responsible: Member 2 (Audio & Multi-Modal Evaluation / XAI)
Project: Durian Ripeness Classification via Cross-Modal Knowledge Distillation (CMKD)
Specification reference:
- Section 1.3.2: Stage 1 - Unimodal Baselines and Multi-Modal Teacher construction.
- Section 1.4 (Objectives 2 & 4): Comprehensive Benchmark Suite (Unimodal vs. Multimodal)
  and Visual Explainability (Explainable AI - XAI) with Physical Limitation Analysis.

Module responsibilities:
1. Independently evaluate the Teacher model (fusing 3 branches: RGB Image + Knock Sound + Sensor/Tabular)
   on the held-out Test Set to establish the "Performance Upper Bound".
2. Evaluate modality contributions (Modality Ablation / Zero-Masking):
   Measure performance degradation when zero-masking each branch (w/o Audio, w/o Sensor, w/o RGB)
   to quantify the value of acoustic and sensor cues for ripeness classification.
3. Aggregate results at 2 levels: Sample-level and Fruit-level.
4. Provide Grad-CAM hooks on the Teacher's vision branch for visual explanation:
   Inspect which morphological regions (spine grooves, pedicel disc, shell) the network attends to
   when guided by knock acoustics and sensor measurements.
5. In-depth Error Analysis: examine misclassified cases and correlate them with
   physical properties (Firmness, TSS, knock acoustic characteristics).

NOTES FOR ASSIGNED MEMBER:
- This file provides the object-oriented scaffolding (classes, method signatures,
  docstrings, and sequential implementation guidelines).
- Read the TODO comments in each method to complete remaining XAI/Error Analysis extensions.
- Standard metrics are provided; additional domain-specific metrics for multi-class / imbalanced
  classification and Grad-CAM evaluation can be added as needed.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from torch.utils.data import DataLoader, Dataset

# Configure logging for evaluation tracking
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eval_teacher")

# ------------------------------------------------------------------------------
# 1. PROJECT CONSTANTS
# ------------------------------------------------------------------------------
CLASS_ORDER = ["Unripe", "Ripe", "Overripe"]  # 0: Unripe, 1: Ripe, 2: Overripe
LABEL_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASS_ORDER)}
IDX_TO_LABEL = {idx: cls_name for idx, cls_name in enumerate(CLASS_ORDER)}
NUM_CLASSES = len(CLASS_ORDER)


# ------------------------------------------------------------------------------
# 2. TEACHER EVALUATOR CONFIGURATION (DATACLASS CONFIG)
# ------------------------------------------------------------------------------
INPUT_MODES = ("multimodal", "audio", "vision", "sensor")
ABLATION_MODALITIES = ("audio", "sensor", "vision")


@dataclass
class TeacherEvaluatorConfig:
    """Parameter configuration for Multi-Modal Teacher evaluation and Grad-CAM.

    Paths and hyperparameters can be customized to match the target execution
    environment or extended with additional flags.
    """
    # Path to best trained Teacher checkpoint
    teacher_checkpoint_path: str = "checkpoints/teacher_best.pth"

    # Path to clean master multimodal manifest CSV
    manifest_path: str = "data/final/final_multimodal_manifest.csv"

    # Directory containing standardized Test Set
    test_data_dir: str = "data/final/test"

    # Output directory for evaluation results (metrics tables, confusion matrices, Grad-CAM images)
    output_dir: str = "workspaces/member2_audio/03_test_eval_teacher_and_gradcam/results"

    # Hardware & data loading configuration
    batch_size: int = 16
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Final convolutional layer name in the Teacher's Vision branch for Grad-CAM
    # Hint: e.g., for ResNet50: 'vision_backbone.layer4'
    # or EfficientNet: 'vision_backbone.features.7'
    gradcam_target_layer: str = "vision_backbone.layer4"

    # List of fruit camera angles for angle-wise consistency evaluation
    camera_angles: List[str] = field(
        default_factory=lambda: ["TOP", "BOT", "SA", "SB", "SC", "SD"]
    )

    # Fruit-level aggregation strategy across angles/knocks ('mean_prob' or 'majority_vote')
    fruit_aggregation_strategy: str = "mean_prob"

    # [ADDED] Input mode of the evaluated model:
    # - "multimodal": batch = (img, audio, sensor, label, fruit_id[, extra]) and calls model(images, audios, sensors)
    # - "audio" / "vision" / "sensor": batch = (x, label, fruit_id[, extra]) or dict {x, label, fruit_id, ...}, calls model(x)
    input_mode: str = "multimodal"

    # [ADDED] Dataset split prefix used in output filenames ("val", "test", ...)
    split_name: str = "test"

    # [ADDED] Model name recorded in metrics tables (for cross-branch comparison)
    model_name: str = "teacher"

    # [ADDED] Number of bins for Expected Calibration Error
    ece_bins: int = 15


# ------------------------------------------------------------------------------
# 3. MAIN CLASS: TEACHER EVALUATOR
# ------------------------------------------------------------------------------
class TeacherEvaluator:
    """Comprehensive evaluator for the Multi-Modal Teacher fusion model.

    Core responsibilities:
    1. Load the trained Teacher model and restore optimal checkpoint weights.
    2. Run inference across the entire held-out Test Set.
    3. Compute classification metrics at 2 levels:
       - Sample-level / Per-sample evaluation.
       - Fruit-level aggregated evaluation.
    4. Assess modality contributions via Zero-Masking / Modality Ablation:
       Measure performance drop when masking Audio, Sensor, or Vision branches.
    5. Export detailed reports to CSV, JSON, logs, and confusion matrix plots.

    Also supports evaluating UNIMODAL models (e.g., Week 2 audio branch) by setting
    `config.input_mode = "audio"` (skips modality ablation).
    """

    # Accepted dictionary keys when batch is a dict
    _KEYS = {
        "image": ("image", "images", "img", "rgb"),
        "audio": ("audio", "audios", "mel", "audio_mel"),
        "sensor": ("sensor", "sensors", "tabular"),
        "x": ("x", "input", "inputs"),
        "label": ("label", "labels", "y", "target"),
        "fruit_id": ("fruit_id", "fruit_ids"),
    }

    def __init__(
        self,
        config: TeacherEvaluatorConfig,
        model: Optional[nn.Module] = None,
        test_loader: Optional[DataLoader] = None,
    ) -> None:
        """Initialize the Teacher evaluator.

        Args:
            config: Configuration dataclass containing paths and hyperparameters.
            model: Initialized Multi-Modal Teacher instance (if None, loaded from checkpoint).
            test_loader: PyTorch DataLoader for the evaluation split.
        """
        if config.input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode must be one of {INPUT_MODES}, got: {config.input_mode!r}")
        self.config = config
        self.device = torch.device(config.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = model
        if self.model is not None:
            self.model.to(self.device)
        self.test_loader = test_loader

        logger.info(f"Initialized TeacherEvaluator on device: {self.device} (input_mode={config.input_mode})")
        logger.info(f"Evaluation results will be saved to: {self.output_dir}")

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _file_prefix(self, split_name: Optional[str] = None) -> str:
        name = self.config.split_name if split_name is None else split_name
        return f"{name}_" if name else ""

    @staticmethod
    def _to_list(value: Any, n: int) -> List[Any]:
        """Convert tensor / ndarray / list / scalar into a Python list of length n."""
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            value = value.tolist() if value.ndim > 0 else [value.item()] * n
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, (str, int, float)) or value is None:
            value = [value] * n
        value = list(value)
        if len(value) != n:
            raise ValueError(f"Auxiliary field length ({len(value)}) differs from batch size ({n}).")
        return value

    @classmethod
    def _pick(cls, batch: Dict[str, Any], name: str) -> Any:
        for key in cls._KEYS[name]:
            if key in batch:
                return batch[key]
        return None

    def _unpack_batch(self, batch: Any) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, Any, Dict[str, Any]]:
        """Unpack batch into (inputs, labels, fruit_ids, extras) according to `config.input_mode`."""
        multimodal = self.config.input_mode == "multimodal"
        extras: Dict[str, Any] = {}
        if isinstance(batch, dict):
            if multimodal:
                inputs = tuple(self._pick(batch, k) for k in ("image", "audio", "sensor"))
            else:
                x = self._pick(batch, "x")
                if x is None:
                    x = self._pick(batch, self.config.input_mode)
                inputs = (x,)
            labels, fruit_ids = self._pick(batch, "label"), self._pick(batch, "fruit_id")
            used = {k for names in self._KEYS.values() for k in names}
            extras = {k: v for k, v in batch.items() if k not in used}
        elif isinstance(batch, (list, tuple)):
            n_in = 3 if multimodal else 1
            if len(batch) < n_in + 2:
                raise ValueError(
                    f"Tuple batch requires at least {n_in + 2} items for input_mode={self.config.input_mode!r}, "
                    f"got {len(batch)}."
                )
            inputs = tuple(batch[:n_in])
            labels, fruit_ids = batch[n_in], batch[n_in + 1]
            for i, extra in enumerate(batch[n_in + 2:]):
                if isinstance(extra, dict):
                    extras.update(extra)
                else:
                    extras[f"extra_{i}"] = extra
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")
        if labels is None or fruit_ids is None or any(t is None for t in inputs):
            raise ValueError("Batch is missing input, label, or fruit_id.")
        return inputs, labels, fruit_ids, extras

    @staticmethod
    def _extract_logits(output: Any) -> torch.Tensor:
        """Extract logits from model output: tensor, (logits, features), or dict with 'logits' key."""
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (list, tuple)) and len(output) > 0:
            return output[0]
        if isinstance(output, dict) and "logits" in output:
            return output["logits"]
        raise TypeError(f"Could not extract logits from output of type {type(output)}")

    # ------------------------------------------------------------------
    # 3.1. Model loading
    # ------------------------------------------------------------------
    def load_teacher_model(self, checkpoint_path: Optional[str] = None) -> nn.Module:
        """Load Teacher model and restore state_dict from checkpoint path.

        Implementation notes:
        ---------------------------------------------------------
        1. Instantiate the Teacher architecture matching the training configuration.
        2. Load checkpoint dictionary via `torch.load(path, map_location=self.device)`.
        3. Extract state_dict (stripping 'module.' prefix if trained with DataParallel).
        4. Load weights into the model instance and freeze parameters in eval mode.

        Note: since the Teacher architecture is defined by Lead, this method does not guess
        the architecture. Pass an initialized model via `TeacherEvaluator(config, model=...)`
        and this method loads weights into it. Supports raw state_dict or dicts with 'model_state_dict'.

        Args:
            checkpoint_path: Path to .pth file (defaults to self.config.teacher_checkpoint_path).

        Returns:
            nn.Module: Teacher model ready for inference.
        """
        ckpt_path = checkpoint_path or self.config.teacher_checkpoint_path
        logger.info(f"Loading Teacher weights from: {ckpt_path}")

        if self.model is None:
            raise RuntimeError(
                "No model instance provided. Initialize the architecture (e.g., MultiModalTeacher) and pass it to "
                "TeacherEvaluator(config, model=model) before calling load_teacher_model()."
            )
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state_dict = checkpoint[key]
                    break
        state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"State dict does not match architecture: missing {len(missing)} keys (e.g., {missing[:3]}), "
                f"unexpected {len(unexpected)} keys (e.g., {unexpected[:3]})."
            )
        self.model.to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("Weights loaded successfully.")
        return self.model

    # ------------------------------------------------------------------
    # 3.2. Sample-level inference
    # ------------------------------------------------------------------
    def evaluate_sample_level(
        self,
        test_loader: Optional[DataLoader] = None,
        mask_modality: Optional[str] = None,
    ) -> pd.DataFrame:
        """Run evaluation loop on individual samples (sample-level evaluation).

        Each durian fruit may have 6 camera views (TOP, BOT, SA...), multiple knock audio
        recordings (Android, iPhone), and sensor parameters. Each combination fed into the
        network forms one sample.

        Accepted batch formats:
        - input_mode="multimodal": `(images, audios, sensors, labels, fruit_ids[, extra...])` or dict.
        - unimodal input_mode: `(x, labels, fruit_ids[, extra...])` or dict `{x, label, fruit_id, ...}`.
        Extra fields (e.g., `device`, `camera_angle`, `file`) are added as columns in the DataFrame.

        Args:
            test_loader: Evaluation DataLoader.
            mask_modality: Modality name to zero-mask for ablation ('audio', 'sensor', 'vision', or None).

        Returns:
            pd.DataFrame: Detailed per-sample predictions with columns
            `['fruit_id', 'camera_angle', 'true_label', 'pred_label', 'prob_unripe', 'prob_ripe',
            'prob_overripe', 'is_correct', 'y_true', 'y_pred']` plus extra columns.
        """
        logger.info(f"Starting sample-level evaluation (mask_modality={mask_modality})...")
        loader = test_loader or self.test_loader
        if loader is None:
            raise ValueError("No DataLoader provided for evaluation.")
        if self.model is None:
            raise RuntimeError("No model provided for evaluation.")
        if mask_modality is not None and mask_modality not in ABLATION_MODALITIES:
            raise ValueError(f"mask_modality must be one of {ABLATION_MODALITIES} or None.")

        multimodal = self.config.input_mode == "multimodal"
        self.model.eval()
        rows: List[Dict[str, Any]] = []
        with torch.no_grad():
            for batch in loader:
                inputs, labels, fruit_ids, extras = self._unpack_batch(batch)
                inputs = [t.to(self.device, non_blocking=True).float() for t in inputs]
                if mask_modality is not None:
                    if multimodal:
                        idx = {"vision": 0, "audio": 1, "sensor": 2}[mask_modality]
                        inputs[idx] = torch.zeros_like(inputs[idx])
                    elif mask_modality == self.config.input_mode:
                        inputs[0] = torch.zeros_like(inputs[0])

                logits = self._extract_logits(self.model(*inputs)).float()
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                preds = probs.argmax(axis=1)

                n = probs.shape[0]
                y_true = self._to_list(labels, n)
                fids = self._to_list(fruit_ids, n)
                extra_lists = {k: self._to_list(v, n) for k, v in extras.items()}
                for i in range(n):
                    row = {
                        "fruit_id": fids[i],
                        "camera_angle": extra_lists.get("camera_angle", [""] * n)[i],
                        "true_label": IDX_TO_LABEL[int(y_true[i])],
                        "pred_label": IDX_TO_LABEL[int(preds[i])],
                        "prob_unripe": float(probs[i, 0]),
                        "prob_ripe": float(probs[i, 1]),
                        "prob_overripe": float(probs[i, 2]),
                        "is_correct": bool(int(y_true[i]) == int(preds[i])),
                        "y_true": int(y_true[i]),
                        "y_pred": int(preds[i]),
                    }
                    for k, v in extra_lists.items():
                        if k != "camera_angle":
                            row[k] = v[i]
                    rows.append(row)

        sample_df = pd.DataFrame(rows)
        if sample_df.empty:
            raise ValueError("DataLoader yielded 0 samples.")
        logger.info(f"Evaluated {len(sample_df)} samples, accuracy = {sample_df['is_correct'].mean():.4f}")
        return sample_df

    # ------------------------------------------------------------------
    # 3.3. Fruit-level aggregation
    # ------------------------------------------------------------------
    def evaluate_fruit_level(
        self,
        sample_df: pd.DataFrame,
        strategy: Optional[str] = None,
    ) -> pd.DataFrame:
        """Aggregate sample-level predictions (multiple views/knocks) into fruit-level predictions (fruit_id).

        In real harvest and commercial grading, decisions are made per fruit rather than
        per individual camera frame. Fruit-level evaluation reflects true system accuracy.

        Strategies:
        - 'mean_prob' (recommended): average Softmax probability vectors across samples of a fruit, take argmax.
        - 'majority_vote': most frequent predicted label; ties broken by highest summed probability.

        Args:
            sample_df: Sample-level prediction DataFrame from `evaluate_sample_level()`.
            strategy: Aggregation strategy ('mean_prob' or 'majority_vote').

        Returns:
            pd.DataFrame: Per-fruit predictions with columns
            `['fruit_id', 'true_label', 'pred_label', 'prob_unripe_mean', 'prob_ripe_mean',
            'prob_overripe_mean', 'is_correct', 'n_samples', 'y_true', 'y_pred']`.
        """
        agg_strategy = strategy or self.config.fruit_aggregation_strategy
        logger.info(f"Aggregating fruit-level predictions (fruit_id) using strategy: {agg_strategy}...")
        if agg_strategy not in ("mean_prob", "majority_vote"):
            raise ValueError("strategy must be 'mean_prob' or 'majority_vote'.")

        prob_cols = ["prob_unripe", "prob_ripe", "prob_overripe"]
        rows = []
        for fruit_id, group in sample_df.groupby("fruit_id", sort=True):
            true_labels = group["true_label"].unique()
            if len(true_labels) > 1:
                logger.warning(f"Fruit {fruit_id} has multiple ground-truth labels {list(true_labels)}; using mode.")
            true_label = group["true_label"].mode().iloc[0]
            mean_probs = group[prob_cols].mean().to_numpy()
            if agg_strategy == "mean_prob":
                pred_idx = int(mean_probs.argmax())
            else:
                votes = np.bincount(group["pred_label"].map(LABEL_TO_IDX).to_numpy(), minlength=NUM_CLASSES)
                candidates = np.flatnonzero(votes == votes.max())
                sums = group[prob_cols].sum().to_numpy()
                pred_idx = int(candidates[np.argmax(sums[candidates])])
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
        logger.info(f"Aggregated {len(fruit_df)} fruits, fruit-level accuracy = {fruit_df['is_correct'].mean():.4f}")
        return fruit_df

    # ------------------------------------------------------------------
    # 3.4. Metrics
    # ------------------------------------------------------------------
    def compute_metrics(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        y_probs: Optional[np.ndarray] = None,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Compute comprehensive classification metrics.

        Computed metrics:
        - Accuracy, Balanced Accuracy, Macro-F1 (primary metric), Weighted-F1,
          Macro Precision / Macro Recall.
        - Per-class Precision / Recall / F1 / Support (Unripe, Ripe, Overripe).
        - Cohen's Kappa, Matthews Correlation Coefficient (MCC).
        - When probabilities are provided: macro OvR ROC-AUC, macro PR-AUC (Average Precision),
          multi-class Log-loss, multi-class Brier score, Expected Calibration Error (ECE).
        Undefined metrics (e.g., missing class in y_true) return NaN with a warning instead of raising.

        Args:
            y_true: Integer ground-truth labels (0, 1, 2).
            y_pred: Integer predicted labels.
            y_probs: Predicted probability matrix of shape (N, NUM_CLASSES).
            prefix: Prefix prepended to metric keys (e.g., 'sample_' or 'fruit_').

        Returns:
            Dict[str, float]: Mapping from metric names to computed values.
        """
        from sklearn.metrics import (
            accuracy_score, average_precision_score, balanced_accuracy_score, cohen_kappa_score,
            f1_score, log_loss, matthews_corrcoef, precision_recall_fscore_support, roc_auc_score,
        )
        import warnings

        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        labels = list(range(NUM_CLASSES))
        metrics: Dict[str, float] = {"n_samples": float(len(y_true))}

        def safe(name: str, fn: Callable[[], float]) -> None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    metrics[name] = float(fn())
            except Exception as exc:  # metric undefined on this data slice
                logger.warning(f"Could not compute {prefix}{name}: {exc}")
                metrics[name] = float("nan")

        safe("accuracy", lambda: accuracy_score(y_true, y_pred))
        safe("balanced_accuracy", lambda: balanced_accuracy_score(y_true, y_pred))
        safe("macro_f1", lambda: f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        safe("weighted_f1", lambda: f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        metrics["macro_precision"] = float(np.mean(p))
        metrics["macro_recall"] = float(np.mean(r))
        for i, cls in enumerate(CLASS_ORDER):
            key = cls.lower()
            metrics[f"precision_{key}"] = float(p[i])
            metrics[f"recall_{key}"] = float(r[i])
            metrics[f"f1_{key}"] = float(f[i])
            metrics[f"support_{key}"] = float(s[i])
        safe("cohen_kappa", lambda: cohen_kappa_score(y_true, y_pred, labels=labels))
        safe("mcc", lambda: matthews_corrcoef(y_true, y_pred))

        if y_probs is not None:
            probs = np.asarray(y_probs, dtype=float)
            probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
            onehot = np.eye(NUM_CLASSES)[y_true]
            present = [c for c in labels if onehot[:, c].any() and not onehot[:, c].all()]

            def roc_auc() -> float:
                if len(present) < NUM_CLASSES:
                    raise ValueError(f"y_true only contains classes {sorted(set(y_true.tolist()))}; macro ROC-AUC undefined")
                return roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=labels)

            def pr_auc() -> float:
                if not present:
                    raise ValueError("no class has both positive and negative samples")
                return float(np.mean([average_precision_score(onehot[:, c], probs[:, c]) for c in present]))

            safe("roc_auc_ovr_macro", roc_auc)
            safe("pr_auc_macro", pr_auc)
            safe("log_loss", lambda: log_loss(y_true, np.clip(probs, 1e-12, 1.0), labels=labels))
            metrics["brier_score"] = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
            metrics["ece"] = self._expected_calibration_error(y_true, probs, self.config.ece_bins)

        return {f"{prefix}{k}": v for k, v in metrics.items()}

    @staticmethod
    def _expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
        """Top-label Expected Calibration Error (ECE) over n_bins uniform bins in [0, 1]."""
        conf = probs.max(axis=1)
        correct = (probs.argmax(axis=1) == y_true).astype(float)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
            if mask.any():
                ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
        return float(ece)

    def _level_metrics(self, df: pd.DataFrame, level: str) -> Dict[str, float]:
        prob_cols = (["prob_unripe", "prob_ripe", "prob_overripe"] if level == "sample"
                     else ["prob_unripe_mean", "prob_ripe_mean", "prob_overripe_mean"])
        return self.compute_metrics(df["y_true"], df["y_pred"], df[prob_cols].to_numpy(), prefix=f"{level}_")

    # ------------------------------------------------------------------
    # 3.5. Ablation
    # ------------------------------------------------------------------
    def evaluate_modality_ablation(
        self,
        test_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate modality contributions (Modality Contribution & Ablation Study).

        Tests Teacher robustness and reliance on each sensory modality:
        1. Full Multi-Modal (RGB + Audio + Sensor): optimal baseline performance.
        2. w/o Audio (Zero-masking audio branch): measures Macro-F1 contribution of knock audio.
        3. w/o Sensor (Zero-masking sensor branch): measures contribution of tabular/thermal/spectral cues.
        4. w/o Vision (Zero-masking RGB branch): checks Teacher performance without visual input.

        Only executed when `config.input_mode == "multimodal"`; returns empty dict for unimodal models.
        Saves results to `<split>_ablation.csv`.

        Returns:
            Dict[str, Dict[str, float]]: Performance comparison across modality ablation settings.
        """
        logger.info("Starting Modality Ablation Study...")
        if self.config.input_mode != "multimodal":
            logger.info(f"Skipping ablation for unimodal model (input_mode={self.config.input_mode}).")
            return {}

        settings = {"full": None, "wo_audio": "audio", "wo_sensor": "sensor", "wo_vision": "vision"}
        results: Dict[str, Dict[str, float]] = {}
        for name, mask in settings.items():
            sample_df = self.evaluate_sample_level(test_loader, mask_modality=mask)
            fruit_df = self.evaluate_fruit_level(sample_df)
            results[name] = {**self._level_metrics(sample_df, "sample"), **self._level_metrics(fruit_df, "fruit")}

        full = results["full"]
        for name, res in results.items():
            for level in ("sample", "fruit"):
                res[f"delta_{level}_macro_f1"] = full[f"{level}_macro_f1"] - res[f"{level}_macro_f1"]

        table = pd.DataFrame(results).T
        table.index.name = "setting"
        path = self.output_dir / f"{self._file_prefix()}ablation.csv"
        table.to_csv(path)
        logger.info(f"Saved ablation table: {path}")
        return results

    # ------------------------------------------------------------------
    # 3.6. Confusion matrix
    # ------------------------------------------------------------------
    def plot_confusion_matrix(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        title: str = "Teacher Confusion Matrix",
        save_name: str = "teacher_confusion_matrix.png",
        normalize: bool = True,
    ) -> str:
        """Plot and save a confusion matrix heatmap.

        - `normalize=True`: each cell displays row-normalized percentage (class Recall) and sample count.
        - `normalize=False`: displays raw sample counts.
        X-axis: predicted label, Y-axis: true label, ordered by `CLASS_ORDER`.

        Returns:
            str: Path to the saved PNG file.
        """
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int),
                              labels=list(range(NUM_CLASSES)))
        if normalize:
            row_sum = cm.sum(axis=1, keepdims=True)
            values = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum > 0)
            annot = np.array([[f"{values[i, j] * 100:.1f}%\n({cm[i, j]})" for j in range(NUM_CLASSES)]
                              for i in range(NUM_CLASSES)])
            fmt, vmax = "", 1.0
        else:
            values, annot, fmt, vmax = cm, True, "d", None

        # Use Figure + Agg canvas (PNG output only): avoids Tk dependency or altering global backend
        fig = Figure(figsize=(6, 5))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        sns.heatmap(values, annot=annot, fmt=fmt, cmap="Blues", vmin=0, vmax=vmax, cbar=normalize,
                    xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=ax, annot_kws={"size": 12})
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(title, fontweight="bold")
        fig.tight_layout()
        path = self.output_dir / save_name
        fig.savefig(path, dpi=150)
        return str(path)

    # ------------------------------------------------------------------
    # 3.7. Pipeline
    # ------------------------------------------------------------------
    def run_full_evaluation_pipeline(
        self,
        test_loader: Optional[DataLoader] = None,
        split_name: Optional[str] = None,
        load_checkpoint: bool = False,
    ) -> Dict[str, Any]:
        """End-to-end Teacher evaluation pipeline.

        Coordinates the full evaluation workflow:
        1. Load Teacher checkpoint (when `load_checkpoint=True` or model is None).
        2. Evaluate sample-level and fruit-level predictions (Full Multi-Modal).
        3. Compute comprehensive metrics (Sample-level & Fruit-level).
        4. Run Modality Ablation Study (Audio / Sensor / Vision) — multimodal only.
        5. Generate confusion matrix heatmaps.
        6. Save structured reports (JSON / CSV / log).

        Output files in `output_dir` (prefixed with `<split>_`):
        `sample_predictions.csv`, `fruit_predictions.csv`, `metrics.csv` (long format: model, split, level,
        metric, value), `metrics.json`, `metrics_by_<col>.csv` (e.g., by device/angle), `ablation.csv`,
        `cm_sample_counts.png`, `cm_sample_normalized.png`, `cm_fruit_counts.png`, `cm_fruit_normalized.png`, `eval.log`.

        Args:
            test_loader: DataLoader to evaluate (defaults to `self.test_loader`).
            split_name: Split name ("val", "test", ...) used as filename prefix (defaults to `config.split_name`).
            load_checkpoint: If True, load weights from `config.teacher_checkpoint_path` first.

        Returns:
            Dict[str, Any]: Dictionary summarizing all experimental results.
        """
        if split_name is not None:
            self.config.split_name = split_name
        prefix = self._file_prefix()
        log_handler = logging.FileHandler(self.output_dir / f"{prefix}eval.log", mode="w", encoding="utf-8")
        log_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(log_handler)
        try:
            logger.info("=== STARTING FULL TEACHER EVALUATION PIPELINE ===")
            logger.info(f"model={self.config.model_name} | split={self.config.split_name} | "
                        f"input_mode={self.config.input_mode} | strategy={self.config.fruit_aggregation_strategy}")
            if self.model is None or load_checkpoint:
                self.load_teacher_model()

            sample_df = self.evaluate_sample_level(test_loader)
            fruit_df = self.evaluate_fruit_level(sample_df)
            sample_df.to_csv(self.output_dir / f"{prefix}sample_predictions.csv", index=False)
            fruit_df.to_csv(self.output_dir / f"{prefix}fruit_predictions.csv", index=False)

            metrics = {**self._level_metrics(sample_df, "sample"), **self._level_metrics(fruit_df, "fruit")}
            long_rows = []
            for key, value in metrics.items():
                level, metric = key.split("_", 1)
                long_rows.append({"model": self.config.model_name, "split": self.config.split_name,
                                  "level": level, "metric": metric, "value": value})
            pd.DataFrame(long_rows).to_csv(self.output_dir / f"{prefix}metrics.csv", index=False)

            # Metrics broken down by auxiliary columns (e.g., recording device, camera angle)
            by_group: Dict[str, Dict[str, Dict[str, float]]] = {}
            for col in ("device", "camera_angle"):
                if col in sample_df and sample_df[col].astype(str).str.len().gt(0).any() \
                        and sample_df[col].nunique() > 1:
                    group_res = {str(g): self._level_metrics(gdf, "sample")
                                 for g, gdf in sample_df.groupby(col)}
                    by_group[col] = group_res
                    pd.DataFrame(group_res).T.rename_axis(col).to_csv(
                        self.output_dir / f"{prefix}metrics_by_{col}.csv")

            figures = {
                "cm_sample_counts": self.plot_confusion_matrix(
                    sample_df["y_true"], sample_df["y_pred"], f"{self.config.model_name} | {self.config.split_name} | sample",
                    f"{prefix}cm_sample_counts.png", normalize=False),
                "cm_sample_normalized": self.plot_confusion_matrix(
                    sample_df["y_true"], sample_df["y_pred"], f"{self.config.model_name} | {self.config.split_name} | sample",
                    f"{prefix}cm_sample_normalized.png", normalize=True),
                "cm_fruit_counts": self.plot_confusion_matrix(
                    fruit_df["y_true"], fruit_df["y_pred"], f"{self.config.model_name} | {self.config.split_name} | fruit",
                    f"{prefix}cm_fruit_counts.png", normalize=False),
                "cm_fruit_normalized": self.plot_confusion_matrix(
                    fruit_df["y_true"], fruit_df["y_pred"], f"{self.config.model_name} | {self.config.split_name} | fruit",
                    f"{prefix}cm_fruit_normalized.png", normalize=True),
            }

            ablation = self.evaluate_modality_ablation(test_loader)

            results = {
                "model": self.config.model_name,
                "split": self.config.split_name,
                "input_mode": self.config.input_mode,
                "fruit_aggregation_strategy": self.config.fruit_aggregation_strategy,
                "metrics": metrics,
                "metrics_by_group": by_group,
                "ablation": ablation,
                "figures": figures,
                "n_samples": int(len(sample_df)),
                "n_fruits": int(len(fruit_df)),
            }
            with open(self.output_dir / f"{prefix}metrics.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=float)
            logger.info(f"[{self.config.split_name}] sample Macro-F1 = {metrics['sample_macro_f1']:.4f} | "
                        f"fruit Macro-F1 = {metrics['fruit_macro_f1']:.4f}")
            logger.info("=== EVALUATION COMPLETE ===")
            return results
        finally:
            logger.removeHandler(log_handler)
            log_handler.close()


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    output_dir: Union[str, Path],
    input_mode: str = "audio",
    split_name: str = "test",
    device: Optional[str] = None,
    model_name: str = "model",
    fruit_aggregation_strategy: str = "mean_prob",
) -> Dict[str, Any]:
    """Convenience function: evaluate a model on a DataLoader and save all outputs to `output_dir`.

    Example (Week 2 audio training script):
        from eval_teacher import evaluate
        results = evaluate(model, val_loader, "results", input_mode="audio", split_name="val",
                           model_name="audio_resnet18")
    """
    config = TeacherEvaluatorConfig(
        output_dir=str(output_dir),
        input_mode=input_mode,
        split_name=split_name,
        model_name=model_name,
        fruit_aggregation_strategy=fruit_aggregation_strategy,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    evaluator = TeacherEvaluator(config, model=model, test_loader=dataloader)
    return evaluator.run_full_evaluation_pipeline()


# ------------------------------------------------------------------------------
# 4. XAI CLASS: GRAD-CAM VISUALIZATION ON TEACHER
# ------------------------------------------------------------------------------
class TeacherGradCAMVisualizer:
    """Grad-CAM activation heatmap extractor and visualizer for the Teacher Model.

    Scientific objective (Specification Section 1.4):
    - Guided by acoustic and physical sensor cues inside the Teacher network,
      which morphological regions on the durian shell do Vision spatial filters focus on?
    - Visual inspection: does the network attend to spine grooves (groove expansion) and
      pedicel disc, or get distracted by background and lighting variations?

    Class responsibilities:
    1. Register Forward & Backward Hooks on the Target Convolutional Layer.
    2. Run forward pass to store spatial feature maps.
    3. Run backward pass with target class gradients to compute importance weights $\alpha_k$.
    4. Compute weighted linear combination followed by ReLU:
       $$L_{Grad-CAM}^c = ReLU\\left(\\sum_k \\alpha_k^c A^k\\right)$$
    5. Resize heatmap to input image size, apply colormap, and overlay on the RGB image.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """Initialize the Grad-CAM Visualizer for Teacher.

        Args:
            model: Multi-Modal Teacher model.
            target_layer_name: Hierarchical module name of the target conv layer (e.g., 'vision_backbone.layer4').
            device: Compute device ('cuda' or 'cpu').
        """
        self.model = model
        self.target_layer_name = target_layer_name
        self.device = torch.device(device)

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hooks: List[Any] = []

        # Register hooks on target layer
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Find layer by name and register forward_hook and backward_hook.

        Implementation guide for Member 2 (TODO):
        ---------------------------------------------------------
        1. Look up `self.target_layer_name` in `dict(self.model.named_modules())`.
        2. Register Forward Hook to capture `self.activations`:
           `def forward_hook(module, input, output): self.activations = output.detach()`
        3. Register Backward Hook to capture `self.gradients`:
           `def backward_hook(module, grad_in, grad_out): self.gradients = grad_out[0].detach()`
        4. Store hook handles in `self._hooks` so they can be removed cleanly later.
        """
        # TODO [Member 2]: Implement forward and backward hook registration
        pass

    def generate_cam(
        self,
        image_tensor: torch.Tensor,
        audio_tensor: torch.Tensor,
        sensor_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate normalized Grad-CAM heatmap for a single multimodal sample.

        Implementation guide for Member 2 (TODO):
        ---------------------------------------------------------
        1. Ensure batched shapes: `(1, C, H, W)` for image, `(1, 1, n_mels, T)` for audio, `(1, D)` for sensor.
        2. Zero gradients: `self.model.zero_grad()`.
        3. Forward pass through Teacher: `logits, _ = self.model(image_tensor, audio_tensor, sensor_tensor)`.
        4. If `target_class is None`: pick top predicted class `target_class = logits.argmax(dim=-1).item()`.
        5. Compute target score: `score = logits[0, target_class]`.
        6. Backward pass to trigger backward hook: `score.backward()`.
        7. Compute weights $\\alpha_k = \\frac{1}{Z} \\sum_i \\sum_j \\frac{\\partial Y^c}{\\partial A_{i,j}^k}$
           (Global Average Pooling over gradients).
        8. Compute linear combination of feature maps: $CAM = \\sum_k \\alpha_k A^k$.
        9. Apply ReLU to keep features with positive contribution: $CAM = ReLU(CAM)$.
        10. Min-Max normalize CAM to $[0, 1]$.
        11. Resize CAM to original image spatial resolution `(H, W)`.

        Args:
            image_tensor: Input RGB image tensor.
            audio_tensor: Input knock Mel-Spectrogram tensor.
            sensor_tensor: Input integrated sensor feature vector tensor.
            target_class: Target class index (0: Unripe, 1: Ripe, 2: Overripe). Defaults to top prediction.

        Returns:
            np.ndarray: 2D array of shape (H, W) with values in [0, 1].
        """
        # TODO [Member 2]: Implement Grad-CAM algorithm here
        raise NotImplementedError(
            "TODO [Member 2]: Please implement generate_cam() following the docstring steps!"
        )

    def overlay_cam_on_image(
        self,
        rgb_image: np.ndarray,
        cam_mask: np.ndarray,
        alpha: float = 0.5,
        colormap: int = 2,  # cv2.COLORMAP_JET
    ) -> np.ndarray:
        """Overlay Grad-CAM heatmap onto the original RGB image for visual display.

        Implementation guide for Member 2 (TODO):
        ---------------------------------------------------------
        1. Ensure `rgb_image` is `uint8` in range [0, 255].
        2. Convert `cam_mask` (0..1) to uint8: `cam_uint8 = np.uint8(255 * cam_mask)`.
        3. Apply OpenCV colormap: `heatmap = cv2.applyColorMap(cam_uint8, colormap)`.
        4. Blend images: `overlay = cv2.addWeighted(rgb_image, 1 - alpha, heatmap, alpha, 0)`.
        5. Return overlaid RGB image ready for saving or plotting with matplotlib.

        Args:
            rgb_image: Original RGB image (H, W, 3) uint8.
            cam_mask: Grad-CAM heatmap mask (H, W) in [0, 1].
            alpha: Heatmap overlay transparency factor (0.0 to 1.0).
            colormap: OpenCV colormap code (default COLORMAP_JET).

        Returns:
            np.ndarray: Overlaid RGB image (H, W, 3).
        """
        # TODO [Member 2]: Implement overlay_cam_on_image()
        raise NotImplementedError(
            "TODO [Member 2]: Please implement overlay_cam_on_image()!"
        )

    def export_gradcam_gallery(
        self,
        samples_list: List[Dict[str, Any]],
        save_dir: str,
        num_samples_per_class: int = 5,
    ) -> None:
        """Export a gallery of Grad-CAM visualizations for representative fruits (Unripe, Ripe, Overripe).

        SUGGESTED GRAD-CAM EVALUATION METRICS (Reference for Member 2):
        ---------------------------------------------------------
        1. Energy-based Pointing Game: Percentage of activation energy falling inside the
           durian fruit mask (filtering out spurious background activations).
        2. Attention Dispersion / Concentration: Spatial concentration of the attention map
           (sharp focused regions vs. diffused attention).
        3. Morphological Alignment Score: Correlation between high-attention regions and spine
           grooves (using edge/ridge detectors such as Sobel or Frangi filters).

        >>> SPECIAL NOTE FOR MEMBER 2:
        >>> These metrics are suggestions; please explore additional quantitative XAI metrics
        >>> to evaluate Grad-CAM faithfulness and interpretability scientifically.

        Args:
            samples_list: List of test sample dicts containing file paths and labels.
            save_dir: Output directory for saving visualization images.
            num_samples_per_class: Number of samples exported per ripeness class.
        """
        # TODO [Member 2]: Iterate over representative samples, call generate_cam + overlay, and save comparisons
        # [Original Image | Grad-CAM Heatmap | Overlay Heatmap]
        pass


# ------------------------------------------------------------------------------
# 5. ERROR ANALYSIS AND PHYSICAL LIMITATION ANALYSIS CLASS
# ------------------------------------------------------------------------------
class TeacherErrorAnalyzer:
    """In-depth error analyzer for Teacher model misclassifications.

    Reference Section 1.4 (Contribution 4):
    Analyze failure cases to uncover biochemical / physical causes:
    - Why is a ripe fruit misclassified as unripe? (e.g., unusually thick shell,
      target TSS achieved but internal cavity resonance not yet developed so knock stays dull).
    - Correlate errors with sensor readings
      (Firmness, TSS, Dry Matter, Weight, thermal contrast delta_temp_fg_bg).
    """

    def __init__(self, output_dir: str) -> None:
        """Initialize error analyzer.

        Args:
            output_dir: Directory for saving error analysis plots and reports.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_failure_cases(self, fruit_level_df: pd.DataFrame) -> pd.DataFrame:
        """Filter and categorize all misclassified fruits at the fruit level.

        Implementation guide for Member 2 (TODO):
        ---------------------------------------------------------
        1. Filter rows where `is_correct == False` (or `true_label != pred_label`).
        2. Tag detailed Error Subtype:
           - 'Unripe_as_Ripe': Unripe fruit predicted as Ripe (Critical error: selling immature fruit).
           - 'Ripe_as_Unripe': Ripe fruit predicted as Unripe (Missed optimal harvest window).
           - 'Ripe_as_Overripe': Ripe fruit predicted as Overripe.
           - 'Overripe_as_Ripe': Overripe/fermenting fruit predicted as Ripe.
        3. Return DataFrame of failure cases with model confidence probabilities.
        """
        # TODO [Member 2]: Implement failure extraction logic here
        raise NotImplementedError(
            "TODO [Member 2]: Please implement extract_failure_cases()!"
        )

    def correlate_errors_with_physical_properties(
        self,
        failures_df: pd.DataFrame,
        manifest_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Analyze correlations between failure cases and biochemical / sensor properties.

        Implementation guide for Member 2 (TODO):
        ---------------------------------------------------------
        1. Merge `failures_df` with `manifest_df` on `fruit_id`.
        2. Extract attributes: `Firmness_AVE`, `TSS_AVE`, `Dry_Matter_AVE`,
           `Harvest_Weight`, `fg_mean_temp`, `delta_temp_fg_bg`.
        3. Compare distributions of Failures vs. Successes via Boxplot or Violin plot.
        4. Test physical hypothesis: do failure cases cluster near boundary thresholds
           between ripeness stages?
        """
        # TODO [Member 2]: Implement error correlation analysis here
        raise NotImplementedError(
            "TODO [Member 2]: Please implement correlate_errors_with_physical_properties()!"
        )


# ------------------------------------------------------------------------------
# 6. MAIN / CLI EXECUTION
# ------------------------------------------------------------------------------
def parse_args():
    """Parse command-line arguments when executing the script directly."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate Multi-Modal Teacher & Visualize Grad-CAM"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/teacher_best.pth",
        help="Path to Teacher model weights file",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/final/final_multimodal_manifest.csv",
        help="Path to Master Manifest CSV",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="data/final/test",
        help="Path to Test split directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="workspaces/member2_audio/03_test_eval_teacher_and_gradcam/results",
        help="Directory to save evaluation reports and figures",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Evaluation DataLoader batch size",
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        default="multimodal",
        choices=["multimodal", "audio", "vision", "sensor"],
        help="Model input mode (multimodal for Teacher, audio/vision/sensor for unimodal models)",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="test",
        help="Evaluated split name (val/test), used as output filename prefix",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="teacher",
        help="Model name recorded in metrics tables",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda or cpu)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 80)
    print(" MULTI-MODAL TEACHER EVALUATION & GRAD-CAM VISUALIZATION (XAI)")
    print(" Responsible: Member 2 (Audio & Multi-Modal Evaluation)")
    print("=" * 80)

    config = TeacherEvaluatorConfig(
        teacher_checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        test_data_dir=args.test_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        input_mode=args.input_mode,
        split_name=args.split_name,
        model_name=args.model_name,
    )

    evaluator = TeacherEvaluator(config=config)
    print(f"\n[INFO] Initialized Teacher evaluation configuration.")
    print(f"[INFO] Target checkpoint: {config.teacher_checkpoint_path}")
    print(f"[INFO] Results will be saved to: {config.output_dir}")
    print("\n[USAGE GUIDE]:")
    print("1. Evaluation pipeline (TeacherEvaluator) is ready; Grad-CAM and error analysis are Week 4 items.")
    print("2. For Teacher architecture: instantiate model then run in Python:")
    print("     from eval_teacher import TeacherEvaluator, TeacherEvaluatorConfig")
    print("     ev = TeacherEvaluator(TeacherEvaluatorConfig(...), model=model, test_loader=loader)")
    print("     ev.run_full_evaluation_pipeline(load_checkpoint=True)")
    print("3. Unimodal models: from eval_teacher import evaluate; evaluate(model, loader, out_dir, input_mode='audio')")

