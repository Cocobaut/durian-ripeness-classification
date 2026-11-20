"""Loss functions for classification and cross-modal knowledge distillation."""

from src.losses.classification_loss import (
    DEFAULT_ORDINAL_COST_MATRIX,
    CostSensitiveFocalLoss,
    DurianClassificationLoss,
    compute_class_weights,
)
from src.losses.distillation_loss import CrossModalDistillationLoss

__all__ = [
    "CostSensitiveFocalLoss",
    "CrossModalDistillationLoss",
    "DEFAULT_ORDINAL_COST_MATRIX",
    "DurianClassificationLoss",
    "compute_class_weights",
]
