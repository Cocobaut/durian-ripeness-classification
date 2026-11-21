"""Classification loss functions for Unimodal Baselines and Multi-Modal Teacher.

Project: Durian Ripeness Classification via Cross-Modal Knowledge Distillation (CMKD)
Specification reference: Section 1.3.2 (Teacher supervised learning from ground-truth labels).

Main loss mechanisms:
1. Class-Weighted Cross-Entropy + Label Smoothing.
2. Focal Loss (focal_gamma > 0) for focusing on hard boundary samples.
3. Cost-Sensitive Focal Loss (Ordinal Penalized):
   - Penalizes predictions using an ordinal cost matrix: Unripe (0) -> Ripe (1) -> Overripe (2).
   - Strongly discourages severe mistakes (confusing Unripe <-> Overripe).
   - Increases penalty for missing Overripe and predicting Ripe (key boundary challenge).
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Default ordinal cost matrix for 3 classes: [0: Unripe, 1: Ripe, 2: Overripe]
# C[y, c] is the cost when true label is y and predicted probability falls on class c.
DEFAULT_ORDINAL_COST_MATRIX = [
    [0.0, 1.0, 3.0],  # True Unripe: pred Ripe costs 1.0, pred Overripe costs 3.0 (severe)
    [1.0, 0.0, 1.5],  # True Ripe: pred Unripe costs 1.0, pred Overripe costs 1.5
    [3.0, 2.0, 0.0],  # True Overripe: pred Unripe costs 3.0, pred Ripe costs 2.0 (prevents missing Overripe)
]


def compute_class_weights(
    labels: Union[Sequence[int], np.ndarray, torch.Tensor],
    num_classes: int = 3,
) -> torch.Tensor:
    """Inverse frequency weights: w_c = N / (C * N_c). Absent classes get N_c = 1."""
    y = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    counts = torch.bincount(y, minlength=num_classes).float()
    return counts.sum() / (num_classes * counts.clamp(min=1.0))


class CostSensitiveFocalLoss(nn.Module):
    """Cost-Sensitive Focal Loss combined with an Ordinal Cost Matrix.

    Formula:
        L_total = L_focal + cost_weight * (1 - p_y)^gamma * sum_c (C[y, c] * p_c)

    Benefits:
        - Focal weight (1 - p_y)^gamma concentrates gradients on ambiguous boundary fruits.
        - Cost matrix C pushes probability mass away from dangerous or missed classes.
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        cost_matrix: Optional[Union[Sequence[Sequence[float]], torch.Tensor]] = None,
        cost_weight: float = 0.5,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

        if cost_matrix is None:
            cost_matrix = DEFAULT_ORDINAL_COST_MATRIX
        self.register_buffer("cost_matrix", torch.as_tensor(cost_matrix, dtype=torch.float32))

        self.cost_weight = float(cost_weight)
        self.focal_gamma = float(focal_gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        # 1. Focal loss with label smoothing
        q = F.one_hot(targets, num_classes).float()
        if self.label_smoothing > 0:
            q = q * (1.0 - self.label_smoothing) + self.label_smoothing / num_classes

        # Per-sample focal cross-entropy
        focal_ce = -(q * (1.0 - probs).pow(self.focal_gamma) * log_probs).sum(dim=-1)  # [B]

        # 2. Ordinal cost-sensitive penalty term
        # pt: probability of the target class
        p_target = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]
        focal_modulator = (1.0 - p_target).pow(self.focal_gamma)     # [B]

        # sample_cost_vec: cost vector C[target, :] per sample
        sample_cost_vec = self.cost_matrix[targets]                  # [B, C]
        expected_cost = (sample_cost_vec * probs).sum(dim=-1)        # [B]
        cost_penalty = focal_modulator * expected_cost               # [B]

        total_sample_loss = focal_ce + self.cost_weight * cost_penalty  # [B]

        # 3. Class-balancing weights
        if self.class_weights is not None:
            w = self.class_weights[targets]
            norm = w.sum().clamp(min=1e-12)
            return (total_sample_loss * w).sum() / norm

        return total_sample_loss.mean()


class DurianClassificationLoss(nn.Module):
    """Unified classification loss for the Durian Ripeness project.

    Supports 3 modes:
    - 'cost_sensitive_focal' (recommended): Focal Loss + Ordinal Cost Matrix.
    - 'focal': Standard Focal Loss + Label Smoothing.
    - 'ce': Standard Class-Weighted Cross-Entropy + Label Smoothing.
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
        focal_gamma: float = 2.0,
        loss_type: str = "cost_sensitive_focal",
        cost_matrix: Optional[Union[Sequence[Sequence[float]], torch.Tensor]] = None,
        cost_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.label_smoothing = float(label_smoothing)
        self.focal_gamma = float(focal_gamma)
        self.cost_weight = float(cost_weight)

        if class_weights is not None:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

        if self.loss_type == "cost_sensitive_focal":
            self.cs_loss = CostSensitiveFocalLoss(
                class_weights=class_weights,
                cost_matrix=cost_matrix,
                cost_weight=cost_weight,
                focal_gamma=focal_gamma,
                label_smoothing=label_smoothing,
            )
        else:
            self.cs_loss = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()

        if self.loss_type == "cost_sensitive_focal" and self.cs_loss is not None:
            return self.cs_loss(logits, targets)

        if self.loss_type == "ce" or self.focal_gamma <= 0:
            return F.cross_entropy(
                logits,
                targets,
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
            )

        # Standard focal loss mode
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        q = F.one_hot(targets, num_classes).float()
        if self.label_smoothing > 0:
            q = q * (1.0 - self.label_smoothing) + self.label_smoothing / num_classes
        loss = -(q * (1.0 - probs).pow(self.focal_gamma) * log_probs)
        if self.class_weights is not None:
            loss = loss * self.class_weights.unsqueeze(0)
            norm = self.class_weights[targets].sum().clamp(min=1e-12)
            return loss.sum() / norm
        return loss.sum(dim=-1).mean()
