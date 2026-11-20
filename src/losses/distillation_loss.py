"""Cross-Modal Knowledge Distillation Loss.

Project: Durian Ripeness Classification via Cross-Modal Knowledge Distillation (CMKD)
Specification reference: Section 1.3.3 (Stage 2 - Vision-Only Student).

Composite 3-term formula:
    L_total = alpha * L_CE(y, z_student)
            + beta  * L_KD(z_student, z_teacher; T)
            + gamma * L_feat(phi(f_img_student), f_mm_teacher)

Where:
1. L_CE (Hard-label Loss):
   Uses `DurianClassificationLoss` (defaults to Cost-Sensitive Focal Loss with an ordinal
   cost matrix or class-weighted Cross-Entropy).
2. L_KD (Response-based Logit Matching):
   T^2 * KL_Div(softmax(z_student / T) || softmax(z_teacher / T)).
   Transfers "dark knowledge" about transition probabilities between ripeness stages.
3. L_feat (Feature-based Latent Alignment):
   Aligns the projected image embedding `f_proj = phi(f_img_student)` [B, 256]
   with the Teacher's fused multimodal embedding `f_mm` [B, 256] using Cosine distance
   (`1 - cos_sim`) or MSE on L2-normalized vectors.
   `relational` instead matches the in-batch cosine-similarity matrices (Relational KD, Park et al. 2019):
   the Student only has to reproduce *how fruits relate to each other* in the Teacher space, not the
   absolute Teacher embedding (which contains audio/sensor information an RGB image cannot recover).

KD weighting (`kd_weighting`):
- "none": plain batch-mean KL.
- "teacher_correct": distill only on samples the Teacher classifies correctly (do not copy Teacher errors).
- "teacher_confidence": weight each sample by the Teacher's probability of the true class.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.classification_loss import DurianClassificationLoss

FEAT_LOSS_TYPES = ("cosine", "mse", "cosine_mse", "relational", "cosine_relational")
KD_WEIGHTINGS = ("none", "teacher_correct", "teacher_confidence")


class CrossModalDistillationLoss(nn.Module):
    """3-term composite loss for training the Vision-Only Student."""

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 2.0,
        gamma: float = 1.0,
        temperature: float = 3.0,
        feat_loss_type: str = "cosine",
        ce_loss_type: str = "cost_sensitive_focal",
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
        focal_gamma: float = 2.0,
        cost_matrix: Optional[Union[Sequence[Sequence[float]], torch.Tensor]] = None,
        cost_weight: float = 0.5,
        kd_weighting: str = "none",
    ) -> None:
        super().__init__()
        if feat_loss_type not in FEAT_LOSS_TYPES:
            raise ValueError(f"feat_loss_type must be in {FEAT_LOSS_TYPES}, got {feat_loss_type!r}")
        if kd_weighting not in KD_WEIGHTINGS:
            raise ValueError(f"kd_weighting must be in {KD_WEIGHTINGS}, got {kd_weighting!r}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.kd_weighting = kd_weighting

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.temperature = float(temperature)
        self.feat_loss_type = feat_loss_type

        self.ce_criterion = DurianClassificationLoss(
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
            loss_type=ce_loss_type,
            cost_matrix=cost_matrix,
            cost_weight=cost_weight,
        )

    def kd_sample_weights(self, logits_teacher: torch.Tensor, targets: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Per-sample KD weights according to `self.kd_weighting` (None = uniform)."""
        if self.kd_weighting == "none" or targets is None:
            return None
        lt = logits_teacher.detach().float()
        if self.kd_weighting == "teacher_correct":
            return (lt.argmax(dim=-1) == targets).float()
        # teacher_confidence: Teacher probability (T = 1) assigned to the ground-truth class
        return F.softmax(lt, dim=-1).gather(1, targets.view(-1, 1)).squeeze(1)

    def compute_kd_loss(
        self,
        logits_student: torch.Tensor,
        logits_teacher: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute L_KD = T^2 * KL(softmax(z_t / T) || softmax(z_s / T)), optionally sample-weighted."""
        t = self.temperature
        log_probs_s = F.log_softmax(logits_student.float() / t, dim=-1)
        probs_t = F.softmax(logits_teacher.detach().float() / t, dim=-1)
        weights = self.kd_sample_weights(logits_teacher, targets)
        if weights is None:
            return F.kl_div(log_probs_s, probs_t, reduction="batchmean") * (t * t)
        per_sample = F.kl_div(log_probs_s, probs_t, reduction="none").sum(dim=-1)
        if weights.sum() <= 0:
            return log_probs_s.sum() * 0.0
        return (per_sample * weights).sum() / weights.sum() * (t * t)

    @staticmethod
    def _relational_loss(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """MSE between in-batch cosine-similarity matrices (off-diagonal entries only)."""
        n = s.size(0)
        if n < 2:
            return s.sum() * 0.0
        s_n = F.normalize(s, p=2, dim=-1, eps=1e-8)
        t_n = F.normalize(t, p=2, dim=-1, eps=1e-8)
        g_s = s_n @ s_n.t()
        g_t = t_n @ t_n.t()
        off_diag = ~torch.eye(n, dtype=torch.bool, device=s.device)
        return F.mse_loss(g_s[off_diag], g_t[off_diag])

    def compute_feat_loss(
        self,
        f_proj_student: torch.Tensor,
        f_mm_teacher: torch.Tensor,
    ) -> torch.Tensor:
        """Compute L_feat between phi(f_img_student) and f_mm_teacher."""
        s = f_proj_student.float()
        t = f_mm_teacher.detach().float()
        if s.shape != t.shape:
            raise ValueError(
                f"f_proj_student shape {tuple(s.shape)} does not match f_mm_teacher shape {tuple(t.shape)}."
            )

        # Compute only over samples where f_mm_teacher is non-zero
        valid_mask = t.abs().amax(dim=-1) > 1e-8
        if not valid_mask.any():
            return s.sum() * 0.0
        s = s[valid_mask]
        t = t[valid_mask]

        if self.feat_loss_type == "cosine":
            cos_sim = F.cosine_similarity(s, t, dim=-1, eps=1e-8)
            return (1.0 - cos_sim).mean()
        if self.feat_loss_type == "mse":
            s_norm = F.normalize(s, p=2, dim=-1, eps=1e-8)
            t_norm = F.normalize(t, p=2, dim=-1, eps=1e-8)
            return F.mse_loss(s_norm, t_norm)
        if self.feat_loss_type == "relational":
            return self._relational_loss(s, t)
        if self.feat_loss_type == "cosine_relational":
            cos_sim = F.cosine_similarity(s, t, dim=-1, eps=1e-8)
            return 0.5 * (1.0 - cos_sim).mean() + 0.5 * self._relational_loss(s, t)
        # cosine_mse
        cos_sim = F.cosine_similarity(s, t, dim=-1, eps=1e-8)
        s_norm = F.normalize(s, p=2, dim=-1, eps=1e-8)
        t_norm = F.normalize(t, p=2, dim=-1, eps=1e-8)
        return 0.5 * (1.0 - cos_sim).mean() + 0.5 * F.mse_loss(s_norm, t_norm)

    def forward(
        self,
        logits_student: torch.Tensor,
        logits_teacher: torch.Tensor,
        f_proj_student: torch.Tensor,
        f_mm_teacher: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute composite L_total and return alongside component dictionary for logging."""
        logits_s = logits_student.float()
        loss_ce = self.ce_criterion(logits_s, targets) if self.alpha > 0 else logits_s.sum() * 0.0
        loss_kd = self.compute_kd_loss(logits_s, logits_teacher, targets) if self.beta > 0 else logits_s.sum() * 0.0
        loss_feat = self.compute_feat_loss(f_proj_student, f_mm_teacher) if self.gamma > 0 else logits_s.sum() * 0.0

        loss_total = self.alpha * loss_ce + self.beta * loss_kd + self.gamma * loss_feat
        details = {
            "loss_total": float(loss_total.detach().item()),
            "loss_ce": float(loss_ce.detach().item()),
            "loss_kd": float(loss_kd.detach().item()),
            "loss_feat": float(loss_feat.detach().item()),
        }
        return loss_total, details

