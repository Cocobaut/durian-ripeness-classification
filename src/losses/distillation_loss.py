"""Hàm mất mát Chưng cất tri thức Liên phương thức (Cross-Modal Knowledge Distillation Loss).

Dự án: Phân loại độ chín sầu riêng bằng Chưng cất tri thức Đa phương thức (CMKD)
Tham chiếu đặc tả: Mục 1.3.3 (Giai đoạn 2 - Vision-Only Student).

Công thức tổng hợp 3 thành phần:
    L_total = alpha * L_CE(y, z_student)
            + beta  * L_KD(z_student, z_teacher; T)
            + gamma * L_feat(phi(f_img_student), f_mm_teacher)

Trong đó:
1. L_CE (Hard-label Loss):
   Sử dụng `DurianClassificationLoss` (mặc định là Cost-Sensitive Focal Loss kết hợp ma trận
   phạt chi phí thứ tự Ordinal hoặc Cross-Entropy có trọng số lớp).
2. L_KD (Response-based Logit Matching):
   T^2 * KL_Div(softmax(z_student / T) || softmax(z_teacher / T)).
   Truyền đạt "tri thức tối" (Dark Knowledge) về xác suất chuyển tiếp giữa các mức độ chín.
3. L_feat (Feature-based Latent Alignment):
   Căn chỉnh vector đặc trưng ảnh đã chiếu `f_proj = phi(f_img_student)` [B, 256]
   với vector đa phương thức hợp nhất `f_mm` [B, 256] của Teacher bằng khoảng cách Cosine
   (`1 - cos_sim`) hoặc MSE trên vector chuẩn hóa L2.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.classification_loss import DurianClassificationLoss


class CrossModalDistillationLoss(nn.Module):
    """Hàm mất mát tổng hợp 3 thành phần cho huấn luyện Vision-Only Student."""

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
    ) -> None:
        super().__init__()
        if feat_loss_type not in ("cosine", "mse", "cosine_mse"):
            raise ValueError(f"feat_loss_type phải thuộc ('cosine', 'mse', 'cosine_mse'), nhận {feat_loss_type!r}")
        if temperature <= 0:
            raise ValueError(f"temperature phải > 0, nhận {temperature}")

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

    def compute_kd_loss(
        self,
        logits_student: torch.Tensor,
        logits_teacher: torch.Tensor,
    ) -> torch.Tensor:
        """Tính L_KD = T^2 * KL(softmax(z_t / T) || softmax(z_s / T))."""
        t = self.temperature
        log_probs_s = F.log_softmax(logits_student.float() / t, dim=-1)
        probs_t = F.softmax(logits_teacher.detach().float() / t, dim=-1)
        return F.kl_div(log_probs_s, probs_t, reduction="batchmean") * (t * t)

    def compute_feat_loss(
        self,
        f_proj_student: torch.Tensor,
        f_mm_teacher: torch.Tensor,
    ) -> torch.Tensor:
        """Tính L_feat giữa phi(f_img_student) và f_mm_teacher."""
        s = f_proj_student.float()
        t = f_mm_teacher.detach().float()
        if s.shape != t.shape:
            raise ValueError(
                f"Kích thước f_proj_student {tuple(s.shape)} không khớp f_mm_teacher {tuple(t.shape)}."
            )

        # Chỉ tính trên các mẫu mà f_mm_teacher khác vector 0
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
        """Tính tổng hợp mất mát L_total và trả về kèm dict thành phần để log."""
        logits_s = logits_student.float()
        loss_ce = self.ce_criterion(logits_s, targets) if self.alpha > 0 else logits_s.sum() * 0.0
        loss_kd = self.compute_kd_loss(logits_s, logits_teacher) if self.beta > 0 else logits_s.sum() * 0.0
        loss_feat = self.compute_feat_loss(f_proj_student, f_mm_teacher) if self.gamma > 0 else logits_s.sum() * 0.0

        loss_total = self.alpha * loss_ce + self.beta * loss_kd + self.gamma * loss_feat
        details = {
            "loss_total": float(loss_total.detach().item()),
            "loss_ce": float(loss_ce.detach().item()),
            "loss_kd": float(loss_kd.detach().item()),
            "loss_feat": float(loss_feat.detach().item()),
        }
        return loss_total, details

