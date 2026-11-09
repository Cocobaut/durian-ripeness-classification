"""Hàm mất mát Chưng cất tri thức Liên phương thức (Cross-Modal Knowledge Distillation Loss).

Dự án: Phân loại độ chín sầu riêng bằng Chưng cất tri thức Đa phương thức (CMKD)
Tham chiếu đặc tả: Mục 1.3.3 (Giai đoạn 2 - Vision-Only Student).

Công thức tổng hợp 3 thành phần (tích hợp Confidence & Modality-Weighted Distillation):
    L_total = alpha * L_CE(y, z_student)
            + beta  * mean_i( w_i * L_KD_i(z_student_i, z_teacher_i; T) )
            + gamma * mean_i( w_i * L_feat_i(phi(f_img_student_i), f_mm_teacher_i) )

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
4. Trọng số mẫu thông minh `w_i = w_modality_i * w_confidence_i`:
   - `w_modality_i`: Giữ nguyên 1.0 khi mẫu huấn luyện có đủ cả Audio + Sensor; giảm xuống
     `missing_modality_scale` (mặc định 0.45) khi mẫu bị khuyết Audio hoặc Sensor.
   - `w_confidence_i`: Giữ mức cao `[0.7 + 0.3 * p_t(y)]` khi Teacher dự đoán đúng nhãn thật `y`;
     giảm xuống `wrong_teacher_scale` (mặc định 0.30) khi Teacher dự đoán sai nhãn thật để tránh
     ép Student học theo sai số của Teacher.
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
        temperature: float = 2.5,
        feat_loss_type: str = "cosine",
        ce_loss_type: str = "cost_sensitive_focal",
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
        focal_gamma: float = 2.0,
        cost_matrix: Optional[Union[Sequence[Sequence[float]], torch.Tensor]] = None,
        cost_weight: float = 0.2,
        confidence_weighting: bool = True,
        missing_modality_scale: float = 0.45,
        wrong_teacher_scale: float = 0.30,
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
        self.confidence_weighting = bool(confidence_weighting)
        self.missing_modality_scale = float(missing_modality_scale)
        self.wrong_teacher_scale = float(wrong_teacher_scale)

        self.ce_criterion = DurianClassificationLoss(
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
            loss_type=ce_loss_type,
            cost_matrix=cost_matrix,
            cost_weight=cost_weight,
        )

    @torch.no_grad()
    def compute_distill_sample_weights(
        self,
        logits_teacher: torch.Tensor,
        targets: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        sensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Tính trọng số chưng cất w_i in (0, 1] cho từng mẫu theo độ đủ phương thức & độ tin cậy Teacher."""
        bsz = logits_teacher.size(0)
        device = logits_teacher.device
        weights = torch.ones(bsz, dtype=torch.float32, device=device)
        if not self.confidence_weighting:
            return weights

        # 1. Trọng số theo mức độ đầy đủ phương thức của Teacher (Audio + Sensor)
        if audio is not None:
            has_audio = audio.detach().abs().flatten(1).amax(dim=1) > 1e-6
            weights = torch.where(has_audio, weights, weights * self.missing_modality_scale)
        if sensor is not None:
            has_sensor = sensor.detach().abs().flatten(1).amax(dim=1) > 1e-6
            weights = torch.where(has_sensor, weights, weights * self.missing_modality_scale)

        # 2. Trọng số theo độ chính xác và độ tự tin của Teacher trên nhãn thật
        probs_t = F.softmax(logits_teacher.detach().float(), dim=-1)
        idx = targets.long().clamp(0, probs_t.size(-1) - 1)
        true_class_prob = probs_t.gather(1, idx.unsqueeze(1)).squeeze(1)
        teacher_correct = probs_t.argmax(dim=-1) == idx

        conf_if_correct = 0.70 + 0.30 * true_class_prob
        conf_if_wrong = self.wrong_teacher_scale * (0.50 + 0.50 * true_class_prob)
        conf_factor = torch.where(teacher_correct, conf_if_correct, conf_if_wrong)

        return (weights * conf_factor).clamp(min=0.05, max=1.0)

    def compute_kd_loss(
        self,
        logits_student: torch.Tensor,
        logits_teacher: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Tính L_KD = T^2 * KL(softmax(z_t / T) || softmax(z_s / T)) có hỗ trợ trọng số mẫu."""
        t = self.temperature
        log_probs_s = F.log_softmax(logits_student.float() / t, dim=-1)
        probs_t = F.softmax(logits_teacher.detach().float() / t, dim=-1)
        kl_per_sample = F.kl_div(log_probs_s, probs_t, reduction="none").sum(dim=-1) * (t * t)
        if sample_weights is not None:
            kl_per_sample = kl_per_sample * sample_weights.float()
        return kl_per_sample.mean()

    def compute_feat_loss(
        self,
        f_proj_student: torch.Tensor,
        f_mm_teacher: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Tính L_feat giữa phi(f_img_student) và f_mm_teacher có hỗ trợ trọng số mẫu."""
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
        w = sample_weights.float()[valid_mask] if sample_weights is not None else None

        if self.feat_loss_type == "cosine":
            dist_per_sample = 1.0 - F.cosine_similarity(s, t, dim=-1, eps=1e-8)
        elif self.feat_loss_type == "mse":
            s_norm = F.normalize(s, p=2, dim=-1, eps=1e-8)
            t_norm = F.normalize(t, p=2, dim=-1, eps=1e-8)
            dist_per_sample = (s_norm - t_norm).pow(2).mean(dim=-1)
        else:
            cos_dist = 1.0 - F.cosine_similarity(s, t, dim=-1, eps=1e-8)
            s_norm = F.normalize(s, p=2, dim=-1, eps=1e-8)
            t_norm = F.normalize(t, p=2, dim=-1, eps=1e-8)
            mse_dist = (s_norm - t_norm).pow(2).mean(dim=-1)
            dist_per_sample = 0.5 * cos_dist + 0.5 * mse_dist

        if w is not None:
            dist_per_sample = dist_per_sample * w
        return dist_per_sample.mean()

    def forward(
        self,
        logits_student: torch.Tensor,
        logits_teacher: torch.Tensor,
        f_proj_student: torch.Tensor,
        f_mm_teacher: torch.Tensor,
        targets: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        sensor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Tính tổng hợp mất mát L_total và trả về kèm dict thành phần để log."""
        logits_s = logits_student.float()
        sample_w = self.compute_distill_sample_weights(
            logits_teacher=logits_teacher,
            targets=targets,
            audio=audio,
            sensor=sensor,
        )

        loss_ce = self.ce_criterion(logits_s, targets) if self.alpha > 0 else logits_s.sum() * 0.0
        loss_kd = (
            self.compute_kd_loss(logits_s, logits_teacher, sample_weights=sample_w)
            if self.beta > 0
            else logits_s.sum() * 0.0
        )
        loss_feat = (
            self.compute_feat_loss(f_proj_student, f_mm_teacher, sample_weights=sample_w)
            if self.gamma > 0
            else logits_s.sum() * 0.0
        )

        loss_total = self.alpha * loss_ce + self.beta * loss_kd + self.gamma * loss_feat
        details = {
            "loss_total": float(loss_total.detach().item()),
            "loss_ce": float(loss_ce.detach().item()),
            "loss_kd": float(loss_kd.detach().item()),
            "loss_feat": float(loss_feat.detach().item()),
            "distill_weight_mean": float(sample_w.detach().mean().item()),
        }
        return loss_total, details
