"""Hàm mất mát phân loại dùng cho Unimodal Baselines và Multi-Modal Teacher.

Dự án: Phân loại độ chín sầu riêng bằng Chưng cất tri thức Đa phương thức (CMKD)
Tham chiếu đặc tả: Mục 1.3.2 (Teacher học có giám sát từ nhãn thật).

Teacher dùng Class-Weighted Cross-Entropy + Label Smoothing:
- Class weights w_c = N / (C * N_c) bù lệch số mẫu giữa các lớp (tính trên tập Train).
- Label smoothing giữ phân phối xác suất "mềm", tránh Teacher quá tự tin
  -> soft targets tốt hơn cho bước chưng cất sang Student (distillation_loss.py, Giai đoạn 2).
- Tùy chọn Focal Loss (focal_gamma > 0) cho thực nghiệm tập trung vào mẫu khó.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(
    labels: Union[Sequence[int], np.ndarray, torch.Tensor],
    num_classes: int = 3,
) -> torch.Tensor:
    """Trọng số nghịch đảo tần suất: w_c = N / (C * N_c). Lớp vắng mặt được gán N_c = 1."""
    y = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    counts = torch.bincount(y, minlength=num_classes).float()
    return counts.sum() / (num_classes * counts.clamp(min=1.0))


class DurianClassificationLoss(nn.Module):
    """Cross-Entropy có trọng số lớp + Label Smoothing (+ Focal tùy chọn).

    Args:
        class_weights: Tensor [C] trọng số lớp (None = không đặt trọng số).
        label_smoothing: Hệ số làm mềm nhãn epsilon (khuyến nghị 0.05 - 0.1 cho Teacher).
        focal_gamma: > 0 để bật Focal Loss; 0 = Cross-Entropy thường (mặc định).
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
        focal_gamma: float = 0.0,
    ) -> None:
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None
        self.label_smoothing = float(label_smoothing)
        self.focal_gamma = float(focal_gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        if self.focal_gamma <= 0:
            return F.cross_entropy(logits, targets, weight=self.class_weights,
                                   label_smoothing=self.label_smoothing)

        # Focal Loss với nhãn đã làm mềm: FL = -sum_c w_c * (1 - p_c)^gamma * q_c * log p_c
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

