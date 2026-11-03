"""Mô hình MLP Sensor Backbone phục vụ huấn luyện đơn phương thức & Multimodal Fusion.

Phụ trách: Member 3 (Nhánh Cảm biến / Bảng số liệu)
Dự án: Phân loại độ chín sầu riêng bằng chưng cất tri thức đa phương thức (CMKD)
"""

from typing import Sequence, Tuple, Union
import torch
import torch.nn as nn


class SensorBackbone(nn.Module):
    """Mạng MLP trích xuất vector đặc trưng cảm biến f_sensor in R^{latent_dim}.
    
    Kiến trúc gồm các khối Linear -> LayerNorm -> ReLU -> Dropout,
    cuối cùng chiếu về latent_dim (mặc định 32) và phân loại ra 3 lớp độ chín.
    Sử dụng LayerNorm để hoạt động ổn định trên mọi batch size (kể cả batch_size=1).
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: Sequence[int] = (64, 48),
        latent_dim: int = 32,
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.extend([
                nn.Linear(in_d, h_d),
                nn.LayerNorm(h_d),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_d = h_d
            
        layers.append(nn.Linear(in_d, latent_dim))
        layers.append(nn.LayerNorm(latent_dim))
        layers.append(nn.ReLU(inplace=True))
        
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Lan truyền xuôi: trả về logits hoặc (logits, f_sensor)."""
        feat = self.encoder(x)
        logits = self.classifier(feat)
        return (logits, feat) if return_latent else logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Trích xuất vector đặc trưng f_sensor in R^{latent_dim} cho Fusion Module."""
        return self.encoder(x)
