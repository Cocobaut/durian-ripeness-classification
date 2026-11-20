"""MLP Sensor Backbone model for unimodal training and Multimodal Fusion.

Responsible: Member 3 (Sensor / Tabular Modality)
Project: Durian Ripeness Classification via Cross-Modal Knowledge Distillation (CMKD)
"""

from typing import Sequence, Tuple, Union
import torch
import torch.nn as nn


class SensorBackbone(nn.Module):
    """MLP network extracting sensor feature vector f_sensor in R^{latent_dim}.
    
    Architecture consists of Linear -> LayerNorm -> ReLU -> Dropout blocks,
    finally projecting to latent_dim (default 32) and classifying into 3 ripeness classes.
    Uses LayerNorm for stability across any batch size (including batch_size=1).
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
        """Forward pass: returns logits or (logits, f_sensor)."""
        feat = self.encoder(x)
        logits = self.classifier(feat)
        return (logits, feat) if return_latent else logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature vector f_sensor in R^{latent_dim} for the Fusion Module."""
        return self.encoder(x)
