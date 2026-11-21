"""Audio modality backbone: ResNet-18 on log-Mel [1, 64, 306] (f_audio in R^512).

Matches Member 2's `AudioResNet18.backbone`
(workspaces/member2_audio/02_train_unimodal_audio/train_audio.py) and loads either
`audio_backbone_teacher.pth` (`backbone_state_dict` key) or `best_audio_unimodal.pth`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import timm
import torch
import torch.nn as nn


class AudioBackbone(nn.Module):
    """ResNet-18 (timm, in_chans=1, num_classes=0) returning feature vectors [B, 512]."""

    def __init__(self, pretrained: bool = False, drop_rate: float = 0.0):
        super().__init__()
        self.backbone = timm.create_model("resnet18", pretrained=pretrained, in_chans=1, num_classes=0,
                                          drop_rate=drop_rate)
        self.num_features = self.backbone.num_features  # 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def load_unimodal(self, ckpt_path: Union[str, Path]) -> "AudioBackbone":
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "backbone_state_dict" in ckpt:
            state = ckpt["backbone_state_dict"]
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = {k[len("backbone."):]: v for k, v in ckpt["model_state_dict"].items()
                     if k.startswith("backbone.")}
        else:
            state = ckpt
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Audio checkpoint mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
        return self

