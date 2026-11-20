"""Vision modality backbone: ConvNeXt V2-Base (f_img in R^1024).

Architecture matches `ConvNeXtV2BaseTeacherVision` from the unimodal vision notebook
(state_dict with `backbone.*` + `head.*`), allowing direct loading of `best_vision_unimodal.pth`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import timm
import torch
import torch.nn as nn

VISION_MODEL_NAME = "convnextv2_base.fcmae_ft_in1k"


def _extract_state_dict(ckpt) -> dict:
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


class VisionBackbone(nn.Module):
    """ConvNeXt V2-Base without classifier, returning feature vectors [B, 1024]."""

    def __init__(self, model_name: str = VISION_MODEL_NAME, pretrained: bool = False,
                 drop_path_rate: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0,
                                          drop_path_rate=drop_path_rate)
        self.num_features = self.backbone.num_features  # 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def load_unimodal(self, ckpt_path: Union[str, Path]) -> "VisionBackbone":
        """Load backbone weights from a unimodal checkpoint (ignores `head.*`, keeps `backbone.*`)."""
        state = _extract_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        if any(k.startswith("backbone.") for k in state):
            state = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Vision checkpoint mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
        return self

