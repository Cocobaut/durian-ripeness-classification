"""Vision-Only Student Model (Stage 2 of CMKD).

Project: Durian Ripeness Classification via Cross-Modal Knowledge Distillation (CMKD)
Specification reference: Section 1.3.3 (Cross-Modal Knowledge Distillation to Vision-Only Student).

Architecture:
    img [B, 3, 384, 384] -> VisionBackbone (e.g., ConvNeXt V2-Tiny) -> f_img [B, d_S] ┬-> Classifier     -> logits [B, 3]
                                                                                     └-> ProjectionHead -> f_proj [B, 256]

Operation:
- During training (Stage 2):
  `forward(img, return_features=True)` returns `(logits, f_img, f_proj)`.
  + `logits` are used for L_CE (hard labels) and L_KD (soft logit matching with Teacher).
  + `f_proj = phi(f_img)` [B, 256] is used for L_feat (latent space alignment with Teacher's f_mm [B, 256]).
- During inference / testing:
  `forward(img)` requires only a single RGB image and returns `logits` [B, 3] (no microphone, sensors, or Teacher required).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.models.vision_backbone import VisionBackbone

STUDENT_BACKBONE_ALIASES: Dict[str, str] = {
    "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in1k",
    "convnextv2_nano": "convnextv2_nano.fcmae_ft_in1k",
    "convnextv2_base": "convnextv2_base.fcmae_ft_in1k",
    "resnet18": "resnet18",
    "resnet34": "resnet34",
    "resnet50": "resnet50",
    "mobilenetv3_large": "mobilenetv3_large_100",
    "efficientnet_b0": "efficientnet_b0",
}

DEFAULT_STUDENT_BACKBONE = "convnextv2_tiny.fcmae_ft_in1k"


def resolve_student_backbone_name(name: str) -> str:
    """Resolve short alias (e.g., 'convnextv2_tiny') to canonical timm model name."""
    key = name.strip().lower()
    return STUDENT_BACKBONE_ALIASES.get(key, name.strip())


@dataclass
class StudentConfig:
    backbone_name: str = DEFAULT_STUDENT_BACKBONE
    pretrained: bool = True
    drop_path_rate: float = 0.2
    num_classes: int = 3
    feat_dim: int = 0  # 0 = auto-infer from backbone.num_features (e.g., 768 for convnextv2_tiny)
    proj_hidden_dim: int = 512
    latent_dim_mm: int = 256
    head_dropout: float = 0.2


class ProjectionHead(nn.Module):
    """Adaptive projection layer phi(.): maps f_img [B, d_S] -> f_proj [B, latent_dim_mm (256)].

    Architecture:
        Linear(in_dim, hidden_dim) -> LayerNorm -> GELU -> Dropout -> Linear(hidden_dim, out_dim) -> LayerNorm
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        out_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VisionStudent(nn.Module):
    """Vision-Only Student model taking RGB images exclusively."""

    def __init__(
        self,
        config: Optional[StudentConfig] = None,
        vision_ckpt: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if config is None:
            config = StudentConfig(**{k: v for k, v in kwargs.items() if k in StudentConfig.__dataclass_fields__})
        elif kwargs:
            cfg_dict = asdict(config)
            cfg_dict.update({k: v for k, v in kwargs.items() if k in StudentConfig.__dataclass_fields__})
            config = StudentConfig(**cfg_dict)

        config.backbone_name = resolve_student_backbone_name(config.backbone_name)
        self.config = config
        c = self.config

        self.vision_backbone = VisionBackbone(
            model_name=c.backbone_name,
            pretrained=c.pretrained,
            drop_path_rate=c.drop_path_rate,
        )
        self.feat_dim = int(self.vision_backbone.num_features)
        self.config.feat_dim = self.feat_dim

        self.projection_head = ProjectionHead(
            in_dim=self.feat_dim,
            hidden_dim=c.proj_hidden_dim,
            out_dim=c.latent_dim_mm,
            dropout=c.head_dropout,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(c.head_dropout),
            nn.Linear(self.feat_dim, c.num_classes),
        )
        self.backbone_frozen = False

        if vision_ckpt is not None and Path(vision_ckpt).is_file():
            self.vision_backbone.load_unimodal(vision_ckpt)

    def freeze_backbone(self) -> None:
        """Freeze VisionBackbone (used during initial warm-up phase if needed)."""
        for p in self.vision_backbone.parameters():
            p.requires_grad = False
        self.backbone_frozen = True
        self.train(self.training)

    def unfreeze_all(self) -> None:
        """Unfreeze the entire Student network."""
        for p in self.vision_backbone.parameters():
            p.requires_grad = True
        self.backbone_frozen = False
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.backbone_frozen:
            self.vision_backbone.eval()
        return self

    def param_groups(
        self,
        backbone_lr: float,
        head_lr: float,
        backbone_wd: float = 1e-4,
        head_wd: float = 1e-2,
    ) -> List[Dict]:
        """Differential learning rates: lower LR for backbone, higher LR for ProjectionHead + Classifier."""
        groups: List[Dict] = []
        bb_params = [p for p in self.vision_backbone.parameters() if p.requires_grad]
        if bb_params:
            groups.append({
                "name": "backbone",
                "params": bb_params,
                "lr": backbone_lr,
                "weight_decay": backbone_wd,
            })
        head_params = [
            p for p in list(self.projection_head.parameters()) + list(self.classifier.parameters())
            if p.requires_grad
        ]
        groups.append({
            "name": "student_head",
            "params": head_params,
            "lr": head_lr,
            "weight_decay": head_wd,
        })
        return groups

    def forward(
        self,
        img: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """RGB-only forward pass.

        Args:
            img: RGB image tensor [B, 3, H, W].
            return_features: If True, returns `(logits, f_img, f_proj)` for distillation loss.
                             If False, returns `logits` [B, num_classes] for inference/evaluation.
        """
        f_img = self.vision_backbone(img)
        logits = self.classifier(f_img)
        if return_features:
            f_proj = self.projection_head(f_img)
            return logits, f_img, f_proj
        return logits


# ------------------------------------------------------------------------------
# Save / load Student checkpoint
# ------------------------------------------------------------------------------
def student_checkpoint_dict(model: VisionStudent, **extra) -> Dict:
    state = {k[7:] if k.startswith("module.") else k: v for k, v in model.state_dict().items()}
    return {
        "model_state_dict": state,
        "student_config": asdict(model.config),
        "class_order": ["Unripe", "Ripe", "Overripe"],
        "feat_dim": model.feat_dim,
        "latent_dim_mm": model.config.latent_dim_mm,
        **extra,
    }


def build_student_from_checkpoint(
    ckpt_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    eval_mode: bool = True,
) -> VisionStudent:
    """Reconstruct VisionStudent from `best_student.pth` (no internet download needed for pretrained weights)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = dict(ckpt.get("student_config", {}))
    cfg_dict["pretrained"] = False
    model = VisionStudent(StudentConfig(**cfg_dict))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    if eval_mode:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    return model

