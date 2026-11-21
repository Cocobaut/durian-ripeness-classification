"""Vision-Only Student Model (Stage 2 of CMKD) with Multi-View Gated Attention Pooling.

Project: Durian Ripeness Classification via Cross-Modal Knowledge Distillation (CMKD)
Specification reference: Section 1.3.3 (Cross-Modal Knowledge Distillation to Vision-Only Student).

Architecture:
    views [B, N, 3, 384, 384] -> VisionBackbone (ConvNeXt V2-Base + 1024->256 Proj + Lightweight Gated Attention)
                              -> F_rgb [B, 256] ┬-> Classifier     -> logits [B, 3]
                                                └-> ProjectionHead -> f_proj [B, 256]

Operation:
- Initialized from the converged Stage 1A Multi-View ConvNeXt V2-Base checkpoint
  (`best_convnextv2_multiview.pth` or `convnextv2_multiview_backbone_256d.pth`).
- During training (Stage 2):
  `forward(img, return_features=True)` returns `(logits, f_img, f_proj)`.
  + `logits` are used for L_CE (hard labels) and L_KD (soft logit matching with Teacher).
  + `f_proj = phi(f_img)` [B, 256] is used for L_feat (latent space alignment with Teacher's f_mm [B, 256]).
- During inference / testing:
  `forward(img)` requires only RGB image view(s) (`[B, N, 3, H, W]` or `[B, 3, H, W]`) and returns `logits` [B, 3].
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.models.vision_backbone import (
    VISION_ATTN_HIDDEN_DIM,
    VISION_EMBED_DIM,
    VISION_MODEL_NAME,
    VisionBackbone,
    _extract_state_dict,
    _strip_module_prefix,
)

STUDENT_BACKBONE_ALIASES: Dict[str, str] = {
    "convnextv2_base": "convnextv2_base.fcmae_ft_in1k",
    "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in1k",
    "convnextv2_nano": "convnextv2_nano.fcmae_ft_in1k",
    "resnet18": "resnet18",
    "resnet34": "resnet34",
    "resnet50": "resnet50",
    "mobilenetv3_large": "mobilenetv3_large_100",
    "efficientnet_b0": "efficientnet_b0",
}

DEFAULT_STUDENT_BACKBONE = VISION_MODEL_NAME


def resolve_student_backbone_name(name: str) -> str:
    """Resolve short alias (e.g., 'convnextv2_base') to canonical timm model name."""
    key = name.strip().lower()
    return STUDENT_BACKBONE_ALIASES.get(key, name.strip())


@dataclass
class StudentConfig:
    backbone_name: str = DEFAULT_STUDENT_BACKBONE
    pretrained: bool = True
    drop_path_rate: float = 0.2
    num_classes: int = 3
    feat_dim: int = VISION_EMBED_DIM  # 256-d F_rgb output from Multi-View Projection + Gated Attention
    attn_hidden_dim: int = VISION_ATTN_HIDDEN_DIM  # 128
    proj_dropout: float = 0.25
    attn_dropout: float = 0.1
    proj_hidden_dim: int = 512
    latent_dim_mm: int = 256
    head_dropout: float = 0.25


class ProjectionHead(nn.Module):
    """Adaptive projection layer phi(.): maps F_rgb [B, 256] -> f_proj [B, latent_dim_mm (256)].

    Architecture:
        Linear(in_dim, hidden_dim) -> LayerNorm -> GELU -> Dropout -> Linear(hidden_dim, out_dim) -> LayerNorm
    """

    def __init__(
        self,
        in_dim: int = VISION_EMBED_DIM,
        hidden_dim: int = 512,
        out_dim: int = 256,
        dropout: float = 0.25,
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
    """Vision-Only Student model taking Multi-View RGB fruit bags (or single RGB views) exclusively."""

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

        has_unimodal_ckpt = vision_ckpt is not None and Path(vision_ckpt).is_file()
        embed_dim = int(c.feat_dim) if int(c.feat_dim) > 0 else VISION_EMBED_DIM

        self.vision_backbone = VisionBackbone(
            model_name=c.backbone_name,
            pretrained=(c.pretrained and not has_unimodal_ckpt),
            drop_path_rate=c.drop_path_rate,
            embed_dim=embed_dim,
            attn_hidden_dim=c.attn_hidden_dim,
            proj_dropout=c.proj_dropout,
            attn_dropout=c.attn_dropout,
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
        # When True, inference-mode forward passes (return_features=False) use 3-view TTA
        # (original + hflip + 95% zoom), matching the unimodal Vision baseline evaluation protocol.
        self.tta_enabled = False

        if has_unimodal_ckpt:
            self.load_unimodal(vision_ckpt)

    def load_unimodal(self, ckpt_path: Union[str, Path]) -> "VisionStudent":
        """Load unimodal Multi-View ConvNeXtV2 weights (backbone + projection + gated_attention, plus classifier if present)."""
        self.vision_backbone.load_unimodal(ckpt_path)
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = _strip_module_prefix(_extract_state_dict(raw))
        if isinstance(state, dict) and any(k.startswith("classifier.") for k in state):
            cls_state = {k[len("classifier."):]: v for k, v in state.items() if k.startswith("classifier.")}
            if "2.weight" in cls_state and cls_state["2.weight"].shape == self.classifier[2].weight.shape:
                self.classifier.load_state_dict(cls_state, strict=False)
        return self

    def freeze_backbone(self) -> None:
        """Freeze VisionBackbone (ConvNeXtV2 + Projection + GatedAttention) during Phase 1 warm-up."""
        for p in self.vision_backbone.parameters():
            p.requires_grad = False
        self.backbone_frozen = True
        self.train(self.training)

    def unfreeze_all(self) -> None:
        """Unfreeze the entire Student network for Phase 2 joint distillation."""
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
        """Differential learning rates: lower LR for VisionBackbone, higher LR for ProjectionHead + Classifier."""
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
        view_counts: Optional[List[int]] = None,
        tta: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """RGB-only forward pass (supports 5D multi-view fruit bags `[B, N, 3, H, W]` or 4D `[B, 3, H, W]`).

        Args:
            img: RGB image tensor `[B, N, 3, H, W]` or `[B, 3, H, W]`.
            return_features: If True, returns `(logits, f_img, f_proj)` for distillation loss.
                             If False, returns `logits` `[B, num_classes]` for inference/evaluation.
            view_counts: Optional view counts per fruit if `img` is flattened `[Total_Views, 3, H, W]`.
            tta: Apply 3-view TTA. Defaults to `self.tta_enabled` for inference (never used for training).
        """
        if tta is None:
            tta = self.tta_enabled and not return_features and not self.training
        f_img = self.vision_backbone(img, view_counts=view_counts, tta=tta)
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
    raw_cfg = dict(ckpt.get("student_config", {}))
    valid_cfg = {k: v for k, v in raw_cfg.items() if k in StudentConfig.__dataclass_fields__}
    valid_cfg["pretrained"] = False
    model = VisionStudent(StudentConfig(**valid_cfg))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    if eval_mode:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    return model
