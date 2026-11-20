"""Multi-Modal Teacher Model (Stage 1B of CMKD).

Project: Durian Ripeness Classification via Cross-Modal Knowledge Distillation (CMKD)
Specification reference: Section 1.3.2.

Architecture:
    img    [B,3,384,384] -> VisionBackbone (ConvNeXt V2-Base) -> f_img    [B,1024] ┐
    audio  [B,1,64,306]  -> AudioBackbone  (ResNet-18)        -> f_audio  [B,512]  ├-> CrossAttentionFusion -> f_mm [B,256] -> classifier -> logits [B,3]
    sensor [B,21]        -> SensorBackbone (MLP)              -> f_sensor [B,32]   ┘

CrossAttentionMultimodalFusion:
1. Project each modality feature to proj_dim (default 256) (Linear + LayerNorm + GELU).
2. Add learnable Modality Embeddings identifying each sensory modality.
3. Transformer Cross-Attention (Multi-Head Self/Cross Attention):
   - Modality tokens (Vision, Audio, Sensor) interact and enrich one another.
   - Vision token is enriched by knock acoustics and sensor measurements.
   - Absent modalities are safely masked via `key_padding_mask` (prevents noise when data is missing).
4. Adaptive Modality Gate: learns the importance of each branch after cross-modal interaction.
5. Concatenate and project through MLP -> f_mm [B, 256].

Missing modality: all-zero input (missing data in dataset, or zero-masking ablation in
eval_teacher.py) => that branch's features are masked and ignored by the gate.
Modality dropout (training only): randomly drops 1 branch so Teacher does not rely on a single modality.

forward(img, audio, sensor) -> (logits, f_mm)   # f_mm is used for feature alignment in Student
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.models.audio_backbone import AudioBackbone
from src.models.sensor_backbone import SensorBackbone
from src.models.vision_backbone import (
    VISION_ATTN_HIDDEN_DIM,
    VISION_EMBED_DIM,
    VISION_MODEL_NAME,
    VisionBackbone,
)

MODALITIES = ("vision", "audio", "sensor")


@dataclass
class TeacherConfig:
    num_classes: int = 3
    proj_dim: int = 256
    latent_dim_mm: int = 256
    fusion_hidden: int = 512
    fusion_dropout: float = 0.3
    modality_dropout: float = 0.15
    fusion_type: str = "cross_attention"  # "cross_attention" or "gated_concat"
    num_attn_heads: int = 4
    num_attn_layers: int = 1
    vision_model_name: str = VISION_MODEL_NAME
    vision_drop_path: float = 0.2
    vision_dim: int = VISION_EMBED_DIM  # 256-d F_rgb from Projection + LightweightGatedAttention
    vision_attn_hidden_dim: int = VISION_ATTN_HIDDEN_DIM  # 128
    vision_proj_dropout: float = 0.25
    vision_attn_dropout: float = 0.1
    audio_dim: int = 512
    sensor_input_dim: int = 21
    sensor_hidden_dims: List[int] = field(default_factory=lambda: [64, 48])
    sensor_latent_dim: int = 32
    sensor_dropout: float = 0.3


class CrossAttentionMultimodalFusion(nn.Module):
    """Multimodal fusion using Multi-Head Transformer Cross-Attention.

    Enables bidirectional cross-interaction across Vision, Audio, and Sensor:
    - Vision is supported by hollow/dense knock cues from Audio and physicochemical readings from Sensor.
    - Handles missing modalities cleanly via `key_padding_mask`.
    - Produces `gate_weights` [B, 3] compatible with the evaluation suite and Modality Ablation.
    """

    def __init__(
        self,
        in_dims: Dict[str, int],
        proj_dim: int = 256,
        hidden: int = 512,
        out_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.modalities = list(in_dims)
        self.proj_dim = proj_dim
        n = len(self.modalities)

        # 1. Project each modality into shared proj_dim space
        self.proj = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(d, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
            for m, d in in_dims.items()
        })

        # 2. Modality Embeddings to distinguish Vision, Audio, and Sensor tokens
        self.modality_embed = nn.Parameter(torch.randn(1, n, proj_dim) * 0.02)

        # 3. Transformer Cross-Attention Layer(s) with Pre-LN
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=proj_dim,
                nhead=num_heads,
                dim_feedforward=hidden,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(proj_dim)

        # 4. Adaptive Modality Gate
        self.gate = nn.Sequential(
            nn.Linear(n * proj_dim, n * 32),
            nn.GELU(),
            nn.Linear(n * 32, n),
        )

        # 5. Final fusion MLP -> f_mm
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n * proj_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, feats: Dict[str, torch.Tensor], present: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feats: {modality: [B, d_m]}.
            present: [B, n_modalities] (1 = present, 0 = absent).
        Returns:
            f_mm [B, out_dim], gate_weights [B, n_modalities].
        """
        # [B, n, proj_dim]
        z = torch.stack([self.proj[m](feats[m]) for m in self.modalities], dim=1)
        z = z + self.modality_embed

        # Mask absent modalities: True in mask means ignored
        key_padding_mask = (present <= 0)
        # Guard against edge case where all modalities are absent (anomalous empty sample)
        all_missing = key_padding_mask.all(dim=-1, keepdim=True)
        safe_mask = torch.where(all_missing, torch.zeros_like(key_padding_mask), key_padding_mask)

        out = z
        for layer in self.layers:
            out = layer(out, src_key_padding_mask=safe_mask)
        out = self.norm(out)

        # Zero out representations of absent branches
        out = out * present.unsqueeze(-1)

        # Compute adaptive gate weights
        gate_logits = self.gate(out.flatten(1)).masked_fill(present <= 0, -1e4)
        w = torch.softmax(gate_logits.float(), dim=-1).to(out.dtype) * present

        # Apply gate weights and aggregate via MLP
        fused = (out * w.unsqueeze(-1) * out.size(1)).flatten(1)
        f_mm = self.mlp(fused)
        return f_mm, w


class GatedConcatMultimodalFusion(nn.Module):
    """Legacy Gated Concat fusion mechanism (retained for backward compatibility)."""

    def __init__(self, in_dims: Dict[str, int], proj_dim: int = 256, hidden: int = 512,
                 out_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.modalities = list(in_dims)
        self.proj = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(d, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
            for m, d in in_dims.items()
        })
        n = len(self.modalities)
        self.gate = nn.Sequential(nn.Linear(n * proj_dim, n * 32), nn.GELU(), nn.Linear(n * 32, n))
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n * proj_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim), nn.GELU(),
        )

    def forward(self, feats: Dict[str, torch.Tensor], present: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.stack([self.proj[m](feats[m]) for m in self.modalities], dim=1)
        z = z * present.unsqueeze(-1)
        gate_logits = self.gate(z.flatten(1)).masked_fill(present <= 0, -1e4)
        w = torch.softmax(gate_logits.float(), dim=-1).to(z.dtype) * present
        fused = (z * w.unsqueeze(-1) * z.size(1)).flatten(1)
        return self.mlp(fused), w


# Use CrossAttentionMultimodalFusion by default
MultimodalFusion = CrossAttentionMultimodalFusion


class MultiModalTeacher(nn.Module):
    def __init__(
        self,
        config: Optional[TeacherConfig] = None,
        vision_ckpt: Optional[Union[str, Path]] = None,
        audio_ckpt: Optional[Union[str, Path]] = None,
        sensor_ckpt: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        super().__init__()
        if config is None:
            config = TeacherConfig(**{k: v for k, v in kwargs.items() if k in TeacherConfig.__dataclass_fields__})
        elif kwargs:
            cfg_dict = asdict(config)
            cfg_dict.update({k: v for k, v in kwargs.items() if k in TeacherConfig.__dataclass_fields__})
            config = TeacherConfig(**cfg_dict)

        # If sensor_ckpt is provided, automatically sync SensorBackbone dimension config
        if sensor_ckpt and os.path.isfile(sensor_ckpt):
            s_meta = torch.load(sensor_ckpt, map_location="cpu", weights_only=False)
            if isinstance(s_meta, dict):
                config.sensor_input_dim = int(s_meta.get("input_dim", config.sensor_input_dim))
                config.sensor_hidden_dims = list(s_meta.get("hidden_dims", config.sensor_hidden_dims))
                config.sensor_latent_dim = int(s_meta.get("latent_dim", config.sensor_latent_dim))
                config.sensor_dropout = float(s_meta.get("dropout", config.sensor_dropout))

        self.config = config
        c = self.config
        self.vision_backbone = VisionBackbone(
            c.vision_model_name,
            pretrained=False,
            drop_path_rate=c.vision_drop_path,
            embed_dim=c.vision_dim,
            attn_hidden_dim=c.vision_attn_hidden_dim,
            proj_dropout=c.vision_proj_dropout,
            attn_dropout=c.vision_attn_dropout,
        )
        self.audio_backbone = AudioBackbone(pretrained=False)
        self.sensor_backbone = SensorBackbone(
            input_dim=c.sensor_input_dim,
            hidden_dims=tuple(c.sensor_hidden_dims),
            latent_dim=c.sensor_latent_dim,
            num_classes=c.num_classes,
            dropout=c.sensor_dropout,
        )
        # Standalone SensorMLP classifier head is unused in Teacher -> permanently frozen
        for p in self.sensor_backbone.classifier.parameters():
            p.requires_grad = False

        if getattr(c, "fusion_type", "cross_attention") == "gated_concat":
            self.fusion = GatedConcatMultimodalFusion(
                {"vision": c.vision_dim, "audio": c.audio_dim, "sensor": c.sensor_latent_dim},
                proj_dim=c.proj_dim,
                hidden=c.fusion_hidden,
                out_dim=c.latent_dim_mm,
                dropout=c.fusion_dropout,
            )
        else:
            self.fusion = CrossAttentionMultimodalFusion(
                {"vision": c.vision_dim, "audio": c.audio_dim, "sensor": c.sensor_latent_dim},
                proj_dim=c.proj_dim,
                hidden=c.fusion_hidden,
                out_dim=c.latent_dim_mm,
                num_heads=c.num_attn_heads,
                num_layers=c.num_attn_layers,
                dropout=c.fusion_dropout,
            )

        self.classifier = nn.Linear(c.latent_dim_mm, c.num_classes)
        self.backbones_frozen = False
        self.last_gate_weights: Optional[torch.Tensor] = None

        if any(ck is not None for ck in (vision_ckpt, audio_ckpt, sensor_ckpt)):
            self.load_unimodal_backbones(
                vision_ckpt if (vision_ckpt and os.path.isfile(vision_ckpt)) else None,
                audio_ckpt if (audio_ckpt and os.path.isfile(audio_ckpt)) else None,
                sensor_ckpt if (sensor_ckpt and os.path.isfile(sensor_ckpt)) else None,
            )

    # ------------------------------------------------------------------
    # Load unimodal weights (Stage 1A)
    # ------------------------------------------------------------------
    def load_unimodal_backbones(
        self,
        vision_ckpt: Optional[Union[str, Path]] = None,
        audio_ckpt: Optional[Union[str, Path]] = None,
        sensor_ckpt: Optional[Union[str, Path]] = None,
    ) -> Dict[str, str]:
        loaded = {}
        if vision_ckpt:
            self.vision_backbone.load_unimodal(vision_ckpt)
            loaded["vision"] = str(vision_ckpt)
        if audio_ckpt:
            self.audio_backbone.load_unimodal(audio_ckpt)
            loaded["audio"] = str(audio_ckpt)
        if sensor_ckpt:
            ckpt = torch.load(sensor_ckpt, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "full_state_dict" in ckpt:
                self.sensor_backbone.load_state_dict(ckpt["full_state_dict"], strict=True)
            elif isinstance(ckpt, dict) and "backbone_state_dict" in ckpt:
                self.sensor_backbone.encoder.load_state_dict(ckpt["backbone_state_dict"], strict=True)
            else:
                self.sensor_backbone.load_state_dict(ckpt, strict=True)
            loaded["sensor"] = str(sensor_ckpt)
        return loaded

    # ------------------------------------------------------------------
    # Two-phase training strategy
    # ------------------------------------------------------------------
    def backbone_modules(self) -> Dict[str, nn.Module]:
        return {"vision": self.vision_backbone, "audio": self.audio_backbone, "sensor": self.sensor_backbone.encoder}

    def freeze_backbones(self) -> None:
        """Phase 1 (Warm-up): freeze all 3 backbones and keep BatchNorm/DropPath in eval mode."""
        for m in self.backbone_modules().values():
            for p in m.parameters():
                p.requires_grad = False
        self.backbones_frozen = True
        self.train(self.training)

    def unfreeze_all(self) -> None:
        """Phase 2 (Joint fine-tuning): unfreeze all modules (except the standalone SensorMLP classifier head)."""
        for m in self.backbone_modules().values():
            for p in m.parameters():
                p.requires_grad = True
        self.backbones_frozen = False
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.backbones_frozen:
            for m in self.backbone_modules().values():
                m.eval()
        return self

    def param_groups(
        self,
        backbone_lr: float,
        head_lr: float,
        backbone_wd: float = 1e-4,
        head_wd: float = 1e-2,
    ) -> List[Dict]:
        """Differential learning rates: lower LR for backbones, higher LR for Fusion + Head."""
        groups = []
        for name, m in self.backbone_modules().items():
            params = [p for p in m.parameters() if p.requires_grad]
            if params:
                groups.append({"name": name, "params": params, "lr": backbone_lr, "weight_decay": backbone_wd})
        head = [p for p in list(self.fusion.parameters()) + list(self.classifier.parameters()) if p.requires_grad]
        groups.append({"name": "fusion_head", "params": head, "lr": head_lr, "weight_decay": head_wd})
        return groups

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    @staticmethod
    def _present(x: torch.Tensor) -> torch.Tensor:
        return (x.detach().flatten(1).abs().amax(dim=1) > 0).float()

    def _modality_dropout(self, present: torch.Tensor) -> torch.Tensor:
        p = self.config.modality_dropout
        if not self.training or p <= 0:
            return present
        drop = (torch.rand_like(present) < p).float()
        kept = present * (1 - drop)
        # Always retain at least 1 branch: restore original if all modalities were dropped
        empty = kept.sum(dim=1, keepdim=True) == 0
        return torch.where(empty, present, kept)

    def forward(
        self,
        img: torch.Tensor,
        audio: torch.Tensor,
        sensor: torch.Tensor,
        return_details: bool = False,
    ):
        present = torch.stack([self._present(img), self._present(audio), self._present(sensor)], dim=1)
        present = self._modality_dropout(present)

        feats = {
            "vision": self.vision_backbone(img),
            "audio": self.audio_backbone(audio),
            "sensor": self.sensor_backbone.extract_features(sensor),
        }
        f_mm, gate_w = self.fusion(feats, present)
        logits = self.classifier(f_mm)
        self.last_gate_weights = gate_w.detach()
        if return_details:
            return logits, f_mm, {"features": feats, "present": present, "gate_weights": gate_w}
        return logits, f_mm


# ------------------------------------------------------------------------------
# Save / load Teacher checkpoint
# ------------------------------------------------------------------------------
def teacher_checkpoint_dict(model: MultiModalTeacher, **extra) -> Dict:
    state = {k[7:] if k.startswith("module.") else k: v for k, v in model.state_dict().items()}
    return {
        "model_state_dict": state,
        "teacher_config": asdict(model.config),
        "class_order": ["Unripe", "Ripe", "Overripe"],
        "feature_dim_mm": model.config.latent_dim_mm,
        **extra,
    }


def build_teacher_from_checkpoint(
    ckpt_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    eval_mode: bool = True,
) -> MultiModalTeacher:
    """Reconstruct Teacher from `best_teacher.pth` (for eval_teacher.py and Student distillation)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_cfg = dict(ckpt.get("teacher_config", {}))
    valid_cfg = {k: v for k, v in raw_cfg.items() if k in TeacherConfig.__dataclass_fields__}
    model = MultiModalTeacher(TeacherConfig(**valid_cfg))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    if eval_mode:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    return model
