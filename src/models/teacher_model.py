"""Mô hình Multi-Modal Teacher (Giai đoạn 1B của CMKD).

Dự án: Phân loại độ chín sầu riêng bằng Chưng cất tri thức Đa phương thức (CMKD)
Tham chiếu đặc tả: Mục 1.3.2.

Kiến trúc:
    img    [B,3,384,384] -> VisionBackbone (ConvNeXt V2-Base) -> f_img    [B,1024] ┐
    audio  [B,1,64,306]  -> AudioBackbone  (ResNet-18)        -> f_audio  [B,512]  ├-> CrossAttentionFusion -> f_mm [B,256] -> classifier -> logits [B,3]
    sensor [B,21]        -> SensorBackbone (MLP)              -> f_sensor [B,32]   ┘

CrossAttentionMultimodalFusion:
1. Chiếu từng nhánh đặc trưng về proj_dim (mặc định 256) (Linear + LayerNorm + GELU).
2. Gán learnable Modality Embeddings định danh từng giác quan.
3. Transformer Cross-Attention (Multi-Head Self/Cross Attention):
   - Các token phương thức (Vision, Audio, Sensor) tương tác và làm giàu thông tin cho nhau.
   - Token Vision được làm giàu bởi âm thanh tiếng gõ và chỉ số cảm biến.
   - Nhánh vắng mặt được che an toàn bằng `key_padding_mask` (chống nhiễu khi thiếu dữ liệu).
4. Cổng thích ứng (Modality Gate): Học tầm quan trọng của từng nhánh sau khi tương tác chéo.
5. Ghép nối và chiếu qua MLP -> f_mm [B, 256].

Vắng mặt phương thức: input toàn 0 (dataset thiếu dữ liệu, hoặc zero-masking ablation của
eval_teacher.py) => đặc trưng nhánh đó bị che và gate bỏ qua nhánh đó.
Modality dropout (chỉ khi train): ngẫu nhiên che 1 nhánh để Teacher không phụ thuộc một nhánh duy nhất.

forward(img, audio, sensor) -> (logits, f_mm)   # f_mm dùng cho feature alignment ở Student
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
from src.models.vision_backbone import VISION_MODEL_NAME, VisionBackbone

MODALITIES = ("vision", "audio", "sensor")


@dataclass
class TeacherConfig:
    num_classes: int = 3
    proj_dim: int = 256
    latent_dim_mm: int = 256
    fusion_hidden: int = 512
    fusion_dropout: float = 0.3
    modality_dropout: float = 0.15
    fusion_type: str = "cross_attention"  # "cross_attention" hoặc "gated_concat"
    num_attn_heads: int = 4
    num_attn_layers: int = 1
    vision_model_name: str = VISION_MODEL_NAME
    vision_drop_path: float = 0.2
    vision_dim: int = 1024
    audio_dim: int = 512
    sensor_input_dim: int = 21
    sensor_hidden_dims: List[int] = field(default_factory=lambda: [64, 48])
    sensor_latent_dim: int = 32
    sensor_dropout: float = 0.3


class CrossAttentionMultimodalFusion(nn.Module):
    """Hợp nhất đa phương thức bằng Transformer Cross-Attention đa đầu.

    Cho phép Vision, Audio, Sensor tương tác chéo hai chiều:
    - Vision được hỗ trợ từ tín hiệu rỗng/đặc của Audio và chỉ số lý hóa của Sensor.
    - Xử lý hoàn hảo nhánh bị thiếu thông qua `key_padding_mask`.
    - Sinh ra `gate_weights` [B, 3] tương thích với hệ thống đánh giá và Modality Ablation.
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

        # 1. Chiếu từng phương thức về không gian chung proj_dim
        self.proj = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(d, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
            for m, d in in_dims.items()
        })

        # 2. Modality Embeddings để phân biệt đặc trưng đến từ Vision, Audio hay Sensor
        self.modality_embed = nn.Parameter(torch.randn(1, n, proj_dim) * 0.02)

        # 3. Transformer Cross-Attention Layer(s) với Pre-LN
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

        # 4. Cổng thích ứng (Modality Gate)
        self.gate = nn.Sequential(
            nn.Linear(n * proj_dim, n * 32),
            nn.GELU(),
            nn.Linear(n * 32, n),
        )

        # 5. MLP tổng hợp cuối cùng -> f_mm
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
            present: [B, n_modalities] (1 = có, 0 = vắng).
        Returns:
            f_mm [B, out_dim], gate_weights [B, n_modalities].
        """
        # [B, n, proj_dim]
        z = torch.stack([self.proj[m](feats[m]) for m in self.modalities], dim=1)
        z = z + self.modality_embed

        # Che các phương thức vắng mặt: True trong mask nghĩa là bị che
        key_padding_mask = (present <= 0)
        # Phòng trường hợp toàn bộ nhánh vắng mặt (mẫu rỗng bất thường)
        all_missing = key_padding_mask.all(dim=-1, keepdim=True)
        safe_mask = torch.where(all_missing, torch.zeros_like(key_padding_mask), key_padding_mask)

        out = z
        for layer in self.layers:
            out = layer(out, src_key_padding_mask=safe_mask)
        out = self.norm(out)

        # Triệt tiêu biểu diễn của nhánh vắng mặt
        out = out * present.unsqueeze(-1)

        # Tính toán cổng thích ứng (Gate)
        gate_logits = self.gate(out.flatten(1)).masked_fill(present <= 0, -1e4)
        w = torch.softmax(gate_logits.float(), dim=-1).to(out.dtype) * present

        # Nhân trọng số cổng và tổng hợp qua MLP
        fused = (out * w.unsqueeze(-1) * out.size(1)).flatten(1)
        f_mm = self.mlp(fused)
        return f_mm, w


class GatedConcatMultimodalFusion(nn.Module):
    """Cơ chế Gated Concat cũ (để giữ tính tương thích ngược)."""

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


# Mặc định sử dụng CrossAttentionMultimodalFusion
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

        # Nếu có sensor_ckpt, tự động đồng bộ cấu hình chiều của SensorBackbone
        if sensor_ckpt and os.path.isfile(sensor_ckpt):
            s_meta = torch.load(sensor_ckpt, map_location="cpu", weights_only=False)
            if isinstance(s_meta, dict):
                config.sensor_input_dim = int(s_meta.get("input_dim", config.sensor_input_dim))
                config.sensor_hidden_dims = list(s_meta.get("hidden_dims", config.sensor_hidden_dims))
                config.sensor_latent_dim = int(s_meta.get("latent_dim", config.sensor_latent_dim))
                config.sensor_dropout = float(s_meta.get("dropout", config.sensor_dropout))

        self.config = config
        c = self.config
        self.vision_backbone = VisionBackbone(c.vision_model_name, pretrained=False, drop_path_rate=c.vision_drop_path)
        self.audio_backbone = AudioBackbone(pretrained=False)
        self.sensor_backbone = SensorBackbone(
            input_dim=c.sensor_input_dim,
            hidden_dims=tuple(c.sensor_hidden_dims),
            latent_dim=c.sensor_latent_dim,
            num_classes=c.num_classes,
            dropout=c.sensor_dropout,
        )
        # Head phân loại riêng của SensorMLP không dùng trong Teacher -> đóng băng vĩnh viễn
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
    # Nạp trọng số unimodal (Giai đoạn 1A)
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
    # Chiến lược huấn luyện 2 pha
    # ------------------------------------------------------------------
    def backbone_modules(self) -> Dict[str, nn.Module]:
        return {"vision": self.vision_backbone, "audio": self.audio_backbone, "sensor": self.sensor_backbone.encoder}

    def freeze_backbones(self) -> None:
        """Pha 1 (Warm-up): đóng băng 3 backbone, giữ BatchNorm/DropPath ở chế độ eval."""
        for m in self.backbone_modules().values():
            for p in m.parameters():
                p.requires_grad = False
        self.backbones_frozen = True
        self.train(self.training)

    def unfreeze_all(self) -> None:
        """Pha 2 (Joint fine-tuning): mở khóa toàn bộ (trừ head riêng của SensorMLP)."""
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
        """Differential learning rates: backbone LR nhỏ, Fusion + Head LR lớn."""
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
        # Luôn giữ ít nhất 1 nhánh: mẫu nào bị che hết thì khôi phục như cũ
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
# Lưu / nạp checkpoint Teacher
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
    """Dựng lại Teacher từ `best_teacher.pth` (dùng cho eval_teacher.py và chưng cất Student)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MultiModalTeacher(TeacherConfig(**ckpt.get("teacher_config", {})))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    if eval_mode:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    return model
