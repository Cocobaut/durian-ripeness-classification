"""Vision modality backbone: Multi-View ConvNeXt V2-Base + 256-d Projection + Lightweight Gated Attention.

Architecture matches `ConvNeXtV2MultiViewGatedAttention` from the improved unimodal vision notebook
(`workspaces/member1_vision_lead/02_train_unimodal_vision/train_unimodal_vision_convnextv2_improve.ipynb`):
1. Backbone: ConvNeXt V2-Base (`convnextv2_base.fcmae_ft_in1k`, classifier removed) -> h_i in R^{1024}
2. Compression Projection Block:
   f_i = Dropout(GELU(LayerNorm(W_proj h_i))) in R^{256}
3. Lightweight Gated Attention Pooling (Ilse et al., D=128):
   a_i = exp(w^T (tanh(V f_i) * sigma(U f_i))) / sum_j exp(w^T (tanh(V f_j) * sigma(U f_j)))
   F_rgb = sum_{i=1}^N a_i f_i in R^{256}

Supports loading:
- `convnextv2_multiview_backbone_256d.pth` (`backbone_state_dict`, `projection_state_dict`, `gated_attention_state_dict`)
- `best_convnextv2_multiview.pth` / `last_convnextv2_multiview.pth` (`model_state_dict` with `backbone.*`, `projection.*`, `gated_attention.*`)
- Legacy single-view `best_vision_unimodal.pth` (`backbone.*`)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

VISION_MODEL_NAME = "convnextv2_base.fcmae_ft_in1k"
VISION_EMBED_DIM = 256
VISION_ATTN_HIDDEN_DIM = 128


def _extract_state_dict(ckpt) -> dict:
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def _strip_module_prefix(state: dict) -> dict:
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}


class LightweightGatedAttention(nn.Module):
    """Lightweight Gated Attention Mechanism (Ilse et al., 2018):

    a_i = exp(w^T (tanh(V f_i) * sigma(U f_i))) / sum_j exp(w^T (tanh(V f_j) * sigma(U f_j)))
    F_rgb = sum_{i=1}^N a_i f_i in R^{embed_dim}
    """

    def __init__(self, embed_dim: int = VISION_EMBED_DIM, hidden_dim: int = VISION_ATTN_HIDDEN_DIM, attn_dropout: float = 0.1):
        super().__init__()
        self.V = nn.Linear(embed_dim, hidden_dim)
        self.U = nn.Linear(embed_dim, hidden_dim)
        self.drop = nn.Dropout(p=attn_dropout)
        self.w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, f_views: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool N view embeddings of a single fruit.

        Args:
            f_views: Tensor of shape [N, embed_dim] representing N views of a single fruit.
        Returns:
            F_rgb       : Aggregated fruit representation [embed_dim]
            attn_weights: Normalized attention weights [N] summing to 1.0
        """
        gated_hidden = torch.tanh(self.V(f_views)) * torch.sigmoid(self.U(f_views))
        gated_hidden = self.drop(gated_hidden)
        scores = self.w(gated_hidden)  # [N, 1]
        attn_weights = F.softmax(scores.float(), dim=0).to(f_views.dtype)  # [N, 1]
        F_rgb = torch.sum(attn_weights * f_views, dim=0)  # [embed_dim]
        return F_rgb, attn_weights.squeeze(-1)


class VisionBackbone(nn.Module):
    """Multi-View ConvNeXt V2 with 256-d Projection & Lightweight Gated Attention Pooling, returning F_rgb [B, 256]."""

    def __init__(
        self,
        model_name: str = VISION_MODEL_NAME,
        pretrained: bool = False,
        drop_path_rate: float = 0.2,
        embed_dim: int = VISION_EMBED_DIM,
        attn_hidden_dim: int = VISION_ATTN_HIDDEN_DIM,
        proj_dropout: float = 0.25,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        self.backbone_dim = int(self.backbone.num_features)  # 1024 for convnextv2_base
        self.embed_dim = int(embed_dim)                      # 256
        self.attn_hidden_dim = int(attn_hidden_dim)          # 128
        self.num_features = self.embed_dim                   # 256 (output feature dimension F_rgb)

        # Feature Projection & Compression Block: f_i = Dropout(GELU(LayerNorm(W_proj h_i))) in R^{256}
        self.projection = nn.Sequential(
            nn.Linear(self.backbone_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(p=proj_dropout),
        )

        # Lightweight Gated Attention (Ilse et al.) on 256-d compressed space
        self.gated_attention = LightweightGatedAttention(
            embed_dim=self.embed_dim,
            hidden_dim=self.attn_hidden_dim,
            attn_dropout=attn_dropout,
        )
        self.last_attn_weights: Optional[List[torch.Tensor]] = None

    def extract_view_embeddings(self, flat_images: torch.Tensor, eval_chunk_size: int = 24) -> torch.Tensor:
        """Pass individual views I_i through ConvNeXt V2 + 256-d Projection Block -> f_i in R^{256}."""
        if not torch.is_grad_enabled() and flat_images.size(0) > eval_chunk_size:
            chunks = []
            for chunk in torch.split(flat_images, eval_chunk_size, dim=0):
                h_c = self.backbone(chunk)
                chunks.append(self.projection(h_c))
            return torch.cat(chunks, dim=0)
        h = self.backbone(flat_images)  # [Total_Views, backbone_dim]
        return self.projection(h)       # [Total_Views, embed_dim]

    def extract_view_embeddings_tta(self, flat_images: torch.Tensor) -> torch.Tensor:
        """3-view TTA identical to the unimodal Vision baseline evaluation:
        0.50 * original + 0.30 * horizontal flip + 0.20 * 95% central zoom.
        """
        f_orig = self.extract_view_embeddings(flat_images)
        f_hflip = self.extract_view_embeddings(torch.flip(flat_images, dims=[3]))
        pad = max(1, int(flat_images.shape[-1] * 0.025))
        cropped = flat_images[:, :, pad:-pad, pad:-pad]
        zoomed = nn.functional.interpolate(cropped, size=flat_images.shape[-2:], mode="bilinear", align_corners=False)
        f_zoom = self.extract_view_embeddings(zoomed)
        return 0.50 * f_orig + 0.30 * f_hflip + 0.20 * f_zoom

    def pool_fruit_bags(
        self,
        f_all_views: torch.Tensor,
        view_counts: List[int],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Split flat view embeddings by fruit and pool each fruit bag via Gated Attention."""
        bag_chunks = torch.split(f_all_views, view_counts, dim=0)
        pooled_fruits: List[torch.Tensor] = []
        attn_weights_list: List[torch.Tensor] = []
        for f_views in bag_chunks:
            if f_views.size(0) == 0:
                pooled_fruits.append(torch.zeros(self.embed_dim, device=f_all_views.device, dtype=f_all_views.dtype))
                attn_weights_list.append(torch.zeros(0, device=f_all_views.device, dtype=f_all_views.dtype))
            else:
                F_rgb, attn_w = self.gated_attention(f_views)
                pooled_fruits.append(F_rgb)
                attn_weights_list.append(attn_w)
        F_rgb_batch = torch.stack(pooled_fruits, dim=0)  # [B, embed_dim]
        return F_rgb_batch, attn_weights_list

    def forward(
        self,
        x: torch.Tensor,
        view_counts: Optional[List[int]] = None,
        return_attn: bool = False,
        tta: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Extract fruit-level representation F_rgb [B, 256] via Multi-View Gated Attention Pooling.

        Accepts:
        - 5D tensor `[B, N, C, H, W]`: batch of B multi-view fruit bags (zero-padded views are ignored).
        - 4D tensor `[Total_Views, C, H, W]` with `view_counts=[N_1, ..., N_B]`: flattened multi-view bags.
        - 4D tensor `[B, C, H, W]` with `view_counts=None`: B single-view samples (each treated as N=1 bag).
        - 3D tensor `[C, H, W]`: single image -> `[1, 256]`.

        `tta=True` applies the per-view 3-crop/flip TTA (`extract_view_embeddings_tta`) before pooling.
        """
        extract = self.extract_view_embeddings_tta if tta else self.extract_view_embeddings
        if x.dim() == 3:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 4 and view_counts is not None:
            f_all = extract(x)
            F_rgb_batch, attn_list = self.pool_fruit_bags(f_all, view_counts)
            self.last_attn_weights = [w.detach() for w in attn_list]
            return (F_rgb_batch, attn_list) if return_attn else F_rgb_batch
        elif x.dim() == 4:
            x = x.unsqueeze(1)  # [B, 1, C, H, W]
        elif x.dim() != 5:
            raise ValueError(f"Expected 3D, 4D, or 5D image tensor, got shape {tuple(x.shape)}")

        # x has shape [B, N, C, H, W]
        B, N, C, H, W = x.shape
        valid_mask = x.detach().flatten(2).abs().amax(dim=2) > 0  # [B, N]
        counts = valid_mask.sum(dim=1).tolist()
        total_valid = int(sum(counts))

        if total_valid == 0:
            # All views in batch are zero (e.g. missing RGB modality or zero-masking ablation)
            zero_out = x.new_zeros((B, self.embed_dim))
            attn_list = [x.new_zeros((0,)) for _ in range(B)]
            self.last_attn_weights = attn_list
            return (zero_out, attn_list) if return_attn else zero_out

        valid_imgs = x[valid_mask]  # [total_valid, C, H, W]
        f_valid = extract(valid_imgs)  # [total_valid, embed_dim]
        F_rgb_batch, attn_list = self.pool_fruit_bags(f_valid, counts)
        self.last_attn_weights = [w.detach() for w in attn_list]
        return (F_rgb_batch, attn_list) if return_attn else F_rgb_batch

    def load_unimodal(self, ckpt_path: Union[str, Path]) -> "VisionBackbone":
        """Load weights from a unimodal checkpoint (`convnextv2_multiview_backbone_256d.pth`,
        `best_convnextv2_multiview.pth`, or legacy `best_vision_unimodal.pth`).
        """
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Format 1: Dedicated 256-d Multi-View Extractor checkpoint (convnextv2_multiview_backbone_256d.pth)
        if isinstance(raw, dict) and "backbone_state_dict" in raw:
            bb_state = _strip_module_prefix(raw["backbone_state_dict"])
            missing, unexpected = self.backbone.load_state_dict(bb_state, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"Vision backbone mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
            if "projection_state_dict" in raw and isinstance(raw["projection_state_dict"], dict):
                proj_state = _strip_module_prefix(raw["projection_state_dict"])
                if proj_state["0.weight"].shape == self.projection[0].weight.shape:
                    self.projection.load_state_dict(proj_state, strict=True)
            if "gated_attention_state_dict" in raw and isinstance(raw["gated_attention_state_dict"], dict):
                attn_state = _strip_module_prefix(raw["gated_attention_state_dict"])
                if attn_state["V.weight"].shape == self.gated_attention.V.weight.shape:
                    self.gated_attention.load_state_dict(attn_state, strict=True)
            return self

        # Format 2: Full Multi-View model checkpoint (best_convnextv2_multiview.pth) or legacy single-view checkpoint
        state = _strip_module_prefix(_extract_state_dict(raw))

        has_backbone_prefix = any(k.startswith("backbone.") for k in state)
        has_proj_prefix = any(k.startswith("projection.") for k in state)
        has_attn_prefix = any(k.startswith("gated_attention.") for k in state)

        if has_backbone_prefix:
            bb_state = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
        else:
            bb_state = {
                k: v for k, v in state.items()
                if not k.startswith(("projection.", "gated_attention.", "classifier.", "head."))
            }
        missing, unexpected = self.backbone.load_state_dict(bb_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Vision checkpoint mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")

        if has_proj_prefix:
            proj_state = {k[len("projection."):]: v for k, v in state.items() if k.startswith("projection.")}
            if "0.weight" in proj_state and proj_state["0.weight"].shape == self.projection[0].weight.shape:
                self.projection.load_state_dict(proj_state, strict=True)

        if has_attn_prefix:
            attn_state = {k[len("gated_attention."):]: v for k, v in state.items() if k.startswith("gated_attention.")}
            if "V.weight" in attn_state and attn_state["V.weight"].shape == self.gated_attention.V.weight.shape:
                self.gated_attention.load_state_dict(attn_state, strict=True)

        return self
