"""Image preprocessing and augmentation module for the Vision modality.

Responsible: Member 1 (Vision Lead)

This module contains a single class `VisionPreprocessor`, consolidating all steps
validated in `01_eda_and_preprocess_vision.ipynb`:
- Fast 24MP image decoding (JPEG low-resolution decoding via draft mode).
- Aspect-ratio-preserving resize (3:2) prior to square cropping.
- Local contrast enhancement with CLAHE on the L channel (LAB color space).
- Training augmentation pipelines across 3 levels: light, medium, strong.
- Deterministic normalization pipeline for val/test splits.
- Batch-level Mixup and CutMix generating soft labels for knowledge distillation.

Usage example:
    preprocessor = VisionPreprocessor(image_size=224, aug_policy="medium")
    train_tf = preprocessor.get_train_transforms()
    val_tf = preprocessor.get_val_transforms()

    # Inside training loop
    images, soft_targets = preprocessor.mix_batch(images, labels)
"""

import os
from typing import Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


class VisionPreprocessor:
    """Preprocessor and augmentation pipeline for durian RGB images.

    Provides:
    - `get_train_transforms()` / `get_val_transforms()`: torchvision pipelines for Datasets.
    - `load_image()`: load an image from file path, numpy array, or PIL Image.
    - `preprocess_image()`: preprocess a single image into a (C, H, W) tensor.
    - `apply_clahe()`: apply CLAHE standalone for visualization.
    - `denormalize()`: convert normalized tensors back to uint8 images for display.
    - `mix_batch()`: batch-level Mixup / CutMix returning soft labels.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    # Configuration for each data augmentation level.
    # color_jitter = (brightness, contrast, saturation, hue); None disables it.
    POLICY_CONFIGS: Dict[str, Dict] = {
        "light": {
            "crop_scale": (0.85, 1.0),
            "rotation": 10,
            "color_jitter": None,
            "blur_prob": 0.0,
            "perspective_prob": 0.0,
            "erasing_prob": 0.0,
        },
        "medium": {
            "crop_scale": (0.8, 1.0),
            "rotation": 15,
            "color_jitter": (0.15, 0.15, 0.15, 0.03),
            "blur_prob": 0.2,
            "perspective_prob": 0.0,
            "erasing_prob": 0.2,
        },
        "strong": {
            "crop_scale": (0.7, 1.0),
            "rotation": 20,
            "color_jitter": (0.2, 0.2, 0.2, 0.04),
            "blur_prob": 0.3,
            "perspective_prob": 0.3,
            "erasing_prob": 0.3,
        },
    }

    class ClaheTransform:
        """Transform applying CLAHE on the L channel of LAB color space.

        Modifies only the lightness channel L to highlight spine grooves and pedicel discs
        without shifting shell color (channels A and B remain untouched).
        Defined as a nested class (instead of lambda) to remain picklable in multi-worker DataLoaders.
        """

        def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
            self.clip_limit = clip_limit
            self.tile_grid_size = tuple(tile_grid_size)

        def __call__(self, img: Union[Image.Image, np.ndarray]) -> Image.Image:
            """Accept a PIL or numpy RGB image and return a contrast-enhanced PIL RGB image."""
            rgb = np.asarray(img.convert("RGB")) if isinstance(img, Image.Image) else img
            l_channel, a_channel, b_channel = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB))
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            lab = cv2.merge((clahe.apply(l_channel), a_channel, b_channel))
            return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))

        def __repr__(self) -> str:
            return f"ClaheTransform(clip_limit={self.clip_limit}, tile_grid_size={self.tile_grid_size})"

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 224,
        use_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        aug_policy: str = "medium",
        resize_ratio: float = 1.14,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        num_classes: int = 3,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        mix_prob: float = 1.0,
    ):
        """Initialize the vision preprocessor.

        Args:
            image_size: Output image size, int (square) or tuple (H, W).
            use_clahe: Enable/disable CLAHE contrast enhancement.
            clahe_clip_limit: Contrast limiting threshold for CLAHE.
            clahe_tile_grid_size: Grid size for CLAHE tiles.
            aug_policy: Training augmentation level: "light", "medium", or "strong".
            resize_ratio: Ratio between resized short side and crop size (default 1.14 ~ 256/224).
            mean: RGB normalization mean; defaults to ImageNet statistics.
            std: RGB normalization std; defaults to ImageNet statistics.
            num_classes: Number of ripeness classes for soft targets.
            mixup_alpha: Beta distribution parameter for Mixup (0 disables Mixup).
            cutmix_alpha: Beta distribution parameter for CutMix (0 disables CutMix).
            mix_prob: Probability of applying Mixup/CutMix per batch.
        """
        policy = aug_policy.lower()
        if policy not in self.POLICY_CONFIGS:
            raise ValueError(f"aug_policy must be one of {list(self.POLICY_CONFIGS)}, got: '{aug_policy}'")

        self.image_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        self.resize_size = int(round(min(self.image_size) * resize_ratio))
        self.use_clahe = use_clahe
        self.aug_policy = policy
        self.mean = tuple(mean) if mean is not None else self.IMAGENET_MEAN
        self.std = tuple(std) if std is not None else self.IMAGENET_STD

        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mix_prob = mix_prob

        self.clahe = self.ClaheTransform(clahe_clip_limit, clahe_tile_grid_size)
        self.train_transforms = self._build_train_transforms()
        self.val_transforms = self._build_val_transforms()

    # ------------------------------------------------------------------
    # Pipeline transforms
    # ------------------------------------------------------------------
    def _build_common_head(self) -> list:
        """Shared initial steps for train and val: aspect-preserving resize, then CLAHE.

        Resizing by short side (passing a single integer) scales 6000x4000 images to
        roughly 382x256 without distortion. CLAHE runs after resize for speed on smaller images.
        """
        head = [transforms.Resize(self.resize_size, interpolation=transforms.InterpolationMode.BILINEAR)]
        if self.use_clahe:
            head.append(self.clahe)
        return head

    def _build_normalize_tail(self) -> list:
        """Shared final steps: convert to tensor and normalize by mean/std."""
        return [transforms.ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)]

    def _build_train_transforms(self) -> transforms.Compose:
        """Build training augmentation pipeline according to `aug_policy`."""
        cfg = self.POLICY_CONFIGS[self.aug_policy]
        steps = self._build_common_head()

        if cfg["perspective_prob"] > 0:
            steps.append(transforms.RandomPerspective(distortion_scale=0.15, p=cfg["perspective_prob"]))

        steps += [
            transforms.RandomResizedCrop(
                self.image_size,
                scale=cfg["crop_scale"],
                ratio=(0.85, 1.15),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=cfg["rotation"]),
        ]

        if cfg["color_jitter"] is not None:
            brightness, contrast, saturation, hue = cfg["color_jitter"]
            steps.append(transforms.ColorJitter(brightness, contrast, saturation, hue))

        if cfg["blur_prob"] > 0:
            steps.append(
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=cfg["blur_prob"])
            )

        steps += self._build_normalize_tail()

        if cfg["erasing_prob"] > 0:
            steps.append(
                transforms.RandomErasing(p=cfg["erasing_prob"], scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random")
            )

        return transforms.Compose(steps)

    def _build_val_transforms(self) -> transforms.Compose:
        """Build deterministic pipeline for val/test: aspect-preserving resize then center crop."""
        steps = self._build_common_head()
        steps.append(transforms.CenterCrop(self.image_size))
        steps += self._build_normalize_tail()
        return transforms.Compose(steps)

    def get_train_transforms(self) -> transforms.Compose:
        """Return transforms pipeline for the train split."""
        return self.train_transforms

    def get_val_transforms(self) -> transforms.Compose:
        """Return transforms pipeline for the val/test splits."""
        return self.val_transforms

    # ------------------------------------------------------------------
    # Single-image processing
    # ------------------------------------------------------------------
    def load_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        """Load image and return an RGB PIL Image.

        For JPEG files, uses `Image.draft` to decode directly at a lower resolution
        (still larger than the resize size), speeding up 24MP image reads significantly.
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            img = Image.open(image_input)
            draft_size = self.resize_size * 2
            img.draft("RGB", (draft_size, draft_size))
            return img.convert("RGB")
        if isinstance(image_input, np.ndarray):
            return Image.fromarray(image_input).convert("RGB")
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def preprocess_image(
        self,
        image_input: Union[str, Image.Image, np.ndarray],
        is_train: bool = False,
    ) -> torch.Tensor:
        """Preprocess a single image, returning a normalized (C, H, W) tensor."""
        pipeline = self.train_transforms if is_train else self.val_transforms
        return pipeline(self.load_image(image_input))

    def apply_clahe(self, img: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Apply CLAHE standalone to an image for visualization purposes."""
        return self.clahe(img)

    def denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert normalized tensor (C, H, W) or (1, C, H, W) back to uint8 RGB numpy image."""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        img = (tensor.detach().cpu() * std + mean).clamp(0.0, 1.0)
        return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Batch-level Mixup / CutMix
    # ------------------------------------------------------------------
    def mix_batch(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Mixup or CutMix (50/50 random choice) to a batch.

        Args:
            images: Image tensor (B, C, H, W) after train transforms.
            labels: Class index labels (B,) or soft labels (B, num_classes), e.g., teacher probabilities.

        Returns:
            Tuple (mixed images, soft labels of shape (B, num_classes)).
        """
        targets = self._to_soft_targets(labels)
        mixup_on, cutmix_on = self.mixup_alpha > 0, self.cutmix_alpha > 0
        if (not mixup_on and not cutmix_on) or torch.rand(1).item() > self.mix_prob:
            return images, targets

        perm = torch.randperm(images.size(0), device=images.device)
        use_cutmix = cutmix_on and (not mixup_on or torch.rand(1).item() < 0.5)

        if use_cutmix:
            lam = float(np.random.beta(self.cutmix_alpha, self.cutmix_alpha))
            y1, x1, y2, x2 = self._random_bbox(images.shape[2], images.shape[3], lam)
            images = images.clone()
            images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
            # Recompute lambda from actual area after clipping bbox to image boundaries
            lam = 1.0 - (y2 - y1) * (x2 - x1) / (images.shape[2] * images.shape[3])
        else:
            lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            images = lam * images + (1.0 - lam) * images[perm]

        return images, lam * targets + (1.0 - lam) * targets[perm]

    def _to_soft_targets(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert class indices to one-hot; keep soft labels unchanged."""
        if labels.dim() == 2:
            return labels.float()
        return torch.nn.functional.one_hot(labels.long(), self.num_classes).float()

    @staticmethod
    def _random_bbox(height: int, width: int, lam: float) -> Tuple[int, int, int, int]:
        """Generate a random CutMix crop box with area approximately (1 - lam) of image area.

        Returns:
            Tuple (y1, x1, y2, x2) clipped to image bounds.
        """
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h, cut_w = int(height * cut_ratio), int(width * cut_ratio)
        cy, cx = np.random.randint(height), np.random.randint(width)
        y1, y2 = np.clip([cy - cut_h // 2, cy + cut_h // 2], 0, height)
        x1, x2 = np.clip([cx - cut_w // 2, cx + cut_w // 2], 0, width)
        return int(y1), int(x1), int(y2), int(x2)
