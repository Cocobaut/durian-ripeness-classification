"""Module tiền xử lý và làm giàu dữ liệu ảnh cho nhánh thị giác (Vision Modality).

Phụ trách: Member 1 (Vision Lead)

Module chỉ gồm một class duy nhất `VisionPreprocessor`, gom toàn bộ các bước đã được
kiểm chứng trong notebook `01_eda_and_preprocess_vision.ipynb`:
- Đọc ảnh 24MP tốc độ cao (giải mã JPEG ở độ phân giải thấp bằng chế độ draft).
- Resize giữ nguyên tỷ lệ khung hình 3:2 rồi mới crop về ảnh vuông.
- Cân bằng sáng cục bộ CLAHE trên kênh L (không gian màu LAB).
- Pipeline tăng cường dữ liệu cho tập train theo 3 mức: light, medium, strong.
- Pipeline chuẩn hóa cố định cho tập val/test.
- Mixup và CutMix ở cấp độ batch, sinh nhãn mềm phục vụ chưng cất tri thức.

Ví dụ sử dụng:
    preprocessor = VisionPreprocessor(image_size=224, aug_policy="medium")
    train_tf = preprocessor.get_train_transforms()
    val_tf = preprocessor.get_val_transforms()

    # Trong vòng lặp huấn luyện
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
    """Bộ tiền xử lý và làm giàu dữ liệu ảnh RGB quả sầu riêng.

    Class cung cấp:
    - `get_train_transforms()` / `get_val_transforms()`: pipeline torchvision dùng cho Dataset.
    - `load_image()`: đọc ảnh từ đường dẫn, numpy array hoặc PIL Image.
    - `preprocess_image()`: tiền xử lý một ảnh đơn lẻ thành tensor (C, H, W).
    - `apply_clahe()`: áp dụng riêng CLAHE để trực quan hóa.
    - `denormalize()`: đưa tensor đã chuẩn hóa về ảnh uint8 để hiển thị.
    - `mix_batch()`: Mixup / CutMix ở cấp độ batch, trả về nhãn mềm.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    # Cấu hình cho từng mức tăng cường dữ liệu.
    # color_jitter = (brightness, contrast, saturation, hue); None nghĩa là tắt.
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
        """Transform áp dụng CLAHE lên kênh L của không gian màu LAB.

        Chỉ can thiệp vào kênh độ sáng L nên làm nổi bật rãnh gai và đĩa cuống
        mà không làm lệch màu vỏ quả (kênh A, B được giữ nguyên).
        Được khai báo dạng class lồng nhau (thay vì lambda) để có thể pickle
        khi DataLoader dùng nhiều worker.
        """

        def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
            self.clip_limit = clip_limit
            self.tile_grid_size = tuple(tile_grid_size)

        def __call__(self, img: Union[Image.Image, np.ndarray]) -> Image.Image:
            """Nhận ảnh PIL hoặc numpy RGB, trả về ảnh PIL RGB đã cân bằng sáng."""
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
        """Khởi tạo bộ tiền xử lý.

        Args:
            image_size: Kích thước ảnh đầu ra, nhận int (ảnh vuông) hoặc tuple (H, W).
            use_clahe: Bật/tắt bước cân bằng sáng CLAHE.
            clahe_clip_limit: Ngưỡng giới hạn tương phản của CLAHE.
            clahe_tile_grid_size: Kích thước lưới chia ô của CLAHE.
            aug_policy: Mức tăng cường dữ liệu cho tập train: "light", "medium" hoặc "strong".
            resize_ratio: Tỷ lệ giữa cạnh ngắn sau resize và kích thước crop (mặc định 1.14 ~ 256/224).
            mean: Trung bình chuẩn hóa theo kênh RGB, mặc định dùng thống kê ImageNet.
            std: Độ lệch chuẩn chuẩn hóa theo kênh RGB, mặc định dùng thống kê ImageNet.
            num_classes: Số lớp độ chín, dùng khi tạo nhãn mềm.
            mixup_alpha: Tham số phân phối Beta của Mixup (0 để tắt Mixup).
            cutmix_alpha: Tham số phân phối Beta của CutMix (0 để tắt CutMix).
            mix_prob: Xác suất áp dụng Mixup/CutMix cho mỗi batch.
        """
        policy = aug_policy.lower()
        if policy not in self.POLICY_CONFIGS:
            raise ValueError(f"aug_policy phải thuộc {list(self.POLICY_CONFIGS)}, nhận được: '{aug_policy}'")

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
        """Các bước đầu chung cho cả train và val: resize giữ tỷ lệ, sau đó CLAHE.

        Resize theo cạnh ngắn (truyền vào một số nguyên) nên ảnh 6000x4000 trở thành
        khoảng 382x256, không bị bóp méo. CLAHE đặt sau resize để chạy nhanh trên ảnh nhỏ.
        """
        head = [transforms.Resize(self.resize_size, interpolation=transforms.InterpolationMode.BILINEAR)]
        if self.use_clahe:
            head.append(self.clahe)
        return head

    def _build_normalize_tail(self) -> list:
        """Các bước cuối chung: chuyển sang tensor và chuẩn hóa theo mean/std."""
        return [transforms.ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)]

    def _build_train_transforms(self) -> transforms.Compose:
        """Xây dựng pipeline tăng cường dữ liệu cho tập train theo `aug_policy`."""
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
        """Xây dựng pipeline cố định (không ngẫu nhiên) cho tập val/test: resize giữ tỷ lệ rồi center crop."""
        steps = self._build_common_head()
        steps.append(transforms.CenterCrop(self.image_size))
        steps += self._build_normalize_tail()
        return transforms.Compose(steps)

    def get_train_transforms(self) -> transforms.Compose:
        """Trả về pipeline transforms cho tập train."""
        return self.train_transforms

    def get_val_transforms(self) -> transforms.Compose:
        """Trả về pipeline transforms cho tập val/test."""
        return self.val_transforms

    # ------------------------------------------------------------------
    # Xử lý ảnh đơn lẻ
    # ------------------------------------------------------------------
    def load_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        """Đọc ảnh và trả về PIL Image RGB.

        Với file JPEG, dùng `Image.draft` để giải mã trực tiếp ở độ phân giải thấp hơn
        (vẫn lớn hơn kích thước resize), giúp đọc ảnh 24MP nhanh hơn nhiều lần.
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_input}")
            img = Image.open(image_input)
            draft_size = self.resize_size * 2
            img.draft("RGB", (draft_size, draft_size))
            return img.convert("RGB")
        if isinstance(image_input, np.ndarray):
            return Image.fromarray(image_input).convert("RGB")
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        raise TypeError(f"Định dạng ảnh không được hỗ trợ: {type(image_input)}")

    def preprocess_image(
        self,
        image_input: Union[str, Image.Image, np.ndarray],
        is_train: bool = False,
    ) -> torch.Tensor:
        """Tiền xử lý một ảnh đơn lẻ, trả về tensor (C, H, W) đã chuẩn hóa."""
        pipeline = self.train_transforms if is_train else self.val_transforms
        return pipeline(self.load_image(image_input))

    def apply_clahe(self, img: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Áp dụng riêng bước CLAHE lên ảnh, dùng cho mục đích trực quan hóa."""
        return self.clahe(img)

    def denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """Chuyển tensor đã chuẩn hóa (C, H, W) hoặc (1, C, H, W) về ảnh numpy RGB uint8."""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        img = (tensor.detach().cpu() * std + mean).clamp(0.0, 1.0)
        return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Mixup / CutMix cấp độ batch
    # ------------------------------------------------------------------
    def mix_batch(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Áp dụng Mixup hoặc CutMix (chọn ngẫu nhiên 50/50) lên một batch.

        Args:
            images: Tensor ảnh (B, C, H, W) đã qua train transforms.
            labels: Nhãn dạng chỉ số (B,) hoặc nhãn mềm (B, num_classes), ví dụ xác suất từ teacher.

        Returns:
            Tuple (ảnh đã trộn, nhãn mềm kích thước (B, num_classes)).
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
            # Tính lại lambda theo diện tích thực tế sau khi bbox bị cắt ở biên ảnh
            lam = 1.0 - (y2 - y1) * (x2 - x1) / (images.shape[2] * images.shape[3])
        else:
            lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            images = lam * images + (1.0 - lam) * images[perm]

        return images, lam * targets + (1.0 - lam) * targets[perm]

    def _to_soft_targets(self, labels: torch.Tensor) -> torch.Tensor:
        """Chuyển nhãn chỉ số sang one-hot; nếu đã là nhãn mềm thì giữ nguyên."""
        if labels.dim() == 2:
            return labels.float()
        return torch.nn.functional.one_hot(labels.long(), self.num_classes).float()

    @staticmethod
    def _random_bbox(height: int, width: int, lam: float) -> Tuple[int, int, int, int]:
        """Sinh hộp cắt ngẫu nhiên cho CutMix, diện tích xấp xỉ (1 - lam) diện tích ảnh.

        Returns:
            Tuple (y1, x1, y2, x2) đã được giới hạn trong biên ảnh.
        """
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h, cut_w = int(height * cut_ratio), int(width * cut_ratio)
        cy, cx = np.random.randint(height), np.random.randint(width)
        y1, y2 = np.clip([cy - cut_h // 2, cy + cut_h // 2], 0, height)
        x1, x2 = np.clip([cx - cut_w // 2, cx + cut_w // 2], 0, width)
        return int(y1), int(x1), int(y2), int(x2)
