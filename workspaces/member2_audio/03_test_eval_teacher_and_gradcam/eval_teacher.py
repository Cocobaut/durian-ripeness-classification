"""Module kiểm thử, đánh giá mô hình Multi-Modal Teacher và trực quan hóa giải thích (Grad-CAM / XAI).

Phụ trách: Member 2 (Nhánh Âm thanh & Đánh giá Đa phương thức / XAI)
Dự án: Nhận diện quả sầu riêng chín hay không bằng Chưng cất tri thức Đa phương thức (CMKD)
Tham chiếu đặc tả:
- Mục 1.3.2: Giai đoạn 1 - Huấn luyện Unimodal Baselines và Xây dựng Multi-Modal Teacher.
- Mục 1.4 (Mục tiêu 2 & 4): Thiết lập hệ thống Benchmark toàn diện (Unimodal vs. Multimodal)
  và Giải thích mô hình trực quan (Explainable AI - XAI) kèm Phân tích giới hạn vật lý.

Mô tả vai trò của file:
1. Đánh giá độc lập mô hình Teacher (hợp nhất 3 nhánh: RGB Image + Knock Sound + Sensor/Tabular)
   trên tập kiểm thử độc lập (Test Set) nhằm thiết lập "Mốc trần chất lượng" (Upper Bound).
2. Đánh giá tính đóng góp của từng phương thức (Modality Ablation / Zero-Masking):
   Kiểm tra sự suy giảm hiệu năng khi che/khuyết từng nhánh (w/o Audio, w/o Sensor, w/o RGB)
   để chứng minh giá trị của thông tin âm học và cảm biến đối với việc nhận diện độ chín.
3. Tổng hợp kết quả ở 2 cấp độ: Cấp độ mẫu đơn lẻ (Sample-level) và Cấp độ từng quả sầu riêng (Fruit-level).
4. Triển khai kỹ thuật Grad-CAM trên nhánh ảnh của Teacher để giải thích trực quan:
   Xem xét mạng tập trung vào vùng hình thái nào (rãnh gai, đĩa cuống, vỏ) dưới sự dẫn dắt
   của thông tin âm thanh gõ và cảm biến.
5. Phân tích lỗi chuyên sâu (Error Analysis): Đào sâu các ca dự đoán sai và đối chiếu với
   các thông số vật lý (độ cứng Firmness, hàm lượng đường TSS, đặc tính âm thanh gõ).

LƯU Ý DÀNH CHO THÀNH VIÊN PHỤ TRÁCH:
- File này đã được thiết kế sẵn khung kiến trúc hướng đối tượng (OOP) gồm các class, method signatures,
  docstrings và các bước gợi ý triển khai tuần tự.
- Thành viên phụ trách cần đọc kỹ các comment TODO trong từng hàm để tự hoàn thiện mã nguồn chi tiết.
- Các metric được gợi ý sẵn ở dưới; thành viên phụ trách CẦN TỰ TÌM THÊM các metric khác phù hợp
  với bài toán phân loại đa lớp / mất cân bằng dữ liệu và bài toán đánh giá bản đồ nhiệt Grad-CAM.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from torch.utils.data import DataLoader, Dataset

# Cấu hình logging phục vụ theo dõi quá trình đánh giá
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eval_teacher")

# ------------------------------------------------------------------------------
# 1. HẰNG SỐ CHUẨN TOÀN DỰ ÁN (PROJECT CONSTANTS)
# ------------------------------------------------------------------------------
CLASS_ORDER = ["Unripe", "Ripe", "Overripe"]  # 0: Chưa chín, 1: Chín, 2: Chín quá / Quá chín
LABEL_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASS_ORDER)}
IDX_TO_LABEL = {idx: cls_name for idx, cls_name in enumerate(CLASS_ORDER)}
NUM_CLASSES = len(CLASS_ORDER)


# ------------------------------------------------------------------------------
# 2. CẤU HÌNH ĐÁNH GIÁ TEACHER MODEL (DATACLASS CONFIG)
# ------------------------------------------------------------------------------
INPUT_MODES = ("multimodal", "audio", "vision", "sensor")
ABLATION_MODALITIES = ("audio", "sensor", "vision")


@dataclass
class TeacherEvaluatorConfig:
    """Cấu hình tham số cho quy trình đánh giá mô hình Multi-Modal Teacher và Grad-CAM.

    Thành viên phụ trách có thể hiệu chỉnh các đường dẫn và tham số cho phù hợp
    với môi trường chạy thực tế hoặc bổ sung thêm các cờ cấu hình mới.
    """
    # Đường dẫn file trọng số Teacher đã huấn luyện tốt nhất
    teacher_checkpoint_path: str = "checkpoints/teacher_best.pth"

    # Đường dẫn manifest dữ liệu sạch hoàn chỉnh (Master Manifest)
    manifest_path: str = "data/final/final_multimodal_manifest.csv"

    # Thư mục chứa tập kiểm thử chuẩn hóa (Test Set)
    test_data_dir: str = "data/final/test"

    # Thư mục lưu trữ kết quả đánh giá (bảng số liệu, ma trận nhầm lẫn, ảnh Grad-CAM)
    output_dir: str = "workspaces/member2_audio/03_test_eval_teacher_and_gradcam/results"

    # Cấu hình phần cứng và nạp dữ liệu
    batch_size: int = 16
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Tên tầng tích chập cuối cùng của nhánh Vision trong Teacher dùng để tính Grad-CAM
    # Gợi ý: ví dụ nếu backbone là ResNet50: 'vision_backbone.layer4'
    # hoặc EfficientNet: 'vision_backbone.features.7'
    gradcam_target_layer: str = "vision_backbone.layer4"

    # Danh sách các góc chụp của quả để kiểm tra độ ổn định theo góc
    camera_angles: List[str] = field(
        default_factory=lambda: ["TOP", "BOT", "SA", "SB", "SC", "SD"]
    )

    # Chiến lược tổng hợp kết quả nhiều góc/nhiều tiếng gõ về cấp độ quả ('mean_prob' hoặc 'majority_vote')
    fruit_aggregation_strategy: str = "mean_prob"

    # [BỔ SUNG] Chế độ đầu vào của mô hình cần đánh giá:
    # - "multimodal": batch = (img, audio, sensor, label, fruit_id[, extra]) và gọi model(images, audios, sensors)
    # - "audio" / "vision" / "sensor": batch = (x, label, fruit_id[, extra]) hoặc dict {x, label, fruit_id, ...}, gọi model(x)
    input_mode: str = "multimodal"

    # [BỔ SUNG] Tên tập dữ liệu đang chấm ("val", "test", ...) dùng làm tiền tố tên file kết quả
    split_name: str = "test"

    # [BỔ SUNG] Tên mô hình ghi vào bảng metrics (để ghép bảng so sánh giữa các nhánh)
    model_name: str = "teacher"

    # [BỔ SUNG] Số bin khi tính Expected Calibration Error
    ece_bins: int = 15


# ------------------------------------------------------------------------------
# 3. CLASS CHÍNH: ĐÁNH GIÁ MÔ HÌNH TEACHER (TEACHER EVALUATOR)
# ------------------------------------------------------------------------------
class TeacherEvaluator:
    """Bộ đánh giá toàn diện mô hình Hợp nhất Đa phương thức (Multi-Modal Teacher).

    Trách nhiệm chính của class:
    1. Tải mô hình Teacher đã huấn luyện và nạp trọng số tối ưu từ checkpoint.
    2. Chạy suy luận trên toàn bộ tập dữ liệu Test độc lập.
    3. Tính toán các chỉ số đo lường hiệu năng (Classification Metrics) ở 2 cấp độ:
       - Cấp độ mẫu đơn lẻ (Sample-level / Per-sample evaluation).
       - Cấp độ cá thể quả (Fruit-level aggregated evaluation).
    4. Đánh giá tính đóng góp của từng phương thức qua kỹ thuật Zero-masking / Modality Ablation:
       Đo lường mức độ sụt giảm hiệu năng khi mất nhánh Âm thanh, Cảm biến hoặc Thị giác.
    5. Xuất báo cáo chi tiết ra file CSV, Markdown và trực quan hóa ma trận nhầm lẫn.

    Ngoài Teacher, class cũng chấm được mô hình ĐƠN phương thức (vd nhánh âm thanh Tuần 2)
    bằng cách đặt `config.input_mode = "audio"` (bỏ qua bước ablation).
    """

    # Tên khóa được chấp nhận khi batch là dict
    _KEYS = {
        "image": ("image", "images", "img", "rgb"),
        "audio": ("audio", "audios", "mel", "audio_mel"),
        "sensor": ("sensor", "sensors", "tabular"),
        "x": ("x", "input", "inputs"),
        "label": ("label", "labels", "y", "target"),
        "fruit_id": ("fruit_id", "fruit_ids"),
    }

    def __init__(
        self,
        config: TeacherEvaluatorConfig,
        model: Optional[nn.Module] = None,
        test_loader: Optional[DataLoader] = None,
    ) -> None:
        """Khởi tạo bộ đánh giá Teacher.

        Args:
            config: Đối tượng cấu hình chứa đường dẫn và siêu tham số.
            model: Mô hình Multi-Modal Teacher đã khởi tạo (nếu None sẽ tự nạp từ checkpoint).
            test_loader: PyTorch DataLoader của tập kiểm thử Test.
        """
        if config.input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode phải thuộc {INPUT_MODES}, nhận được: {config.input_mode!r}")
        self.config = config
        self.device = torch.device(config.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = model
        if self.model is not None:
            self.model.to(self.device)
        self.test_loader = test_loader

        logger.info(f"Khởi tạo TeacherEvaluator trên thiết bị: {self.device} (input_mode={config.input_mode})")
        logger.info(f"Kết quả đánh giá sẽ được lưu tại: {self.output_dir}")

    # ------------------------------------------------------------------
    # Tiện ích nội bộ
    # ------------------------------------------------------------------
    def _file_prefix(self, split_name: Optional[str] = None) -> str:
        name = self.config.split_name if split_name is None else split_name
        return f"{name}_" if name else ""

    @staticmethod
    def _to_list(value: Any, n: int) -> List[Any]:
        """Chuyển tensor / ndarray / list / scalar thành list Python độ dài n."""
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            value = value.tolist() if value.ndim > 0 else [value.item()] * n
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, (str, int, float)) or value is None:
            value = [value] * n
        value = list(value)
        if len(value) != n:
            raise ValueError(f"Độ dài trường phụ ({len(value)}) khác batch size ({n}).")
        return value

    @classmethod
    def _pick(cls, batch: Dict[str, Any], name: str) -> Any:
        for key in cls._KEYS[name]:
            if key in batch:
                return batch[key]
        return None

    def _unpack_batch(self, batch: Any) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, Any, Dict[str, Any]]:
        """Tách batch thành (inputs, labels, fruit_ids, extras) theo `config.input_mode`."""
        multimodal = self.config.input_mode == "multimodal"
        extras: Dict[str, Any] = {}
        if isinstance(batch, dict):
            if multimodal:
                inputs = tuple(self._pick(batch, k) for k in ("image", "audio", "sensor"))
            else:
                x = self._pick(batch, "x")
                if x is None:
                    x = self._pick(batch, self.config.input_mode)
                inputs = (x,)
            labels, fruit_ids = self._pick(batch, "label"), self._pick(batch, "fruit_id")
            used = {k for names in self._KEYS.values() for k in names}
            extras = {k: v for k, v in batch.items() if k not in used}
        elif isinstance(batch, (list, tuple)):
            n_in = 3 if multimodal else 1
            if len(batch) < n_in + 2:
                raise ValueError(
                    f"Batch dạng tuple cần ít nhất {n_in + 2} phần tử cho input_mode={self.config.input_mode!r}, "
                    f"nhận được {len(batch)}."
                )
            inputs = tuple(batch[:n_in])
            labels, fruit_ids = batch[n_in], batch[n_in + 1]
            for i, extra in enumerate(batch[n_in + 2:]):
                if isinstance(extra, dict):
                    extras.update(extra)
                else:
                    extras[f"extra_{i}"] = extra
        else:
            raise TypeError(f"Không hỗ trợ kiểu batch: {type(batch)}")
        if labels is None or fruit_ids is None or any(t is None for t in inputs):
            raise ValueError("Batch thiếu input, label hoặc fruit_id.")
        return inputs, labels, fruit_ids, extras

    @staticmethod
    def _extract_logits(output: Any) -> torch.Tensor:
        """Lấy logits từ output của model: tensor, (logits, features) hoặc dict có khóa 'logits'."""
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (list, tuple)) and len(output) > 0:
            return output[0]
        if isinstance(output, dict) and "logits" in output:
            return output["logits"]
        raise TypeError(f"Không lấy được logits từ output kiểu {type(output)}")

    # ------------------------------------------------------------------
    # 3.1. Nạp mô hình
    # ------------------------------------------------------------------
    def load_teacher_model(self, checkpoint_path: Optional[str] = None) -> nn.Module:
        """Tải mô hình Teacher và load state_dict từ đường dẫn checkpoint.

        Hướng dẫn thực hiện cho Member 2 (TODO):
        ---------------------------------------------------------
        1. Import kiến trúc mô hình Multi-Modal Teacher từ module huấn luyện
           (ví dụ: `from workspaces.member1_vision_lead.04_train_teacher_and_student... import MultiModalTeacher`).
        2. Khởi tạo instance mô hình với cấu hình embedding đúng như lúc huấn luyện.
        3. Sử dụng `torch.load(path, map_location=self.device)` để lấy checkpoint dict.
        4. Trích xuất state_dict (xử lý trường hợp có prefix 'module.' nếu huấn luyện DataParallel).
        5. Gọi `model.load_state_dict(checkpoint['model_state_dict'])`.
        6. Chuyển mô hình sang chế độ đánh giá `model.eval()` và đóng băng gradient (`requires_grad = False`).

        Ghi chú triển khai: kiến trúc Teacher do Lead định nghĩa ở Tuần 3, nên hàm này KHÔNG tự đoán
        kiến trúc. Hãy truyền instance mô hình qua `TeacherEvaluator(config, model=...)`; hàm sẽ nạp
        trọng số vào instance đó. Checkpoint có thể là state_dict thuần hoặc dict có khóa 'model_state_dict'.

        Args:
            checkpoint_path: Đường dẫn tới file .pth (nếu None sẽ lấy từ self.config).

        Returns:
            nn.Module: Mô hình Teacher đã sẵn sàng suy luận.
        """
        ckpt_path = checkpoint_path or self.config.teacher_checkpoint_path
        logger.info(f"Bắt đầu load trọng số Teacher từ: {ckpt_path}")

        if self.model is None:
            raise RuntimeError(
                "Chưa có instance mô hình. Hãy khởi tạo kiến trúc (vd MultiModalTeacher của Lead) rồi truyền vào "
                "TeacherEvaluator(config, model=model) trước khi gọi load_teacher_model()."
            )
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state_dict = checkpoint[key]
                    break
        state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"State dict không khớp kiến trúc: thiếu {len(missing)} khóa (vd {missing[:3]}), "
                f"thừa {len(unexpected)} khóa (vd {unexpected[:3]})."
            )
        self.model.to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("Nạp trọng số thành công.")
        return self.model

    # ------------------------------------------------------------------
    # 3.2. Suy luận cấp độ sample
    # ------------------------------------------------------------------
    def evaluate_sample_level(
        self,
        test_loader: Optional[DataLoader] = None,
        mask_modality: Optional[str] = None,
    ) -> pd.DataFrame:
        """Chạy vòng lặp kiểm thử trên từng mẫu dữ liệu (sample-level evaluation).

        Trong bài toán này, mỗi quả sầu riêng có thể có 6 ảnh chụp các góc (TOP, BOT, SA...),
        nhiều file ghi âm tiếng gõ (Android, iPhone) và các thông số cảm biến. Mỗi tổ hợp nạp
        vào mạng là một sample.

        Batch được chấp nhận:
        - input_mode="multimodal": `(images, audios, sensors, labels, fruit_ids[, extra...])` hoặc dict.
        - input_mode đơn phương thức: `(x, labels, fruit_ids[, extra...])` hoặc dict `{x, label, fruit_id, ...}`.
        Các trường phụ (vd `device`, `camera_angle`, `file`) được đưa thành cột trong DataFrame.

        Args:
            test_loader: Dataloader tập kiểm thử.
            mask_modality: Tên nhánh cần zero-mask để thử nghiệm độ bền vững ('audio', 'sensor', 'vision' hoặc None).

        Returns:
            pd.DataFrame: Bảng kết quả dự đoán chi tiết từng sample, gồm các cột
            `['fruit_id', 'camera_angle', 'true_label', 'pred_label', 'prob_unripe', 'prob_ripe',
            'prob_overripe', 'is_correct', 'y_true', 'y_pred']` cùng các cột phụ.
        """
        logger.info(f"Bắt đầu đánh giá cấp độ sample (mask_modality={mask_modality})...")
        loader = test_loader or self.test_loader
        if loader is None:
            raise ValueError("Chưa có DataLoader để đánh giá.")
        if self.model is None:
            raise RuntimeError("Chưa có mô hình để đánh giá.")
        if mask_modality is not None and mask_modality not in ABLATION_MODALITIES:
            raise ValueError(f"mask_modality phải thuộc {ABLATION_MODALITIES} hoặc None.")

        multimodal = self.config.input_mode == "multimodal"
        self.model.eval()
        rows: List[Dict[str, Any]] = []
        with torch.no_grad():
            for batch in loader:
                inputs, labels, fruit_ids, extras = self._unpack_batch(batch)
                inputs = [t.to(self.device, non_blocking=True).float() for t in inputs]
                if mask_modality is not None:
                    if multimodal:
                        idx = {"vision": 0, "audio": 1, "sensor": 2}[mask_modality]
                        inputs[idx] = torch.zeros_like(inputs[idx])
                    elif mask_modality == self.config.input_mode:
                        inputs[0] = torch.zeros_like(inputs[0])

                logits = self._extract_logits(self.model(*inputs)).float()
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                preds = probs.argmax(axis=1)

                n = probs.shape[0]
                y_true = self._to_list(labels, n)
                fids = self._to_list(fruit_ids, n)
                extra_lists = {k: self._to_list(v, n) for k, v in extras.items()}
                for i in range(n):
                    row = {
                        "fruit_id": fids[i],
                        "camera_angle": extra_lists.get("camera_angle", [""] * n)[i],
                        "true_label": IDX_TO_LABEL[int(y_true[i])],
                        "pred_label": IDX_TO_LABEL[int(preds[i])],
                        "prob_unripe": float(probs[i, 0]),
                        "prob_ripe": float(probs[i, 1]),
                        "prob_overripe": float(probs[i, 2]),
                        "is_correct": bool(int(y_true[i]) == int(preds[i])),
                        "y_true": int(y_true[i]),
                        "y_pred": int(preds[i]),
                    }
                    for k, v in extra_lists.items():
                        if k != "camera_angle":
                            row[k] = v[i]
                    rows.append(row)

        sample_df = pd.DataFrame(rows)
        if sample_df.empty:
            raise ValueError("DataLoader không có mẫu nào.")
        logger.info(f"Đã suy luận {len(sample_df)} sample, accuracy = {sample_df['is_correct'].mean():.4f}")
        return sample_df

    # ------------------------------------------------------------------
    # 3.3. Tổng hợp cấp độ quả
    # ------------------------------------------------------------------
    def evaluate_fruit_level(
        self,
        sample_df: pd.DataFrame,
        strategy: Optional[str] = None,
    ) -> pd.DataFrame:
        """Tổng hợp kết quả từ cấp độ mẫu đơn lẻ (nhiều góc ảnh/nhiều tiếng gõ) về từng quả sầu riêng (fruit_id).

        Trong thực tế thu hoạch và thương phẩm, quyết định được đưa ra cho "từng quả sầu riêng"
        thay vì từng ảnh chụp đơn lẻ. Do đó, đánh giá cấp độ quả (Fruit-level) phản ánh độ chính
        xác thực tế của toàn hệ thống.

        Chiến lược:
        - 'mean_prob' (khuyến nghị): trung bình vector xác suất Softmax của các sample thuộc quả, chọn argmax.
        - 'majority_vote': nhãn xuất hiện nhiều nhất; hòa phiếu thì chọn nhãn có tổng xác suất cao hơn.

        Args:
            sample_df: Bảng kết quả cấp độ sample từ hàm `evaluate_sample_level()`.
            strategy: Chiến lược tổng hợp ('mean_prob' hoặc 'majority_vote').

        Returns:
            pd.DataFrame: Bảng dự đoán hợp nhất cho từng quả gồm
            `['fruit_id', 'true_label', 'pred_label', 'prob_unripe_mean', 'prob_ripe_mean',
            'prob_overripe_mean', 'is_correct', 'n_samples', 'y_true', 'y_pred']`.
        """
        agg_strategy = strategy or self.config.fruit_aggregation_strategy
        logger.info(f"Tổng hợp kết quả cấp độ quả (fruit_id) theo chiến lược: {agg_strategy}...")
        if agg_strategy not in ("mean_prob", "majority_vote"):
            raise ValueError("strategy phải là 'mean_prob' hoặc 'majority_vote'.")

        prob_cols = ["prob_unripe", "prob_ripe", "prob_overripe"]
        rows = []
        for fruit_id, group in sample_df.groupby("fruit_id", sort=True):
            true_labels = group["true_label"].unique()
            if len(true_labels) > 1:
                logger.warning(f"Quả {fruit_id} có nhiều nhãn thật {list(true_labels)}; dùng nhãn xuất hiện nhiều nhất.")
            true_label = group["true_label"].mode().iloc[0]
            mean_probs = group[prob_cols].mean().to_numpy()
            if agg_strategy == "mean_prob":
                pred_idx = int(mean_probs.argmax())
            else:
                votes = np.bincount(group["pred_label"].map(LABEL_TO_IDX).to_numpy(), minlength=NUM_CLASSES)
                candidates = np.flatnonzero(votes == votes.max())
                sums = group[prob_cols].sum().to_numpy()
                pred_idx = int(candidates[np.argmax(sums[candidates])])
            rows.append({
                "fruit_id": fruit_id,
                "true_label": true_label,
                "pred_label": IDX_TO_LABEL[pred_idx],
                "prob_unripe_mean": float(mean_probs[0]),
                "prob_ripe_mean": float(mean_probs[1]),
                "prob_overripe_mean": float(mean_probs[2]),
                "is_correct": bool(true_label == IDX_TO_LABEL[pred_idx]),
                "n_samples": int(len(group)),
                "y_true": LABEL_TO_IDX[true_label],
                "y_pred": pred_idx,
            })
        fruit_df = pd.DataFrame(rows)
        logger.info(f"Đã tổng hợp {len(fruit_df)} quả, accuracy cấp quả = {fruit_df['is_correct'].mean():.4f}")
        return fruit_df

    # ------------------------------------------------------------------
    # 3.4. Metrics
    # ------------------------------------------------------------------
    def compute_metrics(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        y_probs: Optional[np.ndarray] = None,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Tính toán toàn diện các chỉ số đánh giá phân loại (Classification Metrics).

        Metric được tính:
        - Accuracy, Balanced Accuracy, Macro-F1 (metric trọng tâm), Weighted-F1,
          Macro Precision / Macro Recall.
        - Precision / Recall / F1 / Support cho từng lớp (Unripe, Ripe, Overripe).
        - Cohen's Kappa, Matthews Correlation Coefficient (MCC).
        - Khi có xác suất: ROC-AUC macro one-vs-rest, PR-AUC macro (Average Precision),
          Log-loss đa lớp, Brier score đa lớp, Expected Calibration Error (ECE).
        Metric không tính được (vd thiếu lớp trong y_true) trả về NaN kèm cảnh báo, không gây lỗi.

        Args:
            y_true: Mảng nhãn thực tế (dạng số nguyên 0, 1, 2).
            y_pred: Mảng nhãn mô hình dự đoán.
            y_probs: Ma trận xác suất dự đoán kích thước (N, NUM_CLASSES).
            prefix: Tiền tố gắn vào tên metric (ví dụ: 'sample_' hoặc 'fruit_').

        Returns:
            Dict[str, float]: Từ điển chứa tên metric và giá trị tương ứng.
        """
        from sklearn.metrics import (
            accuracy_score, average_precision_score, balanced_accuracy_score, cohen_kappa_score,
            f1_score, log_loss, matthews_corrcoef, precision_recall_fscore_support, roc_auc_score,
        )
        import warnings

        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        labels = list(range(NUM_CLASSES))
        metrics: Dict[str, float] = {"n_samples": float(len(y_true))}

        def safe(name: str, fn: Callable[[], float]) -> None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    metrics[name] = float(fn())
            except Exception as exc:  # metric không xác định trên dữ liệu này
                logger.warning(f"Không tính được {prefix}{name}: {exc}")
                metrics[name] = float("nan")

        safe("accuracy", lambda: accuracy_score(y_true, y_pred))
        safe("balanced_accuracy", lambda: balanced_accuracy_score(y_true, y_pred))
        safe("macro_f1", lambda: f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        safe("weighted_f1", lambda: f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        metrics["macro_precision"] = float(np.mean(p))
        metrics["macro_recall"] = float(np.mean(r))
        for i, cls in enumerate(CLASS_ORDER):
            key = cls.lower()
            metrics[f"precision_{key}"] = float(p[i])
            metrics[f"recall_{key}"] = float(r[i])
            metrics[f"f1_{key}"] = float(f[i])
            metrics[f"support_{key}"] = float(s[i])
        safe("cohen_kappa", lambda: cohen_kappa_score(y_true, y_pred, labels=labels))
        safe("mcc", lambda: matthews_corrcoef(y_true, y_pred))

        if y_probs is not None:
            probs = np.asarray(y_probs, dtype=float)
            probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
            onehot = np.eye(NUM_CLASSES)[y_true]
            present = [c for c in labels if onehot[:, c].any() and not onehot[:, c].all()]

            def roc_auc() -> float:
                if len(present) < NUM_CLASSES:
                    raise ValueError(f"y_true chỉ có lớp {sorted(set(y_true.tolist()))}, ROC-AUC macro không xác định")
                return roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=labels)

            def pr_auc() -> float:
                if not present:
                    raise ValueError("không có lớp nào có cả mẫu dương và âm")
                return float(np.mean([average_precision_score(onehot[:, c], probs[:, c]) for c in present]))

            safe("roc_auc_ovr_macro", roc_auc)
            safe("pr_auc_macro", pr_auc)
            safe("log_loss", lambda: log_loss(y_true, np.clip(probs, 1e-12, 1.0), labels=labels))
            metrics["brier_score"] = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
            metrics["ece"] = self._expected_calibration_error(y_true, probs, self.config.ece_bins)

        return {f"{prefix}{k}": v for k, v in metrics.items()}

    @staticmethod
    def _expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
        """ECE theo độ tin cậy lớn nhất (top-label), chia đều [0, 1] thành n_bins."""
        conf = probs.max(axis=1)
        correct = (probs.argmax(axis=1) == y_true).astype(float)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
            if mask.any():
                ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
        return float(ece)

    def _level_metrics(self, df: pd.DataFrame, level: str) -> Dict[str, float]:
        prob_cols = (["prob_unripe", "prob_ripe", "prob_overripe"] if level == "sample"
                     else ["prob_unripe_mean", "prob_ripe_mean", "prob_overripe_mean"])
        return self.compute_metrics(df["y_true"], df["y_pred"], df[prob_cols].to_numpy(), prefix=f"{level}_")

    # ------------------------------------------------------------------
    # 3.5. Ablation
    # ------------------------------------------------------------------
    def evaluate_modality_ablation(
        self,
        test_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Đánh giá đóng góp của từng phương thức cảm biến (Modality Contribution & Ablation Study).

        Kỹ thuật này kiểm tra độ mạnh mẽ và sự phụ thuộc của Teacher vào từng giác quan:
        1. Full Multi-Modal (Đầy đủ: RGB + Audio + Sensor): Hiệu năng chuẩn tối ưu.
        2. w/o Audio (Zero-masking nhánh âm thanh): Đo lường âm thanh gõ đóng góp bao nhiêu % F1.
        3. w/o Sensor (Zero-masking nhánh cảm biến/bảng số liệu/nhiệt/đa phổ): Đo lường độ đóng góp của cảm biến.
        4. w/o Vision (Zero-masking nhánh ảnh RGB): Kiểm tra nếu "mù" thị giác thì Teacher phân loại ra sao.

        Chỉ chạy khi `config.input_mode == "multimodal"`; với mô hình đơn phương thức trả về dict rỗng.
        Kết quả được lưu ra `<split>_ablation.csv`.

        Returns:
            Dict[str, Dict[str, float]]: Bảng đối chiếu hiệu năng giữa các cấu hình phương thức.
        """
        logger.info("Bắt đầu phân tích đóng góp phương thức (Modality Ablation Study)...")
        if self.config.input_mode != "multimodal":
            logger.info(f"Bỏ qua ablation vì mô hình đơn phương thức (input_mode={self.config.input_mode}).")
            return {}

        settings = {"full": None, "wo_audio": "audio", "wo_sensor": "sensor", "wo_vision": "vision"}
        results: Dict[str, Dict[str, float]] = {}
        for name, mask in settings.items():
            sample_df = self.evaluate_sample_level(test_loader, mask_modality=mask)
            fruit_df = self.evaluate_fruit_level(sample_df)
            results[name] = {**self._level_metrics(sample_df, "sample"), **self._level_metrics(fruit_df, "fruit")}

        full = results["full"]
        for name, res in results.items():
            for level in ("sample", "fruit"):
                res[f"delta_{level}_macro_f1"] = full[f"{level}_macro_f1"] - res[f"{level}_macro_f1"]

        table = pd.DataFrame(results).T
        table.index.name = "setting"
        path = self.output_dir / f"{self._file_prefix()}ablation.csv"
        table.to_csv(path)
        logger.info(f"Đã lưu bảng ablation: {path}")
        return results

    # ------------------------------------------------------------------
    # 3.6. Confusion matrix
    # ------------------------------------------------------------------
    def plot_confusion_matrix(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        title: str = "Teacher Confusion Matrix",
        save_name: str = "teacher_confusion_matrix.png",
        normalize: bool = True,
    ) -> str:
        """Vẽ và lưu trữ biểu đồ nhiệt ma trận nhầm lẫn (Confusion Matrix Heatmap).

        - `normalize=True`: mỗi ô hiển thị tỷ lệ theo hàng (Recall từng lớp) kèm số lượng mẫu.
        - `normalize=False`: hiển thị số lượng mẫu.
        Trục X: nhãn dự đoán, trục Y: nhãn thật, theo thứ tự `CLASS_ORDER`.

        Returns:
            str: Đường dẫn file ảnh đã lưu.
        """
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int),
                              labels=list(range(NUM_CLASSES)))
        if normalize:
            row_sum = cm.sum(axis=1, keepdims=True)
            values = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum > 0)
            annot = np.array([[f"{values[i, j] * 100:.1f}%\n({cm[i, j]})" for j in range(NUM_CLASSES)]
                              for i in range(NUM_CLASSES)])
            fmt, vmax = "", 1.0
        else:
            values, annot, fmt, vmax = cm, True, "d", None

        # Dùng Figure + canvas Agg (chỉ lưu PNG): không phụ thuộc backend Tk, không đổi backend toàn cục
        fig = Figure(figsize=(6, 5))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        sns.heatmap(values, annot=annot, fmt=fmt, cmap="Blues", vmin=0, vmax=vmax, cbar=normalize,
                    xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=ax, annot_kws={"size": 12})
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(title, fontweight="bold")
        fig.tight_layout()
        path = self.output_dir / save_name
        fig.savefig(path, dpi=150)
        return str(path)

    # ------------------------------------------------------------------
    # 3.7. Pipeline
    # ------------------------------------------------------------------
    def run_full_evaluation_pipeline(
        self,
        test_loader: Optional[DataLoader] = None,
        split_name: Optional[str] = None,
        load_checkpoint: bool = False,
    ) -> Dict[str, Any]:
        """Quy trình đánh giá toàn diện mô hình Teacher (End-to-End Evaluation Pipeline).

        Hàm điều phối chạy trọn vẹn toàn bộ các bước:
        1. Load checkpoint Teacher model (khi `load_checkpoint=True`, hoặc khi chưa có model).
        2. Đánh giá cấp độ sample và cấp độ quả (Full Multi-Modal).
        3. Tính toán metrics tổng hợp (Sample-level & Fruit-level).
        4. Chạy Ablation Study đo lường đóng góp của từng kênh (Audio / Sensor / Vision) — chỉ multimodal.
        5. Xuất báo cáo ma trận nhầm lẫn dạng biểu đồ.
        6. Lưu trữ tệp tổng hợp kết quả (JSON / CSV).

        File đầu ra trong `output_dir` (tiền tố `<split>_`):
        `sample_predictions.csv`, `fruit_predictions.csv`, `metrics.csv` (dạng dài: model, split, level,
        metric, value), `metrics.json`, `metrics_by_<cột>.csv` (vd theo thiết bị), `ablation.csv`,
        `cm_sample_counts.png`, `cm_sample_normalized.png`, `cm_fruit_counts.png`, `cm_fruit_normalized.png`, `eval.log`.

        Args:
            test_loader: DataLoader cần chấm (mặc định `self.test_loader`).
            split_name: Tên tập ("val", "test"...) làm tiền tố file (mặc định `config.split_name`).
            load_checkpoint: True để nạp trọng số từ `config.teacher_checkpoint_path` trước khi chấm.

        Returns:
            Dict[str, Any]: Từ điển tổng hợp toàn bộ kết quả thực nghiệm.
        """
        if split_name is not None:
            self.config.split_name = split_name
        prefix = self._file_prefix()
        log_handler = logging.FileHandler(self.output_dir / f"{prefix}eval.log", mode="w", encoding="utf-8")
        log_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(log_handler)
        try:
            logger.info("=== BẮT ĐẦU QUY TRÌNH ĐÁNH GIÁ TOÀN DIỆN TEACHER MODEL ===")
            logger.info(f"model={self.config.model_name} | split={self.config.split_name} | "
                        f"input_mode={self.config.input_mode} | strategy={self.config.fruit_aggregation_strategy}")
            if self.model is None or load_checkpoint:
                self.load_teacher_model()

            sample_df = self.evaluate_sample_level(test_loader)
            fruit_df = self.evaluate_fruit_level(sample_df)
            sample_df.to_csv(self.output_dir / f"{prefix}sample_predictions.csv", index=False)
            fruit_df.to_csv(self.output_dir / f"{prefix}fruit_predictions.csv", index=False)

            metrics = {**self._level_metrics(sample_df, "sample"), **self._level_metrics(fruit_df, "fruit")}
            long_rows = []
            for key, value in metrics.items():
                level, metric = key.split("_", 1)
                long_rows.append({"model": self.config.model_name, "split": self.config.split_name,
                                  "level": level, "metric": metric, "value": value})
            pd.DataFrame(long_rows).to_csv(self.output_dir / f"{prefix}metrics.csv", index=False)

            # Metric tách theo cột phụ (vd thiết bị ghi âm, góc chụp)
            by_group: Dict[str, Dict[str, Dict[str, float]]] = {}
            for col in ("device", "camera_angle"):
                if col in sample_df and sample_df[col].astype(str).str.len().gt(0).any() \
                        and sample_df[col].nunique() > 1:
                    group_res = {str(g): self._level_metrics(gdf, "sample")
                                 for g, gdf in sample_df.groupby(col)}
                    by_group[col] = group_res
                    pd.DataFrame(group_res).T.rename_axis(col).to_csv(
                        self.output_dir / f"{prefix}metrics_by_{col}.csv")

            figures = {
                "cm_sample_counts": self.plot_confusion_matrix(
                    sample_df["y_true"], sample_df["y_pred"], f"{self.config.model_name} | {self.config.split_name} | sample",
                    f"{prefix}cm_sample_counts.png", normalize=False),
                "cm_sample_normalized": self.plot_confusion_matrix(
                    sample_df["y_true"], sample_df["y_pred"], f"{self.config.model_name} | {self.config.split_name} | sample",
                    f"{prefix}cm_sample_normalized.png", normalize=True),
                "cm_fruit_counts": self.plot_confusion_matrix(
                    fruit_df["y_true"], fruit_df["y_pred"], f"{self.config.model_name} | {self.config.split_name} | fruit",
                    f"{prefix}cm_fruit_counts.png", normalize=False),
                "cm_fruit_normalized": self.plot_confusion_matrix(
                    fruit_df["y_true"], fruit_df["y_pred"], f"{self.config.model_name} | {self.config.split_name} | fruit",
                    f"{prefix}cm_fruit_normalized.png", normalize=True),
            }

            ablation = self.evaluate_modality_ablation(test_loader)

            results = {
                "model": self.config.model_name,
                "split": self.config.split_name,
                "input_mode": self.config.input_mode,
                "fruit_aggregation_strategy": self.config.fruit_aggregation_strategy,
                "metrics": metrics,
                "metrics_by_group": by_group,
                "ablation": ablation,
                "figures": figures,
                "n_samples": int(len(sample_df)),
                "n_fruits": int(len(fruit_df)),
            }
            with open(self.output_dir / f"{prefix}metrics.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=float)
            logger.info(f"[{self.config.split_name}] sample Macro-F1 = {metrics['sample_macro_f1']:.4f} | "
                        f"fruit Macro-F1 = {metrics['fruit_macro_f1']:.4f}")
            logger.info("=== HOÀN TẤT ĐÁNH GIÁ ===")
            return results
        finally:
            logger.removeHandler(log_handler)
            log_handler.close()


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    output_dir: Union[str, Path],
    input_mode: str = "audio",
    split_name: str = "test",
    device: Optional[str] = None,
    model_name: str = "model",
    fruit_aggregation_strategy: str = "mean_prob",
) -> Dict[str, Any]:
    """Hàm tiện ích: chấm một mô hình trên một DataLoader và lưu toàn bộ kết quả vào `output_dir`.

    Ví dụ (script train nhánh âm thanh Tuần 2):
        from eval_teacher import evaluate
        results = evaluate(model, val_loader, "results", input_mode="audio", split_name="val",
                           model_name="audio_resnet18")
    """
    config = TeacherEvaluatorConfig(
        output_dir=str(output_dir),
        input_mode=input_mode,
        split_name=split_name,
        model_name=model_name,
        fruit_aggregation_strategy=fruit_aggregation_strategy,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    evaluator = TeacherEvaluator(config, model=model, test_loader=dataloader)
    return evaluator.run_full_evaluation_pipeline()


# ------------------------------------------------------------------------------
# 4. CLASS XAI: TRỰC QUAN HÓA GRAD-CAM TRÊN TEACHER (TEACHER GRAD-CAM VISUALIZER)
# ------------------------------------------------------------------------------
class TeacherGradCAMVisualizer:
    """Bộ trích xuất và trực quan hóa bản đồ kích hoạt nhiệt (Grad-CAM Heatmaps) cho Teacher Model.

    Mục tiêu khoa học (Tham chiếu Mục 1.4 đặc tả đề tài):
    - Dưới sự định hướng của tín hiệu âm thanh và cảm biến vật lý trong mạng Teacher,
      các bộ lọc không gian của nhánh Vision tập trung vào những vùng hình thái nào trên vỏ sầu riêng?
    - Trực quan hóa để đối chiếu: Liệu mạng có thật sự nhìn vào rãnh gai (groove expansion),
      vùng đĩa cuống (pedicel disc), hay bị phân tâm bởi phông nền và điều kiện ánh sáng?

    Trách nhiệm chính của class:
    1. Đăng ký Hook (Forward Hook & Backward Hook) vào tầng tích chập mục tiêu (Target Convolutional Layer).
    2. Lan truyền xuôi (Forward pass) để lưu trữ Feature Maps của ảnh.
    3. Lan truyền ngược (Backward pass) với gradient của lớp quan tâm để tính toán trọng số quan trọng $\alpha_k$.
    4. Tổng hợp bản đồ kích hoạt tuyến tính có trọng số, áp dụng hàm ReLU:
       $$L_{Grad-CAM}^c = ReLU\\left(\\sum_k \\alpha_k^c A^k\\right)$$
    5. Resize bản đồ nhiệt về kích thước ảnh gốc, nội suy màu (Colormap) và chập (Overlay) lên ảnh RGB.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """Khởi tạo Grad-CAM Visualizer cho Teacher.

        Args:
            model: Mô hình Multi-Modal Teacher.
            target_layer_name: Tên phân cấp của tầng tích chập cuối cùng (ví dụ: 'vision_backbone.layer4').
            device: Thiết bị tính toán ('cuda' hoặc 'cpu').
        """
        self.model = model
        self.target_layer_name = target_layer_name
        self.device = torch.device(device)

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hooks: List[Any] = []

        # Đăng ký hook vào tầng mục tiêu
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Tìm tầng layer theo tên và đăng ký forward_hook cùng backward_hook.

        Hướng dẫn thực hiện cho Member 2 (TODO):
        ---------------------------------------------------------
        1. Duyệt qua `dict(self.model.named_modules())` để tìm sub-module trùng tên `self.target_layer_name`.
        2. Đăng ký Forward Hook để trích xuất `self.activations`:
           `def forward_hook(module, input, output): self.activations = output.detach()`
        3. Đăng ký Backward Hook để trích xuất `self.gradients`:
           `def backward_hook(module, grad_in, grad_out): self.gradients = grad_out[0].detach()`
        4. Lưu các handle hook vào `self._hooks` để có thể gỡ bỏ (remove) khi không sử dụng.
        """
        # TODO [Member 2]: Hiện thực hóa việc đăng ký forward và backward hooks
        pass

    def generate_cam(
        self,
        image_tensor: torch.Tensor,
        audio_tensor: torch.Tensor,
        sensor_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Tạo bản đồ nhiệt Grad-CAM chuẩn hóa cho một mẫu đầu vào đa phương thức.

        Hướng dẫn thực hiện cho Member 2 (TODO):
        ---------------------------------------------------------
        1. Đảm bảo tensors có shape batch: `(1, C, H, W)` cho ảnh, `(1, 1, n_mels, T)` cho audio, `(1, D)` cho sensor.
        2. Bật chế độ gradient cho ảnh/tính toán nếu cần: `self.model.zero_grad()`.
        3. Forward pass qua Teacher: `logits, _ = self.model(image_tensor, audio_tensor, sensor_tensor)`.
        4. Nếu `target_class is None`: lấy lớp có logit cao nhất `target_class = logits.argmax(dim=-1).item()`.
        5. Tính điểm số mục tiêu: `score = logits[0, target_class]`.
        6. Backward pass để kích hoạt backward hook: `score.backward()`.
        7. Tính trọng số $\\alpha_k = \\frac{1}{Z} \\sum_i \\sum_j \\frac{\\partial Y^c}{\\partial A_{i,j}^k}$
           (Global Average Pooling trên gradient).
        8. Tính tổ hợp tuyến tính các feature maps: $CAM = \\sum_k \\alpha_k A^k$.
        9. Áp dụng ReLU để chỉ giữ lại các đặc trưng có đóng góp tích cực đến lớp mục tiêu: $CAM = ReLU(CAM)$.
        10. Chuẩn hóa CAM về khoảng $[0, 1]$ qua phép biến đổi Min-Max.
        11. Resize CAM về kích thước không gian ban đầu của ảnh `(H, W)`.

        Args:
            image_tensor: Tensor ảnh RGB đầu vào.
            audio_tensor: Tensor biểu đồ Mel-Spectrogram âm thanh gõ.
            sensor_tensor: Tensor vector đặc trưng cảm biến tích hợp.
            target_class: Chỉ số lớp cần giải thích (0: Unripe, 1: Ripe, 2: Overripe). Nếu None, tự lấy lớp dự đoán.

        Returns:
            np.ndarray: Ma trận 2D kích thước (H, W) với giá trị trong đoạn [0, 1].
        """
        # TODO [Member 2]: Hiện thực hóa thuật toán Grad-CAM tại đây
        raise NotImplementedError(
            "TODO [Member 2]: Hãy hiện thực hóa hàm generate_cam() theo các bước trong docstring!"
        )

    def overlay_cam_on_image(
        self,
        rgb_image: np.ndarray,
        cam_mask: np.ndarray,
        alpha: float = 0.5,
        colormap: int = 2,  # cv2.COLORMAP_JET
    ) -> np.ndarray:
        """Chập bản đồ nhiệt Grad-CAM lên ảnh gốc để hiển thị trực quan.

        Hướng dẫn thực hiện cho Member 2 (TODO):
        ---------------------------------------------------------
        1. Đảm bảo `rgb_image` có giá trị trong đoạn [0, 255] kiểu `uint8`.
        2. Biến đổi `cam_mask` (giá trị 0..1) sang uint8: `cam_uint8 = np.uint8(255 * cam_mask)`.
        3. Dùng OpenCV để tô màu nhiệt: `heatmap = cv2.applyColorMap(cam_uint8, colormap)`.
        4. Trộn hai ảnh theo trọng số: `overlay = cv2.addWeighted(rgb_image, 1 - alpha, heatmap, alpha, 0)`.
        5. Trả về ảnh RGB đã overlay sẵn sàng để lưu hoặc vẽ bằng matplotlib.

        Args:
            rgb_image: Ảnh gốc RGB (H, W, 3) dạng uint8.
            cam_mask: Mặt nạ nhiệt Grad-CAM (H, W) trong khoảng [0, 1].
            alpha: Hệ số hòa trộn độ trong suốt của bản đồ nhiệt (0.0 đến 1.0).
            colormap: Mã màu OpenCV (mặc định COLORMAP_JET).

        Returns:
            np.ndarray: Ảnh RGB (H, W, 3) đã overlay bản đồ nhiệt.
        """
        # TODO [Member 2]: Hiện thực hóa hàm overlay_cam_on_image()
        raise NotImplementedError(
            "TODO [Member 2]: Hãy hoàn thiện hàm overlay_cam_on_image()!"
        )

    def export_gradcam_gallery(
        self,
        samples_list: List[Dict[str, Any]],
        save_dir: str,
        num_samples_per_class: int = 5,
    ) -> None:
        """Xuất bộ sưu tập ảnh Grad-CAM cho các quả đại diện (Chưa chín, Chín, Chín quá).

        GỢI Ý METRIC ĐÁNH GIÁ CHẤT LƯỢNG GRAD-CAM (Member 2 tham khảo):
        ---------------------------------------------------------
        1. Energy-based Pointing Game: Đo tỷ lệ phần trăm năng lượng kích hoạt nằm trọn bên trong
           vùng quả sầu riêng (loại bỏ kích hoạt sai lệch ngoài phông nền).
        2. Attention Dispersion / Concentration: Đo độ tập trung của vùng chú ý (vùng chú ý hội tụ sắc nét
           hay bị phân tán loãng khắp ảnh).
        3. Morphological Alignment Score: Đo lường mức độ tương quan giữa vùng chú ý nhiệt với các rãnh gai
           (sử dụng các bộ lọc phát hiện cạnh/rãnh như Sobel / Frangi filter).

        >>> LƯU Ý ĐẶC BIỆT DÀNH CHO MEMBER 2:
        >>> Các tiêu chí trên là gợi ý; thành viên phụ trách CẦN TỰ TÌM THÊM các metric khác
        >>> để định lượng độ tin cậy và giải thích của Grad-CAM một cách khoa học.

        Args:
            samples_list: Danh sách các mẫu kiểm thử kèm thông tin đường dẫn và nhãn.
            save_dir: Thư mục lưu ảnh trực quan hóa.
            num_samples_per_class: Số lượng mẫu xuất bản đồ nhiệt cho mỗi lớp độ chín.
        """
        # TODO [Member 2]: Duyệt qua các mẫu đại diện, gọi generate_cam, overlay và lưu thành ảnh so sánh
        # [Ảnh gốc | Grad-CAM Heatmap | Overlay Heatmap]
        pass


# ------------------------------------------------------------------------------
# 5. CLASS PHÂN TÍCH LỖI VÀ GIỚI HẠN VẬT LÝ (TEACHER ERROR ANALYZER)
# ------------------------------------------------------------------------------
class TeacherErrorAnalyzer:
    """Bộ phân tích chuyên sâu các trường hợp dự đoán sai lệch của mô hình Teacher.

    Tham chiếu Mục 1.4 (Đóng góp 4):
    Phân tích các ca lỗi (Error Analysis) nhằm tìm hiểu nguyên nhân sinh hóa/vật lý:
    - Tại sao quả chín nhưng mô hình đoán nhầm là chưa chín? (Ví dụ: vỏ quả dày bất thường,
      độ ngọt TSS đã đạt nhưng độ rỗng thịt quả chưa phát triển nên âm gõ còn đục).
    - Phân tích tương quan giữa các ca đoán sai với các đặc tính cảm biến
      (Firmness, TSS, Dry Matter, Khối lượng, Nhiệt độ chênh lệch delta_temp_fg_bg).
    """

    def __init__(self, output_dir: str) -> None:
        """Khởi tạo bộ phân tích lỗi.

        Args:
            output_dir: Thư mục lưu trữ biểu đồ và báo cáo lỗi.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_failure_cases(self, fruit_level_df: pd.DataFrame) -> pd.DataFrame:
        """Lọc và phân loại tất cả các ca dự đoán sai ở cấp độ quả.

        Hướng dẫn thực hiện cho Member 2 (TODO):
        ---------------------------------------------------------
        1. Lọc các dòng có `is_correct == False` (hoặc `true_label != pred_label`).
        2. Gắn nhãn loại lỗi chi tiết (Error Subtype):
           - 'Unripe_as_Ripe': Quả non bị đoán nhầm là chín (Lỗi nghiêm trọng: bán nhầm quả non).
           - 'Ripe_as_Unripe': Quả chín bị đoán nhầm là non (Bỏ sót thời điểm thu hoạch tối ưu).
           - 'Ripe_as_Overripe': Quả chín tới bị đoán là quá chín.
           - 'Overripe_as_Ripe': Quả quá chín/lên men bị đoán là chín ngon.
        3. Trả về DataFrame danh sách các ca lỗi kèm xác suất tin cậy của mô hình.
        """
        # TODO [Member 2]: Hiện thực hóa hàm lọc lỗi tại đây
        raise NotImplementedError(
            "TODO [Member 2]: Hãy hoàn thiện hàm extract_failure_cases()!"
        )

    def correlate_errors_with_physical_properties(
        self,
        failures_df: pd.DataFrame,
        manifest_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Phân tích tương quan giữa các ca lỗi và các đặc tính sinh hóa / cảm biến.

        Hướng dẫn thực hiện cho Member 2 (TODO):
        ---------------------------------------------------------
        1. Merge `failures_df` với `manifest_df` dựa trên `fruit_id`.
        2. Trích xuất các thuộc tính: `Firmness_AVE`, `TSS_AVE`, `Dry_Matter_AVE`,
           `Harvest_Weight`, `fg_mean_temp`, `delta_temp_fg_bg`.
        3. So sánh phân phối của nhóm đoán sai (Failures) với nhóm đoán đúng (Successes)
           thông qua biểu đồ Boxplot hoặc Violin plot.
        4. Kiểm tra giả thuyết vật lý: Các ca lỗi có nằm ở vùng ranh giới phân bố
           (boundary threshold) giữa các trạng thái độ chín hay không?
        """
        # TODO [Member 2]: Hiện thực hóa hàm phân tích tương quan lỗi
        raise NotImplementedError(
            "TODO [Member 2]: Hãy hiện thực hóa correlate_errors_with_physical_properties()!"
        )


# ------------------------------------------------------------------------------
# 6. HÀM MAIN / CLI THỰC THI THỬ NGHIỆM
# ------------------------------------------------------------------------------
def parse_args():
    """Hàm phân tích đối số dòng lệnh khi chạy script trực tiếp."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Đánh giá mô hình Multi-Modal Teacher & Trực quan hóa Grad-CAM"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/teacher_best.pth",
        help="Đường dẫn file trọng số Teacher model",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/final/final_multimodal_manifest.csv",
        help="Đường dẫn Master Manifest CSV",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="data/final/test",
        help="Đường dẫn thư mục tập Test",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="workspaces/member2_audio/03_test_eval_teacher_and_gradcam/results",
        help="Thư mục lưu báo cáo kết quả và hình ảnh",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Kích thước batch khi nạp dữ liệu",
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        default="multimodal",
        choices=["multimodal", "audio", "vision", "sensor"],
        help="Chế độ đầu vào của mô hình (multimodal cho Teacher, audio/vision/sensor cho mô hình đơn phương thức)",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="test",
        help="Tên tập đang chấm (val/test), dùng làm tiền tố tên file kết quả",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="teacher",
        help="Tên mô hình ghi vào bảng metrics",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Thiết bị tính toán (cuda hoặc cpu)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 80)
    print(" SCRIPT ĐÁNH GIÁ MULTI-MODAL TEACHER & TRỰC QUAN HÓA GRAD-CAM (XAI)")
    print(" Phụ trách: Member 2 (Audio & Multi-Modal Evaluation)")
    print("=" * 80)

    config = TeacherEvaluatorConfig(
        teacher_checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        test_data_dir=args.test_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        input_mode=args.input_mode,
        split_name=args.split_name,
        model_name=args.model_name,
    )

    evaluator = TeacherEvaluator(config=config)
    print(f"\n[INFO] Đã khởi tạo cấu hình đánh giá Teacher.")
    print(f"[INFO] File trọng số mục tiêu: {config.teacher_checkpoint_path}")
    print(f"[INFO] Kết quả sẽ được lưu tại: {config.output_dir}")
    print("\n[HƯỚNG DẪN THỰC THI]:")
    print("1. Phần đánh giá (TeacherEvaluator) đã hoàn thiện; Grad-CAM và phân tích lỗi sẽ làm ở Tuần 4.")
    print("2. Kiến trúc Teacher do Lead định nghĩa ở Tuần 3: khởi tạo model rồi dùng trong code Python:")
    print("     from eval_teacher import TeacherEvaluator, TeacherEvaluatorConfig")
    print("     ev = TeacherEvaluator(TeacherEvaluatorConfig(...), model=model, test_loader=loader)")
    print("     ev.run_full_evaluation_pipeline(load_checkpoint=True)")
    print("3. Mô hình đơn phương thức: from eval_teacher import evaluate; evaluate(model, loader, out_dir, input_mode='audio')")

