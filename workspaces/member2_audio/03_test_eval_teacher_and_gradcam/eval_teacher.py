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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """

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
        self.config = config
        self.device = torch.device(config.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = model
        self.test_loader = test_loader

        logger.info(f"Khởi tạo TeacherEvaluator trên thiết bị: {self.device}")
        logger.info(f"Kết quả đánh giá sẽ được lưu tại: {self.output_dir}")

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

        Args:
            checkpoint_path: Đường dẫn tới file .pth (nếu None sẽ lấy từ self.config).

        Returns:
            nn.Module: Mô hình Teacher đã sẵn sàng suy luận.
        """
        ckpt_path = checkpoint_path or self.config.teacher_checkpoint_path
        logger.info(f"Bắt đầu load trọng số Teacher từ: {ckpt_path}")

        # TODO [Member 2]: Hiện thực hóa việc load model và weights tại đây
        raise NotImplementedError(
            "TODO [Member 2]: Hãy hoàn thành hàm load_teacher_model() theo hướng dẫn trong docstring!"
        )

    def evaluate_sample_level(
        self,
        test_loader: Optional[DataLoader] = None,
        mask_modality: Optional[str] = None,
    ) -> pd.DataFrame:
        """Chạy vòng lặp kiểm thử trên từng mẫu dữ liệu (sample-level evaluation).

        Trong bài toán này, mỗi quả sầu riêng có thể có 6 ảnh chụp các góc (TOP, BOT, SA...),
        nhiều file ghi âm tiếng gõ (Android, iPhone) và các thông số cảm biến. Mỗi tổ hợp nạp
        vào mạng là một sample.

        Hướng dẫn thực hiện cho Member 2 (TODO):
        ---------------------------------------------------------
        1. Lấy dataloader (nếu None thì lấy `self.test_loader`).
        2. Đặt `self.model.eval()`.
        3. Sử dụng context `torch.no_grad():` để tránh tốn bộ nhớ VRAM.
        4. Lặp qua từng batch: `for batch in test_loader:`:
           - Trích xuất: `images, audios, sensors, labels, fruit_ids, ...`
           - Chuyển tensors lên thiết bị `self.device`.
           - Nếu `mask_modality` được chỉ định (để ablation study):
             * mask_modality == 'audio': gán `audios = torch.zeros_like(audios)`
             * mask_modality == 'sensor': gán `sensors = torch.zeros_like(sensors)`
             * mask_modality == 'vision': gán `images = torch.zeros_like(images)`
           - Truyền qua Teacher Model: `logits, f_mm = self.model(images, audios, sensors)`
           - Tính xác suất Softmax: `probs = F.softmax(logits, dim=-1)`
           - Lấy nhãn dự đoán: `preds = torch.argmax(probs, dim=-1)`
        5. Thu thập toàn bộ kết quả vào một DataFrame chứa các cột:
           `['fruit_id', 'camera_angle', 'true_label', 'pred_label', 'prob_unripe', 'prob_ripe', 'prob_overripe', 'is_correct']`.

        Args:
            test_loader: Dataloader tập kiểm thử.
            mask_modality: Tên nhánh cần zero-mask để thử nghiệm độ bền vững ('audio', 'sensor', 'vision' hoặc None).

        Returns:
            pd.DataFrame: Bảng kết quả dự đoán chi tiết từng sample.
        """
        logger.info(f"Bắt đầu đánh giá cấp độ sample (mask_modality={mask_modality})...")

        # TODO [Member 2]: Hiện thực hóa vòng lặp suy luận trên Test DataLoader
        raise NotImplementedError(
            "TODO [Member 2]: Hãy hiện thực hóa hàm evaluate_sample_level() theo hướng dẫn trong docstring!"
        )

    def evaluate_fruit_level(
        self,
        sample_df: pd.DataFrame,
        strategy: Optional[str] = None,
    ) -> pd.DataFrame:
        """Tổng hợp kết quả từ cấp độ mẫu đơn lẻ (nhiều góc ảnh/nhiều tiếng gõ) về từng quả sầu riêng (fruit_id).

        Trong thực tế thu hoạch và thương phẩm, quyết định được đưa ra cho "từng quả sầu riêng"
        thay vì từng ảnh chụp đơn lẻ. Do đó, đánh giá cấp độ quả (Fruit-level) phản ánh độ chính
        xác thực tế của toàn hệ thống.

        Hướng dẫn thực hiện cho Member 2 (TODO):
        ---------------------------------------------------------
        1. Nhóm DataFrame theo `fruit_id`: `grouped = sample_df.groupby('fruit_id')`.
        2. Áp dụng chiến lược tổng hợp (Aggregation Strategy):
           - 'mean_prob' (Khuyến nghị): Tính trung bình vector xác suất Softmax của toàn bộ các mẫu
             thuộc về quả đó: $P(c|fruit) = \\frac{1}{K} \\sum_{k=1}^K P(c|sample_k)$, sau đó
             chọn nhãn có xác suất trung bình cao nhất $\\arg\\max$.
           - 'majority_vote': Lấy nhãn xuất hiện nhiều nhất trong các dự đoán thành phần của quả.
             Nếu hòa phiếu, ưu tiên nhãn có tổng xác suất cao hơn.
        3. Trích xuất nhãn thực tế duy nhất của quả (`true_label`).
        4. Trả về DataFrame cấp độ quả gồm:
           `['fruit_id', 'true_label', 'pred_label', 'prob_unripe_mean', 'prob_ripe_mean', 'prob_overripe_mean', 'is_correct']`.

        Args:
            sample_df: Bảng kết quả cấp độ sample từ hàm `evaluate_sample_level()`.
            strategy: Chiến lược tổng hợp ('mean_prob' hoặc 'majority_vote').

        Returns:
            pd.DataFrame: Bảng dự đoán hợp nhất cho từng quả sầu riêng độc lập.
        """
        agg_strategy = strategy or self.config.fruit_aggregation_strategy
        logger.info(f"Tổng hợp kết quả cấp độ quả (fruit_id) theo chiến lược: {agg_strategy}...")

        # TODO [Member 2]: Hiện thực hóa hàm tổng hợp cấp độ quả tại đây
        raise NotImplementedError(
            "TODO [Member 2]: Hãy hiện thực hóa hàm evaluate_fruit_level() theo hướng dẫn trong docstring!"
        )

    def compute_metrics(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        y_probs: Optional[np.ndarray] = None,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Tính toán toàn diện các chỉ số đánh giá phân loại (Classification Metrics).

        GỢI Ý MỘT SỐ METRIC CỐT LÕI (Member 2 tham khảo):
        ---------------------------------------------------------
        1. Accuracy: Tỷ lệ dự đoán đúng trên toàn bộ tập dữ liệu.
        2. Balanced Accuracy: Trung bình cộng Recall của từng lớp (rất quan trọng khi phân bố lớp lệch).
        3. Macro-F1 (Metric trọng tâm của đề tài): Đánh giá công bằng hiệu năng giữa 3 lớp (Unripe, Ripe, Overripe).
        4. Weighted-F1: F1 có trọng số theo số lượng mẫu của từng lớp.
        5. Per-class Precision, Recall, F1:
           - F1-Unripe: Độ chính xác phát hiện sầu riêng non / chưa chín (tránh cắt non gây lỗ).
           - F1-Ripe: Độ chính xác nhận biết quả chín tới đạt chuẩn xuất bán.
           - F1-Overripe: Độ chính xác phát hiện quả chín quá / nứt vỏ / lên men.
        6. ROC-AUC (Macro One-vs-Rest): Đo lường khả năng phân tách giữa các phân phối xác suất.
        7. PR-AUC (Average Precision Macro): Thích hợp khi đánh giá các lớp thiểu số.
        8. Expected Calibration Error (ECE): Đo lường độ tin cậy của xác suất (Soft Targets) do Teacher sinh ra,
           đảm bảo Teacher cung cấp "Soft Targets" đáng tin cậy cho Student trong quá trình chưng cất.

        >>> LƯU Ý ĐẶC BIỆT DÀNH CHO MEMBER 2:
        >>> Các metric trên đây chỉ là gợi ý tham khảo.
        >>> Thành viên phụ trách cần TỰ TÌM THÊM các metric khác phù hợp với bài toán
        >>> và yêu cầu báo cáo học thuật (ví dụ: Cohen's Kappa, Matthew's Correlation Coefficient - MCC,
        >>> Brier Score, Log-Loss đa lớp, độ tin cậy nhiệt độ,...).

        Args:
            y_true: Mảng nhãn thực tế (dạng số nguyên 0, 1, 2).
            y_pred: Mảng nhãn mô hình dự đoán.
            y_probs: Ma trận xác suất dự đoán kích thước (N, NUM_CLASSES).
            prefix: Tiền tố gắn vào tên metric (ví dụ: 'sample_' hoặc 'fruit_').

        Returns:
            Dict[str, float]: Từ điển chứa tên metric và giá trị tương ứng.
        """
        # TODO [Member 2]: Dùng sklearn.metrics (accuracy_score, f1_score, precision_recall_fscore_support,
        # roc_auc_score, cohen_kappa_score...) để tính toán các chỉ số gợi ý trên
        # VÀ TỰ TÌM THÊM CÁC METRIC KHÁC để hoàn thiện hàm này.
        raise NotImplementedError(
            "TODO [Member 2]: Hãy hoàn thành hàm compute_metrics() và bổ sung thêm metric tự tìm!"
        )

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

        Hướng dẫn thực hiện cho Member 2 (TODO):
        ---------------------------------------------------------
        1. Chạy `self.evaluate_sample_level()` lần lượt cho từng trường hợp:
           - `mask_modality = None` (Full Teacher)
           - `mask_modality = 'audio'`
           - `mask_modality = 'sensor'`
           - `mask_modality = 'vision'`
        2. Với mỗi trường hợp, tổng hợp cấp độ quả bằng `self.evaluate_fruit_level()`.
        3. Tính toán các chỉ số `compute_metrics()`.
        4. Tính mức độ sụt giảm hiệu năng: $\\Delta F1 = F1_{Full} - F1_{masked}$.
        5. Đóng gói kết quả thành bảng so sánh và lưu thành file CSV.

        Returns:
            Dict[str, Dict[str, float]]: Bảng đối chiếu hiệu năng giữa các cấu hình phương thức.
        """
        logger.info("Bắt đầu phân tích đóng góp phương thức (Modality Ablation Study)...")

        # TODO [Member 2]: Hiện thực hóa quy trình thử nghiệm Ablation Study
        raise NotImplementedError(
            "TODO [Member 2]: Hãy hiện thực hóa hàm evaluate_modality_ablation() theo hướng dẫn trong docstring!"
        )

    def plot_confusion_matrix(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        title: str = "Teacher Confusion Matrix",
        save_name: str = "teacher_confusion_matrix.png",
        normalize: bool = True,
    ) -> str:
        """Vẽ và lưu trữ biểu đồ nhiệt ma trận nhầm lẫn (Confusion Matrix Heatmap).

        Hướng dẫn thực hiện cho Member 2 (TODO):
        ---------------------------------------------------------
        1. Tính ma trận nhầm lẫn qua `sklearn.metrics.confusion_matrix(y_true, y_pred)`.
        2. Nếu `normalize=True`, chia từng hàng cho tổng của hàng để ra tỷ lệ phần trăm (Recall per class).
        3. Sử dụng `seaborn.heatmap()` kết hợp matplotlib để hiển thị đẹp mắt:
           - Trục X: Predicted Label (`Unripe`, `Ripe`, `Overripe`).
           - Trục Y: True Label (`Unripe`, `Ripe`, `Overripe`).
           - Hiển thị giá trị phần trăm và số lượng mẫu trong từng ô (`annot=True`).
        4. Lưu biểu đồ vào `self.output_dir / save_name`.

        Returns:
            str: Đường dẫn file ảnh đã lưu.
        """
        # TODO [Member 2]: Hiện thực hóa việc vẽ confusion matrix
        raise NotImplementedError(
            "TODO [Member 2]: Hãy hiện thực hóa hàm plot_confusion_matrix()!"
        )

    def run_full_evaluation_pipeline(self) -> Dict[str, Any]:
        """Quy trình đánh giá toàn diện mô hình Teacher (End-to-End Evaluation Pipeline).

        Hàm điều phối chạy trọn vẹn toàn bộ các bước:
        1. Load checkpoint Teacher model.
        2. Đánh giá cấp độ sample và cấp độ quả (Full Multi-Modal).
        3. Tính toán metrics tổng hợp (Sample-level & Fruit-level).
        4. Chạy Ablation Study đo lường đóng góp của từng kênh (Audio / Sensor / Vision).
        5. Xuất báo cáo ma trận nhầm lẫn dạng biểu đồ.
        6. Lưu trữ tệp tổng hợp kết quả (JSON / CSV).

        Returns:
            Dict[str, Any]: Từ điển tổng hợp toàn bộ kết quả thực nghiệm.
        """
        logger.info("=== BẮT ĐẦU QUY TRÌNH ĐÁNH GIÁ TOÀN DIỆN TEACHER MODEL ===")

        # TODO [Member 2]: Điều phối thực thi tuần tự các phương thức đã xây dựng ở trên
        raise NotImplementedError(
            "TODO [Member 2]: Hãy hoàn thành pipeline điều phối trong run_full_evaluation_pipeline()!"
        )


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
    )

    evaluator = TeacherEvaluator(config=config)
    print(f"\n[INFO] Đã khởi tạo cấu hình đánh giá Teacher.")
    print(f"[INFO] File trọng số mục tiêu: {config.teacher_checkpoint_path}")
    print(f"[INFO] Kết quả sẽ được lưu tại: {config.output_dir}")
    print("\n[HƯỚNG DẪN THỰC THI]:")
    print("1. Thành viên 2 cần hoàn thành các hàm có đánh dấu TODO trong các class.")
    print("2. Đảm bảo file checkpoint mô hình Teacher đã được huấn luyện sẵn sàng.")
    print("3. Chạy lệnh: python eval_teacher.py --checkpoint <path_to_model>")

