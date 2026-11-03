"""Module kiểm thử, đánh giá mô hình Vision-Only Student và đo lường hiệu năng triển khai (CMKD Benchmark).

Phụ trách: Member 3 (Nhánh Cảm biến & Đánh giá Student / Triển khai CMKD)
Dự án: Nhận diện quả sầu riêng chín hay không bằng Chưng cất tri thức Đa phương thức (CMKD)
Tham chiếu đặc tả:
- Mục 1.3.3: Giai đoạn 2 - Chưng cất tri thức liên phương thức sang Vision-Only Student.
- Mục 1.4 (Mục tiêu 3 & 4): Nâng cao hiệu năng mô hình Vision-Only thông qua Cross-Modal Distillation,
  Giải thích mô hình trực quan (Explainable AI - XAI) và Phân tích giới hạn vật lý.

Mô tả vai trò của file:
1. Đánh giá độc lập mô hình Student chỉ nhận duy nhất ảnh RGB (Vision-Only) trên tập kiểm thử (Test Set).
2. Thiết lập bảng so sánh đối chuẩn 3 mốc (Tri-Benchmark Comparison):
   - Vision Baseline: Mô hình thị giác huấn luyện độc lập với nhãn cứng thông thường (Hard Cross-Entropy).
   - Vision Student (CMKD): Mô hình thị giác được chưng cất tri thức từ Multi-Modal Teacher
     thông qua 3 thành phần hàm mất mát: $L_{total} = \\alpha L_{CE} + \\beta L_{KD} + \\gamma L_{feat}$.
   - Multi-Modal Teacher: Mô hình đa phương thức (Ảnh + Âm thanh + Cảm biến) làm mốc trần lý thuyết (Upper Bound).
3. Định lượng mức nhảy vọt hiệu năng (Distillation Gain):
   Tính toán $\\Delta$ Accuracy, $\\Delta$ Macro-F1 để chứng minh giả thuyết khoa học:
   Tri thức từ âm thanh gõ và cảm biến giúp cải thiện năng lực nhận diện trên ảnh bề mặt vỏ ngoài.
4. Đo lường hiệu năng phần cứng và tính khả thi triển khai thực tế (Deployability & Hardware Profiling):
   Đo độ trễ suy luận (Inference Latency - ms), tốc độ xử lý (Throughput - FPS), số lượng tham số (Parameters),
   FLOPs và dung lượng mô hình trên CPU/GPU để chứng minh khả năng chạy trên điện thoại hoặc camera băng chuyền.
5. Tổng hợp kết quả đa góc nhìn theo từng cá thể quả (Fruit-level aggregation): Đánh giá khi nạp 1 góc chụp
   bất kỳ so với khi kết hợp cả 6 góc chụp (TOP, BOT, SA, SB, SC, SD).
6. So sánh đối chứng Grad-CAM giữa Vision Baseline và Student: Minh chứng trực quan rằng Student sau khi học
   từ Teacher đã tự động điều chỉnh vùng chú ý vào rãnh gai và đĩa cuống thay vì học vẹt vào nền/ánh sáng.

LƯU Ý DÀNH CHO THÀNH VIÊN PHỤ TRÁCH:
- File này đã được thiết kế sẵn khung kiến trúc hướng đối tượng (OOP) gồm các class, method signatures,
  docstrings và các bước gợi ý triển khai tuần tự.
- Thành viên phụ trách cần đọc kỹ các comment TODO trong từng hàm để tự hoàn thiện mã nguồn chi tiết.
- Các metric được gợi ý sẵn ở dưới; thành viên phụ trách CẦN TỰ TÌM THÊM các metric khác phù hợp
  với bài toán kiểm thử mô hình nén/chưng cất tri thức và đánh giá hiệu năng suy luận biên (Edge AI).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
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

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eval_student")

# ------------------------------------------------------------------------------
# 1. HẰNG SỐ CHUẨN TOÀN DỰ ÁN (PROJECT CONSTANTS)
# ------------------------------------------------------------------------------
CLASS_ORDER = ["Unripe", "Ripe", "Overripe"]  # 0: Chưa chín, 1: Chín, 2: Quá chín
LABEL_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASS_ORDER)}
IDX_TO_LABEL = {idx: cls_name for idx, cls_name in enumerate(CLASS_ORDER)}
NUM_CLASSES = len(CLASS_ORDER)


# ------------------------------------------------------------------------------
# 2. CẤU HÌNH ĐÁNH GIÁ MÔ HÌNH STUDENT (DATACLASS CONFIG)
# ------------------------------------------------------------------------------
@dataclass
class StudentEvaluatorConfig:
    """Cấu hình tham số cho quy trình đánh giá mô hình Student và đối chuẩn so sánh.

    Thành viên phụ trách có thể hiệu chỉnh các đường dẫn và tham số cho phù hợp
    với môi trường chạy thực tế hoặc bổ sung thêm các cờ cấu hình mới.
    """
    # Đường dẫn checkpoint mô hình Student đã chưng cất thành công
    student_checkpoint_path: str = "checkpoints/student_distilled_best.pth"

    # Đường dẫn checkpoint mô hình Vision Baseline (huấn luyện độc lập không KD)
    baseline_checkpoint_path: str = "checkpoints/vision_baseline_best.pth"

    # Đường dẫn checkpoint mô hình Teacher (dùng để đối chuẩn mốc trần)
    teacher_checkpoint_path: str = "checkpoints/teacher_best.pth"

    # Đường dẫn thư mục test dữ liệu RGB sạch
    test_rgb_dir: str = "data/final/test/rgb"

    # Đường dẫn file Master Manifest
    manifest_path: str = "data/final/final_multimodal_manifest.csv"

    # Thư mục lưu kết quả xuất báo cáo và biểu đồ
    output_dir: str = "workspaces/member3_sensor/03_test_eval_student/results"

    # Cấu hình phần cứng và nạp dữ liệu
    batch_size: int = 16
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Cấu hình kiểm thử độ trễ phần cứng (Latency Profiling)
    latency_warmup_runs: int = 50
    latency_test_runs: int = 200
    input_image_size: Tuple[int, int] = (224, 224)

    # Danh sách các góc chụp camera của sầu riêng
    camera_angles: List[str] = field(
        default_factory=lambda: ["TOP", "BOT", "SA", "SB", "SC", "SD"]
    )

    # Chiến lược tổng hợp góc chụp về cấp độ quả ('mean_prob' hoặc 'majority_vote')
    fruit_aggregation_strategy: str = "mean_prob"


# ------------------------------------------------------------------------------
# 3. CLASS CHÍNH: ĐÁNH GIÁ MÔ HÌNH STUDENT & SO SÁNH (STUDENT EVALUATOR)
# ------------------------------------------------------------------------------
class StudentEvaluator:
    """Bộ đánh giá toàn diện mô hình học sinh thị giác (Vision-Only Student).

    Trách nhiệm chính của class:
    1. Nạp mô hình Student, mô hình Vision Baseline và mô hình Teacher.
    2. Chạy suy luận trên tập Test độc lập (chỉ nhận ảnh RGB).
    3. Đánh giá hiệu năng ở cấp độ từng góc nhìn (Angle/Sample-level) và cấp độ từng quả (Fruit-level).
    4. Thiết lập bảng so sánh đối chuẩn (Tri-Benchmark Comparison):
       Đo lường mức cải thiện Distillation Gain ($\\Delta$ Acc, $\\Delta$ Macro-F1) giữa Student và Baseline.
    5. Đánh giá tính nhất quán của mô hình giữa các góc chụp khác nhau của cùng một quả.
    6. Xuất báo cáo khoa học dạng CSV, Markdown và biểu đồ trực quan.
    """

    def __init__(
        self,
        config: StudentEvaluatorConfig,
        student_model: Optional[nn.Module] = None,
        baseline_model: Optional[nn.Module] = None,
        teacher_model: Optional[nn.Module] = None,
        test_loader: Optional[DataLoader] = None,
    ) -> None:
        """Khởi tạo bộ đánh giá mô hình Student.

        Args:
            config: Đối tượng cấu hình.
            student_model: Mô hình Student đã khởi tạo (nếu None sẽ load từ checkpoint).
            baseline_model: Mô hình Baseline thị giác độc lập (nếu None sẽ load từ checkpoint).
            teacher_model: Mô hình Teacher đối chuẩn (tùy chọn, để so sánh mốc trần).
            test_loader: DataLoader nạp dữ liệu ảnh tập Test.
        """
        self.config = config
        self.device = torch.device(config.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.student_model = student_model
        self.baseline_model = baseline_model
        self.teacher_model = teacher_model
        self.test_loader = test_loader

        logger.info(f"Khởi tạo StudentEvaluator trên thiết bị: {self.device}")
        logger.info(f"Thư mục xuất kết quả: {self.output_dir}")

    def load_model_from_checkpoint(
        self,
        checkpoint_path: str,
        architecture_type: str = "student",
    ) -> nn.Module:
        """Tải mô hình từ file checkpoint (.pth).

        Hướng dẫn thực hiện cho Member 3 (TODO):
        ---------------------------------------------------------
        1. Khởi tạo instance kiến trúc mô hình tương ứng (ví dụ: ResNet18, EfficientNet-B0
           hoặc mô hình Vision Backbone kèm Projection Head tùy theo thiết kế của nhóm).
        2. Tải trọng số: `ckpt = torch.load(checkpoint_path, map_location=self.device)`.
        3. Nạp `state_dict` vào mô hình (chú ý bỏ qua prefix 'module.' hoặc các key của Projection Head
           nếu lúc kiểm thử chỉ dùng Backbone + Classification Head).
        4. Chuyển sang chế độ `model.eval()`.
        5. Đưa mô hình lên `self.device`.

        Args:
            checkpoint_path: Đường dẫn file checkpoint.
            architecture_type: Loại mô hình ('student', 'baseline', hoặc 'teacher').

        Returns:
            nn.Module: Mô hình đã nạp trọng số sẵn sàng kiểm thử.
        """
        logger.info(f"Đang nạp mô hình [{architecture_type}] từ: {checkpoint_path}")

        # TODO [Member 3]: Hiện thực hóa việc load checkpoint mô hình
        raise NotImplementedError(
            "TODO [Member 3]: Hãy hiện thực hóa hàm load_model_from_checkpoint() theo hướng dẫn trong docstring!"
        )

    def evaluate_vision_model(
        self,
        model: nn.Module,
        test_loader: Optional[DataLoader] = None,
        model_name: str = "Student",
    ) -> pd.DataFrame:
        """Chạy suy luận và đánh giá trên từng mẫu ảnh RGB độc lập (Sample/View-level).

        Hướng dẫn thực hiện cho Member 3 (TODO):
        ---------------------------------------------------------
        1. Sử dụng `test_loader` (hoặc `self.test_loader`).
        2. Chuyển mô hình sang `model.eval()`.
        3. Dùng `with torch.no_grad():`:
           - Lặp qua từng batch `images, labels, fruit_ids, angles, ...`
           - Đưa `images` lên thiết bị: `images = images.to(self.device)`
           - Forward qua mô hình: `logits = model(images)`
           - Tính xác suất Softmax: `probs = F.softmax(logits, dim=-1)`
           - Lấy nhãn dự đoán: `preds = torch.argmax(probs, dim=-1)`
        4. Thu thập toàn bộ dự đoán thành DataFrame:
           `['fruit_id', 'camera_angle', 'true_label', 'pred_label', 'prob_unripe', 'prob_ripe', 'prob_overripe', 'is_correct']`.

        Args:
            model: Mô hình thị giác cần đánh giá (Student hoặc Baseline).
            test_loader: DataLoader chứa ảnh RGB tập test.
            model_name: Tên định danh mô hình phục vụ logging.

        Returns:
            pd.DataFrame: Bảng kết quả dự đoán chi tiết từng góc ảnh.
        """
        logger.info(f"Bắt đầu đánh giá mô hình [{model_name}] ở cấp độ góc nhìn (View-level)...")

        # TODO [Member 3]: Hiện thực hóa vòng lặp suy luận cho mô hình Vision-Only
        raise NotImplementedError(
            "TODO [Member 3]: Hãy hoàn thành hàm evaluate_vision_model() theo hướng dẫn!"
        )

    def evaluate_fruit_level_aggregation(
        self,
        sample_df: pd.DataFrame,
        strategy: Optional[str] = None,
    ) -> pd.DataFrame:
        """Tổng hợp kết quả từ 6 góc chụp (TOP, BOT, SA, SB, SC, SD) về cấp độ quả (Fruit-level).

        Trong thực tế sử dụng hệ thống camera trên băng chuyền hoặc chụp nhiều ảnh bằng điện thoại,
        mô hình sẽ đưa ra nhận định tổng hợp cho toàn bộ quả sầu riêng.

        Hướng dẫn thực hiện cho Member 3 (TODO):
        ---------------------------------------------------------
        1. Nhóm dữ liệu theo `fruit_id`.
        2. Tính trung bình vector xác suất Softmax của các góc chụp của cùng một quả (hoặc Majority Vote).
        3. Xác định nhãn dự đoán cuối cùng của quả: $\\hat{y}_{fruit} = \\arg\\max_c \\bar{P}_c(fruit)$.
        4. Đánh giá tính nhất quán dự đoán (Angle Consistency Score): Tỷ lệ phần trăm các góc chụp
           có cùng dự đoán với nhãn tổng hợp cuối cùng của quả đó.
        5. Trả về DataFrame cấp độ quả.

        Args:
            sample_df: Bảng kết quả dự đoán từng góc ảnh từ `evaluate_vision_model()`.
            strategy: Chiến lược tổng hợp ('mean_prob' hoặc 'majority_vote').

        Returns:
            pd.DataFrame: Bảng kết quả cấp độ quả kèm chỉ số nhất quán theo góc.
        """
        agg_strategy = strategy or self.config.fruit_aggregation_strategy
        logger.info(f"Tổng hợp kết quả cấp độ quả theo chiến lược: {agg_strategy}...")

        # TODO [Member 3]: Hiện thực hóa hàm tổng hợp đa góc nhìn
        raise NotImplementedError(
            "TODO [Member 3]: Hãy hoàn thành hàm evaluate_fruit_level_aggregation()!"
        )

    def compute_metrics(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        y_probs: Optional[np.ndarray] = None,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Tính toán các chỉ số đánh giá phân loại toàn diện cho mô hình Vision.

        GỢI Ý MỘT SỐ METRIC CỐT LÕI (Member 3 tham khảo):
        ---------------------------------------------------------
        1. Top-1 Accuracy: Tỷ lệ dự đoán hoàn toàn chính xác.
        2. Balanced Accuracy: Trung bình độ nhạy (Recall) của cả 3 lớp độ chín.
        3. Macro-F1 (Metric trọng tâm của đề tài): Đánh giá công bằng, không bị lớp đa số chi phối.
        4. Weighted-F1: Đánh giá có cân nhắc tỷ trọng số mẫu thực tế.
        5. Per-Class Precision, Recall, F1:
           - Lớp Unripe (Chưa chín): Đánh giá nguy cơ thu hoạch non.
           - Lớp Ripe (Chín): Đánh giá độ nhạy bắt đúng thời điểm tiêu thụ tốt nhất.
           - Lớp Overripe (Quá chín): Đánh giá nguy cơ quả bị nứt rách/hỏng thối.
        6. ROC-AUC Macro & PR-AUC Macro: Đánh giá chất lượng của hàm phân bố xác suất Softmax.
        7. Expected Calibration Error (ECE): Kiểm tra xem xác suất mô hình tin tưởng (Confidence)
           có khớp với tỷ lệ chính xác thực tế hay không (đặc biệt quan trọng với mô hình chưng cất).

        >>> LƯU Ý ĐẶC BIỆT DÀNH CHO MEMBER 3:
        >>> Các metric trên đây chỉ là gợi ý tham khảo.
        >>> Thành viên phụ trách cần TỰ TÌM THÊM các metric khác phù hợp với bài toán
        >>> và yêu cầu báo cáo học thuật (ví dụ: Top-2 Accuracy, Matthew's Correlation Coefficient - MCC,
        >>> Cohen's Kappa, Log Loss, Brier Score Loss, False Positive Rate ở từng lớp...).

        Args:
            y_true: Mảng nhãn thực tế (0: Unripe, 1: Ripe, 2: Overripe).
            y_pred: Mảng nhãn mô hình dự đoán.
            y_probs: Ma trận xác suất Softmax kích thước (N, 3).
            prefix: Tiền tố gắn vào tên metric.

        Returns:
            Dict[str, float]: Từ điển chứa tên metric và giá trị.
        """
        # TODO [Member 3]: Hiện thực hóa hàm tính toán metrics dùng sklearn.metrics
        # VÀ TỰ TÌM THÊM CÁC METRIC KHÁC để hoàn thiện hàm này.
        raise NotImplementedError(
            "TODO [Member 3]: Hãy hoàn thành hàm compute_metrics() và bổ sung thêm metric tự tìm!"
        )

    def compare_tri_benchmark(
        self,
        baseline_metrics: Dict[str, float],
        student_metrics: Dict[str, float],
        teacher_metrics: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Lập bảng so sánh 3 mốc (Tri-Benchmark Table) và tính toán Distillation Gain.

        Tham chiếu Mục 1.4 (Mục tiêu 3 đặc tả đề tài):
        - Đo lường độ nhảy vọt hiệu năng:
          $$\\Delta Accuracy = Acc_{Student} - Acc_{Baseline}$$
          $$\\Delta Macro-F1 = F1_{Student} - F1_{Baseline}$$
        - Tính tỷ lệ bảo toàn tri thức (Knowledge Retention Rate) so với Teacher:
          $$Retention = \\frac{F1_{Student}}{F1_{Teacher}} \\times 100\\%$$

        Hướng dẫn thực hiện cho Member 3 (TODO):
        ---------------------------------------------------------
        1. Tạo bảng so sánh dạng DataFrame với các cột:
           `['Metric', 'Baseline (No KD)', 'Student (CMKD)', 'Distillation Gain (Δ)', 'Teacher (Upper Bound)', 'Retention (%)']`.
        2. Tính toán độ chênh lệch $\\Delta$ cho từng chỉ số chính (Accuracy, Macro-F1, F1-Unripe, F1-Ripe, F1-Overripe).
        3. Lưu bảng so sánh ra file CSV và định dạng bảng Markdown để đưa vào báo cáo nghiệm thu đề tài.

        Returns:
            pd.DataFrame: Bảng tổng hợp đối chuẩn 3 mốc.
        """
        logger.info("Thiết lập bảng so sánh đối chuẩn 3 mốc (Tri-Benchmark)...")

        # TODO [Member 3]: Hiện thực hóa hàm lập bảng so sánh và tính Distillation Gain
        raise NotImplementedError(
            "TODO [Member 3]: Hãy hoàn thiện hàm compare_tri_benchmark()!"
        )

    def plot_comparative_confusion_matrices(
        self,
        baseline_y_true: Sequence[int],
        baseline_y_pred: Sequence[int],
        student_y_true: Sequence[int],
        student_y_pred: Sequence[int],
        save_name: str = "comparative_confusion_matrix.png",
    ) -> str:
        """Vẽ biểu đồ so sánh ma trận nhầm lẫn song song (Side-by-Side Confusion Matrix).

        Vẽ 2 ma trận nhiệt đặt cạnh nhau:
        - Bên trái: Vision Baseline (nhiều ca nhầm lẫn giữa Unripe và Ripe do chỉ nhìn vỏ).
        - Bên phải: Student Model (giảm hẳn nhầm lẫn nhờ tri thức âm thanh gõ và cảm biến).

        Hướng dẫn thực hiện cho Member 3 (TODO):
        ---------------------------------------------------------
        1. Dùng `plt.subplots(1, 2, figsize=(14, 6))`.
        2. Subplot 1: Heatmap Confusion Matrix của Baseline (chuẩn hóa tỷ lệ %).
        3. Subplot 2: Heatmap Confusion Matrix của Student Model (chuẩn hóa tỷ lệ %).
        4. Lưu biểu đồ vào `self.output_dir / save_name`.

        Returns:
            str: Đường dẫn file ảnh đã lưu.
        """
        # TODO [Member 3]: Hiện thực hóa việc vẽ ma trận so sánh song song
        raise NotImplementedError(
            "TODO [Member 3]: Hãy hoàn thiện hàm plot_comparative_confusion_matrices()!"
        )


# ------------------------------------------------------------------------------
# 4. CLASS ĐO LƯỜNG HIỆU NĂNG PHẦN CỨNG & TRIỂN KHAI THỰC TẾ (DEPLOYMENT BENCHMARK)
# ------------------------------------------------------------------------------
class StudentDeploymentBenchmark:
    """Bộ đo lường tính khả thi triển khai thực tế (Hardware & Deployment Profiler).

    Ý nghĩa thực tiễn (Tham chiếu Mục 1.1, 1.2, 1.4 đặc tả đề tài):
    - Đề tài giải quyết mâu thuẫn "Accuracy - Deployability Trade-off":
      Mô hình Teacher dù chính xác nhưng cồng kềnh và đòi hỏi buồng đo đắt đỏ,
      trong khi Student chỉ cần một camera đơn kênh với chi phí suy luận cực thấp.
    - Class này định lượng chính xác:
      1. Độ trễ suy luận trên CPU (mô phỏng thiết bị di động của nông dân).
      2. Độ trễ suy luận trên GPU (mô phỏng camera tốc độ cao trên băng chuyền phân loại).
      3. Thông lượng xử lý (Throughput - Khung hình/giây FPS).
      4. Số lượng tham số (Parameters) và dung lượng file mô hình (MB).
      5. Độ phức tạp tính toán FLOPs / MACs.
    """

    def __init__(self, output_dir: str) -> None:
        """Khởi tạo bộ đo lường hiệu năng triển khai.

        Args:
            output_dir: Thư mục lưu kết quả báo cáo hiệu năng.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def measure_latency_and_throughput(
        self,
        model: nn.Module,
        input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
        device: str = "cpu",
        warmup_runs: int = 50,
        test_runs: int = 200,
    ) -> Dict[str, float]:
        """Đo lường thời gian suy luận (Latency) và thông lượng (Throughput) chuẩn xác.

        GỢI Ý CÁC CHỈ SỐ ĐO LƯỜNG PHẦN CỨNG (Member 3 tham khảo):
        ---------------------------------------------------------
        1. Mean Latency: Thời gian xử lý trung bình một ảnh (mili-giây ms).
        2. Std Latency: Độ lệch chuẩn thời gian suy luận.
        3. P95 / P99 Latency: Độ trễ ở phân vị thứ 95 và 99 (đảm bảo không bị trễ đỉnh bất thường).
        4. Throughput (FPS): Số lượng khung hình xử lý được trong 1 giây ($FPS = 1000 / Latency_{ms}$).

        >>> LƯU Ý ĐẶC BIỆT DÀNH CHO MEMBER 3:
        >>> Các chỉ số trên là gợi ý; thành viên phụ trách CẦN TỰ TÌM THÊM các thước đo khác
        >>> (ví dụ: VRAM Peak Memory Allocation bằng torch.cuda.max_memory_allocated(),
        >>> CPU Utilization %, Energy Consumption nếu có công cụ hỗ trợ).

        Hướng dẫn thực hiện cho Member 3 (TODO):
        ---------------------------------------------------------
        1. Chuyển mô hình sang thiết bị đích `dev = torch.device(device)` và `model.eval()`.
        2. Tạo dummy tensor: `dummy_input = torch.randn(*input_size, device=dev)`.
        3. Khởi động phần cứng (Warm-up): Chạy vòng lặp `for _ in range(warmup_runs): _ = model(dummy_input)`.
           Nếu chạy trên CUDA, bắt buộc gọi `torch.cuda.synchronize()`.
        4. Đo lường chính thức (Benchmark loop):
           - Ghi nhận thời gian bắt đầu và kết thúc từng lần chạy (sử dụng `torch.cuda.Event` trên GPU
             hoặc `time.perf_counter()` trên CPU).
           - Lưu danh sách thời gian của từng lần suy luận.
        5. Tính toán các đại lượng thống kê: Mean, Median, Min, Max, P95, P99 và FPS.

        Args:
            model: Mô hình PyTorch cần đo.
            input_size: Kích thước tensor đầu vào (B, C, H, W).
            device: 'cpu' hoặc 'cuda'.
            warmup_runs: Số lần chạy khởi động trước khi đo.
            test_runs: Số lần chạy đo lường chính thức.

        Returns:
            Dict[str, float]: Kết quả đo lường độ trễ và thông lượng.
        """
        logger.info(f"Đang đo lường Latency & Throughput trên thiết bị [{device}]...")

        # TODO [Member 3]: Hiện thực hóa việc đo lường độ trễ chính xác
        raise NotImplementedError(
            "TODO [Member 3]: Hãy hoàn thành hàm measure_latency_and_throughput()!"
        )

    def profile_model_complexity(
        self,
        model: nn.Module,
        input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    ) -> Dict[str, Any]:
        """Đo lường số lượng tham số, độ phức tạp tính toán (FLOPs) và dung lượng mô hình.

        GỢI Ý CÁC CHỈ SỐ ĐỘ PHỨC TẠP (Member 3 tham khảo):
        ---------------------------------------------------------
        1. Total Parameters: Tổng số tham số của mạng (triệu tham số - Millions / M).
        2. Trainable Parameters: Số tham số có thể huấn luyện.
        3. FLOPs / MACs: Số phép tính dấu phẩy động cần thiết cho 1 lượt suy luận ảnh
           (sử dụng thư viện `thop` hoặc `fvcore`).
        4. Checkpoint File Size: Dung lượng lưu trữ trên đĩa cứng (Megabytes - MB).

        >>> LƯU Ý ĐẶC BIỆT DÀNH CHO MEMBER 3:
        >>> Thành viên phụ trách CẦN TỰ TÌM THÊM các phương pháp tối ưu hóa triển khai
        >>> (như so sánh hiệu năng sau khi lượng tử hóa INT8 Post-Training Quantization - PTQ,
        >>> hoặc chuyển đổi sang định dạng ONNX / TensorRT / OpenVINO).

        Hướng dẫn thực hiện cho Member 3 (TODO):
        ---------------------------------------------------------
        1. Tính tổng tham số: `sum(p.numel() for p in model.parameters())`.
        2. Tính kích thước file checkpoint (.pth) tính bằng MB.
        3. Dùng `thop.profile(model, inputs=(dummy_input,))` hoặc `fvcore.nn.FlopCountAnalysis`
           để tính số lượng FLOPs/MACs.

        Returns:
            Dict[str, Any]: Từ điển chứa các thông số độ phức tạp mô hình.
        """
        logger.info("Đang tính toán độ phức tạp mô hình (Parameters, FLOPs, Size)...")

        # TODO [Member 3]: Hiện thực hóa việc tính toán độ phức tạp mô hình
        raise NotImplementedError(
            "TODO [Member 3]: Hãy hoàn thành hàm profile_model_complexity()!"
        )


# ------------------------------------------------------------------------------
# 5. CLASS ĐÁNH GIÁ THỬ NGHIỆM ABLATION STUDY (ABLATION EVALUATOR)
# ------------------------------------------------------------------------------
class StudentDistillationAblationEvaluator:
    """Bộ đánh giá thực nghiệm Ablation Study các siêu tham số chưng cất tri thức.

    Tham chiếu Mục 1.4 (Mục tiêu 3 đặc tả đề tài):
    Đánh giá tác động độc lập của các siêu tham số trong hàm mất mát tổng hợp:
    $$L_{total} = \\alpha L_{CE} + \\beta L_{KD}(T) + \\gamma L_{feat}$$
    - Tác động của nhiệt độ làm mềm Softmax ($T \\in \\{2, 3, 4, 5\\}$).
    - Tác động của tỷ trọng loss nhãn cứng so với loss nhãn mềm ($\\alpha$ vs $\\beta$).
    - Tác động của tầng căn chỉnh không gian ẩn Feature Alignment ($\\gamma = 0$ so với $\\gamma > 0$).
    - So sánh dạng hàm mất mát căn chỉnh đặc trưng: Cosine Distance vs Mean Squared Error (MSE).
    """

    def __init__(self, output_dir: str) -> None:
        """Khởi tạo bộ đánh giá Ablation Study.

        Args:
            output_dir: Thư mục lưu trữ biểu đồ và bảng kết quả ablation.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_ablation_grid(
        self,
        checkpoints_dict: Dict[str, str],
        test_loader: DataLoader,
    ) -> pd.DataFrame:
        """Đánh giá và so sánh danh sách các checkpoint mô hình tương ứng với các cấu hình ablation khác nhau.

        Hướng dẫn thực hiện cho Member 3 (TODO):
        ---------------------------------------------------------
        1. Nhận từ điển `checkpoints_dict = {'alpha1_beta1_gamma0': 'path/to/ckpt1.pth', ...}`.
        2. Với mỗi cấu hình, load trọng số và đánh giá trên `test_loader`.
        3. Thu thập các chỉ số: Accuracy, Macro-F1, F1 per class, ECE.
        4. Tổng hợp vào một bảng tổng kết Ablation Study.
        5. Xuất biểu đồ đường hoặc biểu đồ cột thể hiện xu hướng biến thiên hiệu năng theo $T$ và $\\gamma$.
        """
        # TODO [Member 3]: Hiện thực hóa hàm đánh giá lưới cấu hình ablation
        raise NotImplementedError(
            "TODO [Member 3]: Hãy hiện thực hóa hàm evaluate_ablation_grid()!"
        )


# ------------------------------------------------------------------------------
# 6. CLASS ĐỐI CHIẾU GRAD-CAM: BASELINE VS STUDENT (COMPARATIVE GRAD-CAM)
# ------------------------------------------------------------------------------
class StudentBaselineCAMComparer:
    """Bộ đối chiếu trực quan bản đồ kích hoạt nhiệt Grad-CAM giữa Baseline và Student.

    Ý nghĩa khoa học cốt lõi (Tham chiếu Mục 1.3.3 & 1.4 đặc tả đề tài):
    - Khi chỉ học bằng Hard Label (Vision Baseline), mạng thường bị quá khớp (overfit) vào nền,
      chân gai có bóng tối, hoặc điều kiện ánh sáng cục bộ.
    - Khi được dẫn dắt bởi Teacher (Student CMKD với loss căn chỉnh $L_{feat}$), các bộ lọc tích chập
      của Student được uốn nắn để tập trung vào đúng những vùng hình thái mang tính sinh học:
      rãnh gai giãn nở (groove expansion) và đĩa cuống quả (pedicel disc).
    - Class này trích xuất và đặt hai bản đồ Grad-CAM cạnh nhau trên cùng một ảnh kiểm thử
      để chứng minh thực nghiệm sự chuyển dịch vùng tập trung (Spatial Attention Shift).
    """

    def __init__(
        self,
        baseline_model: nn.Module,
        student_model: nn.Module,
        target_layer_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """Khởi tạo bộ so sánh Grad-CAM.

        Args:
            baseline_model: Mô hình Vision Baseline.
            student_model: Mô hình Vision Student sau chưng cất.
            target_layer_name: Tên tầng tích chập cuối cùng để trích xuất Feature Maps.
            device: Thiết bị tính toán.
        """
        self.baseline_model = baseline_model
        self.student_model = student_model
        self.target_layer_name = target_layer_name
        self.device = torch.device(device)

    def generate_side_by_side_cam(
        self,
        rgb_tensor: torch.Tensor,
        raw_rgb_image: np.ndarray,
        target_class: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Tạo và xuất ảnh so sánh trực quan Grad-CAM 3 khung hình:
        [ Ảnh gốc RGB ] | [ Baseline CAM (Trước KD) ] | [ Student CAM (Sau KD) ]

        Hướng dẫn thực hiện cho Member 3 (TODO):
        ---------------------------------------------------------
        1. Trích xuất Grad-CAM của Baseline Model trên ảnh `rgb_tensor`.
        2. Trích xuất Grad-CAM của Student Model trên cùng ảnh `rgb_tensor`.
        3. Chập (Overlay) bản đồ nhiệt lên `raw_rgb_image` cho cả hai mô hình.
        4. Dựng figure gồm 3 cột:
           - Cột 1: Ảnh gốc RGB của quả sầu riêng.
           - Cột 2: Bản đồ chú ý của Vision Baseline (kèm nhãn dự đoán và xác suất).
           - Cột 3: Bản đồ chú ý của Vision Student (kèm nhãn dự đoán và xác suất).
        5. Lưu hình ảnh đối chứng để phục vụ viết bài báo khoa học hoặc báo cáo nghiệm thu.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Bản đồ CAM của Baseline và của Student.
        """
        # TODO [Member 3]: Hiện thực hóa việc tạo ảnh đối chiếu Grad-CAM song song
        raise NotImplementedError(
            "TODO [Member 3]: Hãy hoàn thiện hàm generate_side_by_side_cam()!"
        )


# ------------------------------------------------------------------------------
# 7. HÀM MAIN / CLI THỰC THI THỬ NGHIỆM
# ------------------------------------------------------------------------------
def parse_args():
    """Hàm phân tích đối số dòng lệnh khi chạy script trực tiếp."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Đánh giá mô hình Vision-Only Student & Đo lường hiệu năng triển khai (CMKD)"
    )
    parser.add_argument(
        "--student-checkpoint",
        type=str,
        default="checkpoints/student_distilled_best.pth",
        help="Đường dẫn file trọng số Student model",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=str,
        default="checkpoints/vision_baseline_best.pth",
        help="Đường dẫn file trọng số Vision Baseline",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=str,
        default="checkpoints/teacher_best.pth",
        help="Đường dẫn file trọng số Teacher model (đối chuẩn)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/final/final_multimodal_manifest.csv",
        help="Đường dẫn file manifest tổng hợp",
    )
    parser.add_argument(
        "--test-rgb-dir",
        type=str,
        default="data/final/test/rgb",
        help="Đường dẫn thư mục ảnh RGB tập test",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="workspaces/member3_sensor/03_test_eval_student/results",
        help="Thư mục lưu kết quả báo cáo",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size khi kiểm thử",
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
    print(" SCRIPT ĐÁNH GIÁ VISION-ONLY STUDENT MODEL & BENCHMARK TRIỂN KHAI (CMKD)")
    print(" Phụ trách: Member 3 (Sensor Modality & Student Evaluation)")
    print("=" * 80)

    config = StudentEvaluatorConfig(
        student_checkpoint_path=args.student_checkpoint,
        baseline_checkpoint_path=args.baseline_checkpoint,
        teacher_checkpoint_path=args.teacher_checkpoint,
        manifest_path=args.manifest,
        test_rgb_dir=args.test_rgb_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
    )

    evaluator = StudentEvaluator(config=config)
    print(f"\n[INFO] Đã khởi tạo cấu hình đánh giá Student.")
    print(f"[INFO] Student Checkpoint: {config.student_checkpoint_path}")
    print(f"[INFO] Baseline Checkpoint: {config.baseline_checkpoint_path}")
    print(f"[INFO] Teacher Checkpoint: {config.teacher_checkpoint_path}")
    print(f"[INFO] Kết quả sẽ được lưu tại: {config.output_dir}")
    print("\n[HƯỚNG DẪN THỰC THI]:")
    print("1. Thành viên 3 cần hoàn thành các hàm có đánh dấu TODO trong các class.")
    print("2. Đảm bảo các checkpoint cần thiết đã được huấn luyện đầy đủ.")
    print("3. Chạy lệnh: python eval_student.py --student-checkpoint <path> --baseline-checkpoint <path>")

