"""Module tiền xử lý bảng đặc tính & cảm biến sầu riêng (Sensor / Tabular Modality).
Phụ trách: Member 3 (Nhánh Cảm biến / Bảng số liệu)

Mục tiêu:
- Định nghĩa class SensorPreprocessor để làm sạch missing values và chuẩn hóa dữ liệu số học.
- Member 1 chỉ cần gọi class này ở bước '03_create_final_dataset' và khi nạp mô hình.
"""

# import pandas as pd
# import numpy as np
# import torch
# from sklearn.preprocessing import StandardScaler
# import joblib


class SensorPreprocessor:
    """Class tiền xử lý cho nhánh cảm biến / bảng số liệu (Sensor / Tabular Modality)."""

    def __init__(self, feature_cols=None):
        # TODO: Khởi tạo danh sách các cột đặc tính số học cần dùng
        # Gợi ý: Harvest_Weight, Horizontal_Circumference, Vertical_Circumference,
        #        Pedicel_Diameter, Firmness_AVE, TSS_AVE, Dry_Matter_AVE, ESR_Days
        self.feature_cols = feature_cols
        self.scaler = None
        pass