"""PyTorch Dataset cho nhánh cảm biến (Sensor / Tabular Modality).

Phụ trách: Member 3 (Nhánh Cảm biến / Bảng số liệu)
Dự án: Phân loại độ chín sầu riêng bằng chưng cất tri thức đa phương thức (CMKD)
"""

from typing import List, Optional, Tuple, Union
import pandas as pd
import torch
from torch.utils.data import Dataset

import sys
from pathlib import Path

# Đảm bảo import được SensorPreprocessor từ workspaces/member3_sensor/01_eda_and_preprocess_sensor
ROOT_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_MEMBER3 = ROOT_DIR / "workspaces" / "member3_sensor" / "01_eda_and_preprocess_sensor"
if str(WORKSPACE_MEMBER3) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_MEMBER3))

try:
    from preprocess_sensor import SensorPreprocessor
except ImportError:
    from workspaces.member3_sensor.01_eda_and_preprocess_sensor.preprocess_sensor import SensorPreprocessor


class SensorDataset(Dataset):
    """PyTorch Dataset nạp dữ liệu cảm biến (Tabular / Integrated).
    
    Cung cấp mẫu dữ liệu:
    - Nếu has_labels=True: trả về tuple (x_sensor, label, fruit_id)
    - Nếu has_labels=False: trả về tuple (x_sensor, fruit_id)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        preprocessor: SensorPreprocessor,
        has_labels: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.preprocessor = preprocessor
        self.has_labels = has_labels
        
        self.features, self.fruit_ids = preprocessor.to_tensor(self.df, return_ids=True)
        if self.has_labels:
            self.labels = preprocessor.encode_labels(self.df)
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        x = self.features[idx]
        fid = self.fruit_ids[idx]
        if self.has_labels:
            return x, self.labels[idx], fid
        return x, fid

