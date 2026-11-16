"""PyTorch Dataset cho nhánh cảm biến (Sensor / Tabular Modality).

Phụ trách: Member 3 (Nhánh Cảm biến / Bảng số liệu)
Dự án: Phân loại độ chín sầu riêng bằng chưng cất tri thức đa phương thức (CMKD)

[BỔ SUNG cho Teacher] Các hàm dựng bảng 21 đặc trưng (C_integrated19 + 2 cờ mask) giống hệt
lúc huấn luyện `best_sensor_unimodal.pth`:
- 8 cột bảng số liệu (durian_characteristics_clean.csv hoặc final_multimodal_manifest.csv)
- 3 cột nhiệt + 8 cột đa phổ lấy từ cache của Member 3
  (workspaces/member3_sensor/02_train_unimodal_sensor/results/sensor_features_{split}.csv, NaN = thiếu)
- mask_thermal / mask_multispectral
- SensorPreprocessor (median Train + log1p Firmness + StandardScaler fit CHỈ trên Train).
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import warnings
import pandas as pd
import torch
from torch.utils.data import Dataset

import sys
from pathlib import Path

# Đảm bảo import được SensorPreprocessor từ workspaces/member3_sensor/01_eda_and_preprocess_sensor
# (Thư mục bắt đầu bằng số nên KHÔNG import dạng package được -> chỉ dùng sys.path)
ROOT_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_MEMBER3 = ROOT_DIR / "workspaces" / "member3_sensor" / "01_eda_and_preprocess_sensor"
if str(WORKSPACE_MEMBER3) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_MEMBER3))

import preprocess_sensor as P  # noqa: E402
from preprocess_sensor import SensorPreprocessor  # noqa: E402

SENSOR_RESULTS_DIR = ROOT_DIR / "workspaces" / "member3_sensor" / "02_train_unimodal_sensor" / "results"
SENSOR_PREPROCESSOR_PATH = SENSOR_RESULTS_DIR / "sensor_preprocessor_unimodal.joblib"
MASK_COLS = ["mask_thermal", "mask_multispectral"]
# Bộ đặc trưng chính thức của best_sensor_unimodal.pth (feature_set = C_integrated19, 21 chiều)
OFFICIAL_SENSOR_COLS = list(P.DEFAULT_FEATURE_COLS) + list(P.THERMAL_FEATURE_COLS) \
    + list(P.MULTISPECTRAL_FEATURE_COLS) + MASK_COLS


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


# ------------------------------------------------------------------------------
# [BỔ SUNG] Dựng bảng đặc trưng cảm biến cho Teacher / đánh giá
# ------------------------------------------------------------------------------
def build_sensor_table(
    split: str,
    manifest: pd.DataFrame,
    data_dir: Union[str, Path],
    features_cache_dir: Union[str, Path] = SENSOR_RESULTS_DIR,
) -> pd.DataFrame:
    """Bảng 1 dòng / quả của một split với đủ 21 cột của bộ C_integrated19 + mask.

    Args:
        split: "train" | "valid" | "test" (tên thư mục trong data/final).
        manifest: final_multimodal_manifest.csv (đã đọc).
        data_dir: thư mục data/final.
        features_cache_dir: nơi chứa sensor_features_{split}.csv của Member 3.
    """
    tab_csv = Path(data_dir) / split / "durian_characteristics_clean.csv"
    if tab_csv.exists():
        tab = pd.read_csv(tab_csv)
        tab["fruit_id"] = P.make_fruit_id(tab)
    else:
        tab = manifest[manifest["Split"] == split].copy()
    keep = ["fruit_id", "Replicate", "Ripeness"] + list(P.DEFAULT_FEATURE_COLS)
    tab = tab[[c for c in keep if c in tab.columns]].reset_index(drop=True)

    cache = Path(features_cache_dir) / f"sensor_features_{split}.csv"
    if not cache.exists():
        raise FileNotFoundError(
            f"Không thấy {cache}. Chạy workspaces/member3_sensor/02_train_unimodal_sensor/train_unimodal_sensor.py "
            "để trích đặc trưng nhiệt/đa phổ trước."
        )
    feats = pd.read_csv(cache)
    df = tab.merge(feats, on="fruit_id", how="left", validate="one_to_one")
    df["mask_thermal"] = df[P.THERMAL_FEATURE_COLS[0]].notna().astype(float)
    df["mask_multispectral"] = df[P.MULTISPECTRAL_FEATURE_COLS[0]].notna().astype(float)
    return df


def load_or_fit_sensor_preprocessor(
    train_table: pd.DataFrame,
    feature_cols: Sequence[str] = OFFICIAL_SENSOR_COLS,
    joblib_path: Optional[Union[str, Path]] = SENSOR_PREPROCESSOR_PATH,
) -> SensorPreprocessor:
    """Nạp preprocessor của Member 3; nếu lỗi (vd khác phiên bản pandas) thì fit lại trên Train.

    Fit lại là tất định (median + StandardScaler trên cùng tập Train) nên cho kết quả như bản gốc.
    """
    if joblib_path is not None and Path(joblib_path).exists():
        try:
            pre = SensorPreprocessor.load(joblib_path)
            if list(pre.feature_cols) == list(feature_cols):
                return pre
            warnings.warn("feature_cols của joblib khác checkpoint sensor, sẽ fit lại trên Train.")
        except Exception as exc:  # pandas/sklearn khác phiên bản
            warnings.warn(f"Không nạp được {joblib_path} ({type(exc).__name__}), sẽ fit lại trên Train.")
    pre = SensorPreprocessor(feature_cols=list(feature_cols), log_cols=("Firmness_AVE",), scaler_type="standard")
    return pre.fit_scaler(train_table)


def sensor_vectors(preprocessor: SensorPreprocessor, table: pd.DataFrame) -> Dict[str, torch.Tensor]:
    """fruit_id -> vector đặc trưng đã chuẩn hóa [F]."""
    X = preprocessor.to_tensor(table)
    return {fid: X[i] for i, fid in enumerate(table["fruit_id"].tolist())}
