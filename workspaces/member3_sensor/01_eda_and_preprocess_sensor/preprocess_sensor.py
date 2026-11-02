"""Module tiền xử lý dữ liệu cảm biến toàn diện (Sensor Modality Pipeline).

Phụ trách: Member 3 (Nhánh Cảm biến / Bảng số liệu / Quang phổ / Nhiệt)
Dự án: Phân loại độ chín sầu riêng bằng chưng cất tri thức đa phương thức (CMKD)

Module tích hợp trọn vẹn 3 phân nhánh cảm biến:
1. Bảng số liệu đặc tính sinh hóa (Tabular Physicochemical CSV: Firmness, TSS, Dry Matter...).
2. Ảnh nhiệt (Thermal Matrix CSV: tách nền Otsu, bù trừ nhiệt độ phòng delta_temp_fg_bg).
3. Ảnh đa phổ (Multispectral Mosaic TIF: tách 8 bands 256x256, phổ phản xạ trung bình).

Cung cấp:
- Class `SensorPreprocessor`: Tiền xử lý, lọc outlier, chuẩn hóa và xuất Tensor PyTorch.
- Các hàm trích xuất ảnh nhiệt và ảnh đa phổ cấp độ quả (fruit_id).
- Class `SensorDataset`: PyTorch Dataset phục vụ nạp dữ liệu cho Teacher/Student.
- Class `SensorBackbone`: Mạng MLP trích xuất vector đặc trưng f_sensor trong Fusion Module.
"""

import hashlib
import io
import os
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from PIL import Image
from skimage.filters import threshold_otsu
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import Dataset

# ------------------------------------------------------------------------------
# 1. HẰNG SỐ CHUẨN TOÀN DỰ ÁN
# ------------------------------------------------------------------------------
CLASS_ORDER = ["Unripe", "Ripe", "Overripe"]  # Unripe=0, Ripe=1, Overripe=2
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}
IDX_TO_LABEL = {i: c for i, c in enumerate(CLASS_ORDER)}

# 8 đặc trưng bảng số liệu cốt lõi (loại bỏ biến rò rỉ sau thu hoạch)
DEFAULT_FEATURE_COLS = [
    "Harvest_Weight", "Horizontal_Circumference", "Vertical_Circumference",
    "Pedicel_Diameter", "Firmness_AVE", "TSS_AVE", "Dry_Matter_AVE", "ESR_Days",
]

# Các đặc trưng rút trích từ ảnh nhiệt
THERMAL_FEATURE_COLS = ["fg_mean_temp", "fg_std_temp", "delta_temp_fg_bg"]

# Các đặc trưng rút trích từ ảnh đa phổ (8 băng tần)
MULTISPECTRAL_FEATURE_COLS = [f"band_{i}_mean" for i in range(1, 9)]

# Bộ đặc trưng cảm biến tích hợp toàn diện (Tabular + Thermal + Multispectral = 19 chiều)
INTEGRATED_SENSOR_COLS = DEFAULT_FEATURE_COLS + THERMAL_FEATURE_COLS + MULTISPECTRAL_FEATURE_COLS

FRUIT_ID_FMT = "durian_{:03d}"
ID_COL = "Replicate"


def make_fruit_id(df: pd.DataFrame, id_col: str = ID_COL, fmt: str = FRUIT_ID_FMT) -> List[str]:
    """Tạo danh sách fruit_id chuẩn từ cột Replicate hoặc Code: 'R7' -> 'durian_007'."""
    if id_col not in df.columns:
        # Fallback thử cột 'Code' nếu không có Replicate
        if "Code" in df.columns:
            id_col = "Code"
        else:
            raise KeyError(f"Thiếu cột '{id_col}' để tạo fruit_id")
            
    digits = df[id_col].astype(str).str.extract(r"(\d+)")[0]
    if digits.isna().any():
        bad = df.loc[digits.isna(), id_col].tolist()
        raise ValueError(f"Không tách được số từ {id_col}: {bad}")
    return [fmt.format(int(d)) for d in digits]


# ------------------------------------------------------------------------------
# 2. HÀM XỬ LÝ ẢNH NHIỆT (THERMAL MATRIX)
# ------------------------------------------------------------------------------
def load_thermal_matrix(csv_path: Union[str, Path]) -> Optional[np.ndarray]:
    """Đọc ma trận nhiệt độ từ file CSV (xử lý linh hoạt dấu phân cách ',' hoặc ';')."""
    try:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            return None
        mat = pd.read_csv(csv_path, header=None).values
        if mat.shape[1] == 1:
            mat = pd.read_csv(csv_path, header=None, sep=";").values
        mat = mat[~np.isnan(mat).all(axis=1)][:, ~np.isnan(mat).all(axis=0)]
        return mat.astype(np.float32)
    except Exception:
        return None


def extract_thermal_features(mat: np.ndarray) -> Dict[str, float]:
    """Tách nhiễu nhiệt nền bằng Otsu kết hợp tâm quả và trích xuất đặc trưng."""
    valid_vals = mat[np.isfinite(mat)]
    if len(valid_vals) == 0:
        return {
            "raw_mean": 0.0, "fg_mean": 0.0, "fg_std": 0.0,
            "fg_max": 0.0, "fg_min": 0.0, "bg_mean": 0.0,
            "delta_temp_fg_bg": 0.0, "thresh": 0.0,
        }
        
    thresh = threshold_otsu(valid_vals)
    h, w = mat.shape
    center_val = np.nanmedian(mat[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3])
    fruit_mask = (mat >= thresh) if center_val >= thresh else (mat < thresh)
    
    fg_temps = mat[fruit_mask & np.isfinite(mat)]
    bg_temps = mat[(~fruit_mask) & np.isfinite(mat)]
    
    fg_mean = float(np.mean(fg_temps)) if len(fg_temps) > 0 else float(np.nanmean(mat))
    bg_mean = float(np.mean(bg_temps)) if len(bg_temps) > 0 else float(np.nanmean(mat))
    
    return {
        "raw_mean": float(np.nanmean(mat)),
        "fg_mean": fg_mean,
        "fg_std": float(np.std(fg_temps)) if len(fg_temps) > 0 else 0.0,
        "fg_max": float(np.max(fg_temps)) if len(fg_temps) > 0 else fg_mean,
        "fg_min": float(np.min(fg_temps)) if len(fg_temps) > 0 else fg_mean,
        "bg_mean": bg_mean,
        "delta_temp_fg_bg": fg_mean - bg_mean,  # Độ chênh nhiệt quả - nền (triệt tiêu nhiệt độ phòng)
        "thresh": float(thresh),
    }


# ------------------------------------------------------------------------------
# 3. HÀM XỬ LÝ ẢNH ĐA PHỔ (MULTISPECTRAL MOSAIC TIF)
# ------------------------------------------------------------------------------
def safe_read_tif(file_path: Union[str, Path]) -> Optional[np.ndarray]:
    """Đọc ảnh .tif an toàn (hỗ trợ cả tifffile và PIL fallback)."""
    try:
        import tifffile as tiff
        return tiff.imread(str(file_path)).astype(np.float32)
    except Exception:
        try:
            with Image.open(str(file_path)) as pil_img:
                return np.array(pil_img, dtype=np.float32)
        except Exception:
            return None


def unpack_multispectral_grid(img_raw: np.ndarray) -> np.ndarray:
    """Chuyển ảnh mosaic 2D (512, 1024) gồm 8 băng tần thành mảng 3D (256, 256, 8)."""
    if img_raw.ndim == 2 and img_raw.shape == (512, 1024):
        bands = []
        for r in range(2):
            for c in range(4):
                band_crop = img_raw[r * 256 : (r + 1) * 256, c * 256 : (c + 1) * 256]
                bands.append(band_crop)
        return np.stack(bands, axis=-1)
    
    if img_raw.ndim == 3:
        if img_raw.shape[0] <= 20 and img_raw.shape[0] < min(img_raw.shape[1], img_raw.shape[2]):
            return np.transpose(img_raw, (1, 2, 0))
        return img_raw
        
    return img_raw


def extract_multispectral_features(img_raw: np.ndarray, fg_percentile: float = 15.0) -> Dict[str, float]:
    """Rút trích cường độ phản xạ phổ của 8 băng tần sau khi loại bỏ nền tối."""
    img = unpack_multispectral_grid(img_raw)
    if img.ndim != 3 or img.shape[-1] < 8:
        return {f"band_{i}_mean": 0.0 for i in range(1, 9)}
        
    mean_intensity = np.mean(img, axis=-1)
    fg_mask = mean_intensity > np.percentile(mean_intensity, fg_percentile)
    
    feats = {}
    for b in range(8):
        vals = img[:, :, b][fg_mask]
        feats[f"band_{b+1}_mean"] = float(np.mean(vals)) if len(vals) > 0 else 0.0
        feats[f"band_{b+1}_std"]  = float(np.std(vals)) if len(vals) > 0 else 0.0
    return feats


# ------------------------------------------------------------------------------
# 4. TRÍCH XUẤT ĐẶC TRƯNG TÍCH HỢP CHO TỪNG QUẢ (MULTI-ANGLE AGGREGATION)
# ------------------------------------------------------------------------------
def extract_fruit_thermal_vector(fruit_id: str, search_dir: Union[str, Path]) -> Dict[str, float]:
    """Gộp trung bình đặc trưng ảnh nhiệt qua các góc chụp của một quả."""
    search_path = Path(search_dir)
    pattern = f"*{fruit_id}*_thermal.csv"
    matched_files = list(search_path.glob(f"**/{pattern}"))
    
    if not matched_files:
        return {k: np.nan for k in THERMAL_FEATURE_COLS}
        
    rows = []
    for f in matched_files:
        mat = load_thermal_matrix(f)
        if mat is not None:
            feats = extract_thermal_features(mat)
            rows.append({
                "fg_mean_temp": feats["fg_mean"],
                "fg_std_temp": feats["fg_std"],
                "delta_temp_fg_bg": feats["delta_temp_fg_bg"],
            })
            
    if not rows:
        return {k: np.nan for k in THERMAL_FEATURE_COLS}
        
    df_rows = pd.DataFrame(rows)
    return df_rows.mean().to_dict()


def extract_fruit_multispectral_vector(fruit_id: str, search_dir: Union[str, Path]) -> Dict[str, float]:
    """Gộp trung bình phổ phản xạ qua các góc chụp của một quả."""
    search_path = Path(search_dir)
    pattern = f"*{fruit_id}*.tif"
    matched_files = list(search_path.glob(f"**/{pattern}"))
    
    if not matched_files:
        return {k: np.nan for k in MULTISPECTRAL_FEATURE_COLS}
        
    rows = []
    for f in matched_files:
        raw_img = safe_read_tif(f)
        if raw_img is not None:
            feats = extract_multispectral_features(raw_img)
            rows.append({k: feats[k] for k in MULTISPECTRAL_FEATURE_COLS})
            
    if not rows:
        return {k: np.nan for k in MULTISPECTRAL_FEATURE_COLS}
        
    df_rows = pd.DataFrame(rows)
    return df_rows.mean().to_dict()


def build_integrated_sensor_dataframe(
    tabular_df: pd.DataFrame,
    split_dir: Union[str, Path],
    include_thermal: bool = True,
    include_multispectral: bool = True
) -> pd.DataFrame:
    """Tạo bảng dữ liệu cảm biến tích hợp hoàn chỉnh (Tabular + Thermal + Multispectral)."""
    df_out = tabular_df.copy()
    if "fruit_id" not in df_out.columns:
        df_out["fruit_id"] = make_fruit_id(df_out)
        
    split_path = Path(split_dir)
    
    if include_thermal:
        th_records = [extract_fruit_thermal_vector(fid, split_path) for fid in df_out["fruit_id"]]
        th_df = pd.DataFrame(th_records, index=df_out.index)
        for c in THERMAL_FEATURE_COLS:
            df_out[c] = th_df[c]
            
    if include_multispectral:
        ms_records = [extract_fruit_multispectral_vector(fid, split_path) for fid in df_out["fruit_id"]]
        ms_df = pd.DataFrame(ms_records, index=df_out.index)
        for c in MULTISPECTRAL_FEATURE_COLS:
            df_out[c] = ms_df[c]
            
    return df_out


# ------------------------------------------------------------------------------
# 5. CLASS SENSOR PREPROCESSOR
# ------------------------------------------------------------------------------
class SensorPreprocessor:
    """Bộ tiền xử lý nâng cao cho nhánh cảm biến (Sensor / Tabular / Thermal / Spectral)."""

    def __init__(
        self,
        feature_cols: Optional[Sequence[str]] = None,
        label_col: str = "Ripeness",
        log_cols: Sequence[str] = ("Firmness_AVE",),
        class_order: Optional[Sequence[str]] = None,
        add_missing_mask: bool = False,
        id_col: str = ID_COL,
        clip_outliers: bool = False,
        quantile_range: Tuple[float, float] = (0.01, 0.99),
        scaler_type: str = "standard",
    ):
        self.feature_cols = list(feature_cols or DEFAULT_FEATURE_COLS)
        self.label_col = label_col
        self.log_cols = [c for c in log_cols if c in self.feature_cols]
        self.class_order = list(class_order or CLASS_ORDER)
        self.label_map = {c: i for i, c in enumerate(self.class_order)}
        self.add_missing_mask = bool(add_missing_mask)
        self.id_col = id_col
        self.clip_outliers = bool(clip_outliers)
        self.quantile_range = quantile_range
        self.scaler_type = scaler_type.lower()
        
        # Các tham số học được từ tập Train
        self.medians_: Optional[pd.Series] = None
        self.lower_bounds_: Optional[pd.Series] = None
        self.upper_bounds_: Optional[pd.Series] = None
        self.scaler: Optional[Union[StandardScaler, RobustScaler]] = None
        self.n_fit_: Optional[int] = None
        self.fit_ids_: Optional[List[str]] = None
        self.sklearn_version_: Optional[str] = None

    @property
    def feature_names_out_(self) -> List[str]:
        names = list(self.feature_cols)
        if self.add_missing_mask:
            names += [f"{c}_isna" for c in self.feature_cols]
        return names

    def fruit_ids(self, df: pd.DataFrame) -> List[str]:
        return make_fruit_id(df, id_col=self.id_col)

    def check_no_overlap(self, df: pd.DataFrame, name: str = "dữ liệu") -> bool:
        """Kiểm tra chống rò rỉ dữ liệu giữa Train và Val/Test."""
        if self.fit_ids_ is None:
            raise RuntimeError("Chưa fit hoặc lúc fit df không có cột id: không kiểm tra được.")
        dup = sorted(set(self.fruit_ids(df)) & set(self.fit_ids_))
        if dup:
            raise ValueError(f"{name} chứa {len(dup)} quả đã dùng để fit scaler: {dup[:5]}...")
        return True

    def _check_columns(self, df: pd.DataFrame) -> None:
        missing_cols = [c for c in self.feature_cols if c not in df.columns]
        if missing_cols:
            raise KeyError(f"Thiếu các cột đặc trưng trong dữ liệu: {missing_cols}")

    def fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Điền NaN bằng median của Train."""
        if self.medians_ is None:
            raise RuntimeError("Chưa fit: hãy gọi fit_scaler(train_df) trước.")
        self._check_columns(df)
        return df[self.feature_cols].astype(float).fillna(self.medians_)

    def missing_mask(self, df: pd.DataFrame) -> np.ndarray:
        """Ma trận cờ missing (1 = ô gốc bị NaN), shape [N, F], float32."""
        self._check_columns(df)
        return df[self.feature_cols].astype(float).isna().values.astype(np.float32)

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        x = self.fill_missing(df)
        # Biến đổi log1p cho các cột lệch phải
        for c in self.log_cols:
            if (x[c] < 0).any():
                warnings.warn(f"Cột {c} có giá trị âm, sẽ bị clip về 0 trước log1p.")
            x[c] = np.log1p(x[c].clip(lower=0))
            
        # Kẹp giá trị ngoại lai (Winsorization) nếu được bật
        if self.clip_outliers and self.lower_bounds_ is not None and self.upper_bounds_ is not None:
            x = x.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)
            
        return x

    def fit_scaler(self, train_df: pd.DataFrame, expected_n: Optional[int] = None) -> "SensorPreprocessor":
        """Fit median + bounds + scaler CHỈ trên tập Train."""
        if expected_n is not None and len(train_df) != expected_n:
            raise ValueError(f"Số mẫu fit = {len(train_df)} khác kỳ vọng {expected_n}: "
                             "kiểm tra có đang truyền nhầm tập Train không.")
        self._check_columns(train_df)
        
        # 1. Tính Median điền khuyết
        medians = train_df[self.feature_cols].astype(float).median()
        if medians.isna().any():
            bad = list(medians[medians.isna()].index)
            raise ValueError(f"Cột toàn NaN ở Train, không tính được median: {bad}")
        self.medians_ = medians
        
        # 2. Tính ngưỡng ngoại lai nếu bật clip_outliers
        if self.clip_outliers:
            q_low, q_high = self.quantile_range
            prep_temp = train_df[self.feature_cols].astype(float).fillna(self.medians_)
            for c in self.log_cols:
                prep_temp[c] = np.log1p(prep_temp[c].clip(lower=0))
            self.lower_bounds_ = prep_temp.quantile(q_low)
            self.upper_bounds_ = prep_temp.quantile(q_high)
            
        # 3. Khởi tạo và fit scaler
        if self.scaler_type == "robust":
            self.scaler = RobustScaler().fit(self._prepare(train_df).values)
        else:
            self.scaler = StandardScaler().fit(self._prepare(train_df).values)
            
        self.n_fit_ = len(train_df)
        self.fit_ids_ = (sorted(set(self.fruit_ids(train_df)))
                         if self.id_col in train_df.columns or "Code" in train_df.columns else None)
        self.sklearn_version_ = sklearn.__version__
        return self

    def fit(self, train_df: pd.DataFrame, expected_n: Optional[int] = None) -> "SensorPreprocessor":
        return self.fit_scaler(train_df, expected_n=expected_n)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Chuyển đổi dữ liệu sang numpy ndarray float32."""
        if self.scaler is None:
            raise RuntimeError("Chưa fit scaler: hãy gọi fit_scaler(train_df) trước.")
        x = self.scaler.transform(self._prepare(df).values).astype(np.float32)
        if self.add_missing_mask:
            x = np.concatenate([x, self.missing_mask(df)], axis=1)
        if not np.isfinite(x).all():
            raise ValueError("Đầu ra có NaN/Inf, kiểm tra lại dữ liệu đầu vào.")
        return x

    def fit_transform(self, train_df: pd.DataFrame, expected_n: Optional[int] = None) -> np.ndarray:
        return self.fit_scaler(train_df, expected_n=expected_n).transform(train_df)

    def to_tensor(self, df: pd.DataFrame, return_ids: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[str]]]:
        """Trả về Tensor float32 [N, num_features]."""
        t = torch.from_numpy(self.transform(df))
        return (t, self.fruit_ids(df)) if return_ids else t

    def encode_labels(self, df_or_series: Union[pd.DataFrame, pd.Series]) -> torch.Tensor:
        """Mã hóa nhãn sang chỉ số số nguyên [N] (torch.long)."""
        s = (df_or_series[self.label_col]
             if isinstance(df_or_series, pd.DataFrame) else df_or_series)
        y = s.astype(str).str.strip().map(self.label_map)
        if y.isna().any():
            raise ValueError(f"Nhãn lạ không xác định: {sorted(set(s[y.isna()]))}")
        return torch.tensor(y.values, dtype=torch.long)

    def save(self, path: Union[str, Path]) -> None:
        """Lưu preprocessor ra file joblib."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "feature_cols": self.feature_cols,
            "label_col": self.label_col,
            "log_cols": self.log_cols,
            "class_order": self.class_order,
            "medians_": self.medians_,
            "lower_bounds_": self.lower_bounds_,
            "upper_bounds_": self.upper_bounds_,
            "scaler": self.scaler,
            "add_missing_mask": self.add_missing_mask,
            "id_col": self.id_col,
            "clip_outliers": self.clip_outliers,
            "quantile_range": self.quantile_range,
            "scaler_type": self.scaler_type,
            "n_fit_": self.n_fit_,
            "fit_ids_": self.fit_ids_,
            "sklearn_version_": self.sklearn_version_,
        }, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SensorPreprocessor":
        """Nạp preprocessor từ file joblib."""
        d = joblib.load(path)
        obj = cls(
            feature_cols=d["feature_cols"],
            label_col=d["label_col"],
            log_cols=d["log_cols"],
            class_order=d["class_order"],
            add_missing_mask=d.get("add_missing_mask", False),
            id_col=d.get("id_col", ID_COL),
            clip_outliers=d.get("clip_outliers", False),
            quantile_range=d.get("quantile_range", (0.01, 0.99)),
            scaler_type=d.get("scaler_type", "standard"),
        )
        obj.medians_ = d["medians_"]
        obj.lower_bounds_ = d.get("lower_bounds_")
        obj.upper_bounds_ = d.get("upper_bounds_")
        obj.scaler = d["scaler"]
        obj.n_fit_ = d.get("n_fit_")
        obj.fit_ids_ = d.get("fit_ids_")
        obj.sklearn_version_ = d.get("sklearn_version_")
        return obj


# ------------------------------------------------------------------------------
# 6. PYTORCH DATASET CHO NHÁNH CẢM BIẾN
# ------------------------------------------------------------------------------
class SensorDataset(Dataset):
    """PyTorch Dataset nạp dữ liệu cảm biến (Tabular / Integrated)."""

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
# 7. MẠNG MLP SENSOR BACKBONE PHỤC VỤ UNIMODAL & TEACHER FUSION
# ------------------------------------------------------------------------------
class SensorBackbone(nn.Module):
    """Mạng MLP trích xuất đặc trưng cảm biến f_sensor in R^{latent_dim}.
    
    Sử dụng LayerNorm để đảm bảo tính ổn định tuyệt đối khi kích thước batch nhỏ
    hoặc khi đánh giá từng mẫu đơn lẻ (batch_size=1).
    """

    def __init__(
        self,
        input_dim: int = len(DEFAULT_FEATURE_COLS),
        hidden_dims: Sequence[int] = (64, 48),
        latent_dim: int = 32,
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.extend([
                nn.Linear(in_d, h_d),
                nn.LayerNorm(h_d),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_d = h_d
            
        layers.append(nn.Linear(in_d, latent_dim))
        layers.append(nn.LayerNorm(latent_dim))
        layers.append(nn.ReLU(inplace=True))
        
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x: torch.Tensor, return_latent: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Trả về logits (phân loại đơn phương thức) hoặc vector đặc trưng ẩn f_sensor."""
        feat = self.encoder(x)
        logits = self.classifier(feat)
        return (logits, feat) if return_latent else logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Trích xuất vector biểu diễn f_sensor in R^{latent_dim} cho Multimodal Fusion."""
        return self.encoder(x)
