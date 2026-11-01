"""Module tiền xử lý bảng đặc tính & cảm biến sầu riêng (Sensor / Tabular Modality).
Phụ trách: Member 3 (Nhánh Cảm biến / Bảng số liệu)

Mục tiêu:
- Định nghĩa class SensorPreprocessor để làm sạch missing values và chuẩn hóa dữ liệu số học.
- Member 1 chỉ cần gọi class này ở bước '03_create_final_dataset' và khi nạp mô hình.
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.preprocessing import StandardScaler

CLASS_ORDER = ["Unripe", "Ripe", "Overripe"]  # Unripe=0, Ripe=1, Overripe=2

DEFAULT_FEATURE_COLS = [
    "Harvest_Weight", "Horizontal_Circumference", "Vertical_Circumference",
    "Pedicel_Diameter", "Firmness_AVE", "TSS_AVE", "Dry_Matter_AVE", "ESR_Days",
]

# Định dạng fruit_id dùng chung giữa các nhánh. 
FRUIT_ID_FMT = "durian_{:03d}"
ID_COL = "Replicate"   # cột chứa mã quả dạng 'R1', 'R2', ...


def make_fruit_id(df, id_col=ID_COL, fmt=FRUIT_ID_FMT):
    """Tạo danh sách fruit_id từ cột Replicate (lấy phần số): 'R7' -> 'durian_007'."""
    if id_col not in df.columns:
        raise KeyError(f"Thiếu cột '{id_col}' để tạo fruit_id")
    digits = df[id_col].astype(str).str.extract(r"(\d+)")[0]
    if digits.isna().any():
        bad = df.loc[digits.isna(), id_col].tolist()
        raise ValueError(f"Không tách được số từ {id_col}: {bad}")
    return [fmt.format(int(d)) for d in digits]


class SensorPreprocessor:
    """Class tiền xử lý cho nhánh cảm biến / bảng số liệu (Sensor / Tabular Modality)."""

    def __init__(self, feature_cols=None, label_col="Ripeness",
                 log_cols=("Firmness_AVE",), class_order=None,
                 add_missing_mask=False, id_col=ID_COL):
        # Danh sách các cột đặc tính số học cần dùng
        self.feature_cols = list(feature_cols or DEFAULT_FEATURE_COLS)
        self.label_col = label_col
        # Firmness_AVE lệch phải mạnh (median 0.58, max 12.75) nên dùng log1p;
        # chỉ áp dụng cho cột nào thực sự nằm trong feature_cols
        self.log_cols = [c for c in log_cols if c in self.feature_cols]
        self.class_order = list(class_order or CLASS_ORDER)
        self.label_map = {c: i for i, c in enumerate(self.class_order)}
        self.add_missing_mask = bool(add_missing_mask)
        self.id_col = id_col
        self.medians_ = None   # median của Train cho từng cột (dùng để điền NaN)
        self.scaler = None     # StandardScaler đã fit trên Train
        self.n_fit_ = None     # số mẫu đã dùng để fit
        self.fit_ids_ = None   # tập fruit_id đã dùng để fit (None nếu df không có id_col)
        self.sklearn_version_ = None

    # ------------------------------------------------------------------
    # Tên đặc trưng đầu ra, fruit_id
    # ------------------------------------------------------------------
    @property
    def feature_names_out_(self):
        names = list(self.feature_cols)
        if self.add_missing_mask:
            names += [f"{c}_isna" for c in self.feature_cols]
        return names

    def fruit_ids(self, df):
        """fruit_id của từng hàng trong df, cùng thứ tự với transform(df)."""
        return make_fruit_id(df, id_col=self.id_col)

    def check_no_overlap(self, df, name="dữ liệu"):
        """Báo lỗi nếu df chứa quả đã dùng để fit (chống rò rỉ Train -> Val/Test).
        Gọi cho val_df, test_df sau khi fit_scaler(train_df)."""
        if self.fit_ids_ is None:
            raise RuntimeError("Chưa fit hoặc lúc fit df không có cột id: không kiểm tra được.")
        if self.id_col not in df.columns:
            raise KeyError(f"Thiếu cột '{self.id_col}' trong {name}")
        dup = sorted(set(self.fruit_ids(df)) & set(self.fit_ids_))
        if dup:
            raise ValueError(f"{name} chứa {len(dup)} quả đã dùng để fit scaler: {dup[:5]}...")
        return True

    # ------------------------------------------------------------------
    # 1. Xử lý missing values
    # ------------------------------------------------------------------
    def _check_columns(self, df):
        missing_cols = [c for c in self.feature_cols if c not in df.columns]
        if missing_cols:
            raise KeyError(f"Thiếu cột trong dữ liệu: {missing_cols}")

    def fill_missing(self, df):
        """Điền NaN bằng median của Train. Phải gọi fit_scaler(train_df) trước."""
        if self.medians_ is None:
            raise RuntimeError("Chưa fit: hãy gọi fit_scaler(train_df) trước.")
        self._check_columns(df)
        return df[self.feature_cols].astype(float).fillna(self.medians_)

    def missing_mask(self, df):
        """Ma trận cờ missing (1 = ô gốc bị NaN), shape [N, F], float32."""
        self._check_columns(df)
        return df[self.feature_cols].astype(float).isna().values.astype(np.float32)

    def _prepare(self, df):
        x = self.fill_missing(df)
        for c in self.log_cols:
            if (x[c] < 0).any():
                warnings.warn(f"Cột {c} có giá trị âm, sẽ bị clip về 0 trước log1p.")
            x[c] = np.log1p(x[c].clip(lower=0))
        return x

    # ------------------------------------------------------------------
    # 2. Fit scaler (Chỉ trên Train)
    # ------------------------------------------------------------------
    def fit_scaler(self, train_df, expected_n=None):
        """Fit median + scaler. CHỈ truyền tập Train.
        expected_n: nếu truyền vào (vd 132) sẽ báo lỗi khi số mẫu không khớp."""
        if expected_n is not None and len(train_df) != expected_n:
            raise ValueError(f"Số mẫu fit = {len(train_df)} khác kỳ vọng {expected_n}: "
                             "kiểm tra có đang truyền nhầm tập Train không.")
        self._check_columns(train_df)
        medians = train_df[self.feature_cols].astype(float).median()
        if medians.isna().any():
            bad = list(medians[medians.isna()].index)
            raise ValueError(f"Cột toàn NaN ở Train, không tính được median: {bad}")
        self.medians_ = medians
        self.scaler = StandardScaler().fit(self._prepare(train_df).values)
        self.n_fit_ = len(train_df)
        self.fit_ids_ = (sorted(set(self.fruit_ids(train_df)))
                         if self.id_col in train_df.columns else None)
        self.sklearn_version_ = sklearn.__version__
        return self

    def fit(self, train_df, expected_n=None):
        """Bí danh của fit_scaler, theo quy ước sklearn."""
        return self.fit_scaler(train_df, expected_n=expected_n)

    # ------------------------------------------------------------------
    # 3. Biến đổi dữ liệu
    # ------------------------------------------------------------------
    def transform(self, df):
        """Trả về np.ndarray float32, shape [N, F] (hoặc [N, 2F] nếu add_missing_mask)."""
        if self.scaler is None:
            raise RuntimeError("Chưa fit scaler: hãy gọi fit_scaler(train_df) trước.")
        x = self.scaler.transform(self._prepare(df).values).astype(np.float32)
        if self.add_missing_mask:
            x = np.concatenate([x, self.missing_mask(df)], axis=1)
        if not np.isfinite(x).all():
            raise ValueError("Đầu ra có NaN/Inf, kiểm tra dữ liệu đầu vào.")
        return x

    def fit_transform(self, train_df, expected_n=None):
        return self.fit_scaler(train_df, expected_n=expected_n).transform(train_df)

    def to_tensor(self, df, return_ids=False):
        """Trả về torch.FloatTensor shape [N, num_features].
        return_ids=True: trả thêm danh sách fruit_id cùng thứ tự hàng."""
        t = torch.from_numpy(self.transform(df))
        return (t, self.fruit_ids(df)) if return_ids else t

    # ------------------------------------------------------------------
    # 4. Mã hóa nhãn: Unripe -> 0, Ripe -> 1, Overripe -> 2
    # ------------------------------------------------------------------
    def encode_labels(self, df_or_series):
        s = (df_or_series[self.label_col]
             if isinstance(df_or_series, pd.DataFrame) else df_or_series)
        y = s.astype(str).str.strip().map(self.label_map)
        if y.isna().any():
            raise ValueError(f"Nhãn lạ: {sorted(set(s[y.isna()]))}")
        return torch.tensor(y.values, dtype=torch.long)

    # ------------------------------------------------------------------
    # 5. Lưu / nạp lại
    # ------------------------------------------------------------------
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "feature_cols": self.feature_cols, "label_col": self.label_col,
            "log_cols": self.log_cols, "class_order": self.class_order,
            "medians_": self.medians_, "scaler": self.scaler,
            "add_missing_mask": self.add_missing_mask, "id_col": self.id_col,
            "n_fit_": self.n_fit_, "fit_ids_": self.fit_ids_,
            "sklearn_version_": self.sklearn_version_,
        }, path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        obj = cls(d["feature_cols"], d["label_col"], d["log_cols"], d["class_order"],
                  add_missing_mask=d.get("add_missing_mask", False),
                  id_col=d.get("id_col", ID_COL))
        obj.medians_, obj.scaler = d["medians_"], d["scaler"]
        obj.n_fit_ = d.get("n_fit_")
        obj.fit_ids_ = d.get("fit_ids_")
        obj.sklearn_version_ = d.get("sklearn_version_")
        if obj.sklearn_version_ and obj.sklearn_version_ != sklearn.__version__:
            warnings.warn(f"File lưu bằng sklearn {obj.sklearn_version_}, "
                          f"môi trường hiện tại là {sklearn.__version__}.")
        return obj