"""PyTorch Dataset for the Sensor / Tabular modality.

Responsible: Member 3 (Sensor / Tabular Modality)
Project: Durian Ripeness Classification via Cross-Modal Knowledge Distillation (CMKD)

[ADDED for Teacher] Helper functions to build the 21-feature table (C_integrated19 + 2 mask flags)
matching the training configuration of `best_sensor_unimodal.pth`:
- 8 tabular columns (durian_characteristics_clean.csv or final_multimodal_manifest.csv)
- 3 thermal + 8 multispectral columns from Member 3's cache
  (workspaces/member3_sensor/02_train_unimodal_sensor/results/sensor_features_{split}.csv, NaN = missing)
- mask_thermal / mask_multispectral
- SensorPreprocessor (Train median + log1p Firmness + StandardScaler fit ONLY on Train).
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import warnings
import pandas as pd
import torch
from torch.utils.data import Dataset

import sys
from pathlib import Path

# Ensure SensorPreprocessor can be imported from workspaces/member3_sensor/01_eda_and_preprocess_sensor
# (Directory starts with a digit so it CANNOT be imported as a package -> use sys.path only)
ROOT_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT_DIR.parent
WORKSPACE_MEMBER3 = REPO_ROOT / "workspaces" / "member3_sensor" / "01_eda_and_preprocess_sensor"
if str(WORKSPACE_MEMBER3) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_MEMBER3))

import preprocess_sensor as P  # noqa: E402
from preprocess_sensor import SensorPreprocessor  # noqa: E402

SENSOR_RESULTS_DIR = REPO_ROOT / "workspaces" / "member3_sensor" / "02_train_unimodal_sensor" / "results"
SENSOR_PREPROCESSOR_PATH = SENSOR_RESULTS_DIR / "sensor_preprocessor_unimodal.joblib"
MASK_COLS = ["mask_thermal", "mask_multispectral"]
# Official feature set of best_sensor_unimodal.pth (feature_set = C_integrated19, 21 dims)
OFFICIAL_SENSOR_COLS = list(P.DEFAULT_FEATURE_COLS) + list(P.THERMAL_FEATURE_COLS) \
    + list(P.MULTISPECTRAL_FEATURE_COLS) + MASK_COLS


class SensorDataset(Dataset):
    """PyTorch Dataset for loading sensor data (Tabular / Integrated).
    
    Provides samples:
    - If has_labels=True: returns tuple (x_sensor, label, fruit_id)
    - If has_labels=False: returns tuple (x_sensor, fruit_id)
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
# [ADDED] Build sensor feature table for Teacher / evaluation
# ------------------------------------------------------------------------------
def build_sensor_table(
    split: str,
    manifest: pd.DataFrame,
    data_dir: Union[str, Path],
    features_cache_dir: Union[str, Path] = SENSOR_RESULTS_DIR,
) -> pd.DataFrame:
    """Build 1-row-per-fruit table for a split with all 21 columns of C_integrated19 + masks.

    Args:
        split: "train" | "valid" | "test" (folder name inside data/final).
        manifest: final_multimodal_manifest.csv (already loaded).
        data_dir: path to data/final.
        features_cache_dir: directory containing Member 3's sensor_features_{split}.csv.
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
            f"{cache} not found. Run workspaces/member3_sensor/02_train_unimodal_sensor/train_unimodal_sensor.py "
            "to extract thermal/multispectral features first."
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
    """Load Member 3's preprocessor; if loading fails (e.g., pandas version mismatch), refit on Train.

    Refitting is deterministic (median + StandardScaler on the same Train split), yielding identical results.
    """
    if joblib_path is not None and Path(joblib_path).exists():
        try:
            pre = SensorPreprocessor.load(joblib_path)
            if list(pre.feature_cols) == list(feature_cols):
                return pre
            warnings.warn("joblib feature_cols differ from sensor checkpoint; refitting on Train.")
        except Exception as exc:  # pandas/sklearn version mismatch
            warnings.warn(f"Failed to load {joblib_path} ({type(exc).__name__}); refitting on Train.")
    pre = SensorPreprocessor(feature_cols=list(feature_cols), log_cols=("Firmness_AVE",), scaler_type="standard")
    return pre.fit_scaler(train_table)


def sensor_vectors(preprocessor: SensorPreprocessor, table: pd.DataFrame) -> Dict[str, torch.Tensor]:
    """Map fruit_id -> normalized feature vector [F]."""
    X = preprocessor.to_tensor(table)
    return {fid: X[i] for i, fid in enumerate(table["fruit_id"].tolist())}
