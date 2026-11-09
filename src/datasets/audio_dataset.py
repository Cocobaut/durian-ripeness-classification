"""Dataset & tiền xử lý nhánh Âm thanh gõ (Knock Sound -> log-Mel).

Dự án: Phân loại độ chín sầu riêng bằng Chưng cất tri thức Đa phương thức (CMKD)

Tái sử dụng NGUYÊN `AudioPreprocessor` của Member 2
(workspaces/member2_audio/01_eda_and_preprocess_audio/preprocess_audio.py) để đầu vào
giống hệt lúc huấn luyện backbone ResNet-18 âm thanh: log-Mel [1, 64, 306] (6 tiếng gõ x 0.5 s).

Tên file chuẩn của data/final/{split}/sound: `durian_XXX_{android|iphone}_{Ripeness}_knock[_k].wav`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.common import PROJECT_ROOT, add_to_sys_path

add_to_sys_path(PROJECT_ROOT / "workspaces" / "member2_audio" / "01_eda_and_preprocess_audio")
from preprocess_audio import AudioPreprocessor  # noqa: E402

AUDIO_FILE_PATTERN = re.compile(
    r"^durian_(\d+)_(android|iphone)_(Unripe|Ripe|Overripe)_knock(?:_\d+)?\.wav$", re.IGNORECASE
)


def list_audio_files(sound_dir: Union[str, Path]) -> pd.DataFrame:
    """Liệt kê file âm thanh của một split: cột [path, file, fruit_id, device]."""
    sound_dir = Path(sound_dir)
    rows = []
    if sound_dir.is_dir():
        for p in sorted(sound_dir.glob("*.wav")):
            m = AUDIO_FILE_PATTERN.match(p.name)
            if not m:
                continue
            rows.append({"path": str(p), "file": p.name,
                         "fruit_id": f"durian_{int(m.group(1)):03d}", "device": m.group(2).lower()})
    return pd.DataFrame(rows, columns=["path", "file", "fruit_id", "device"])


def build_audio_preprocessor(seed: int = 42) -> AudioPreprocessor:
    """AudioPreprocessor với cấu hình mặc định = cấu hình đã train backbone âm thanh."""
    return AudioPreprocessor(random_state=seed)


class AudioKnockBank:
    """Cắt sẵn tiếng gõ của mọi file một lần (bước tốn thời gian nhất) và giữ trong RAM.

    Mỗi lần lấy mẫu chỉ còn bước chọn K tiếng gõ + log-Mel (+ SpecAugment khi train).
    """

    def __init__(self, files: pd.DataFrame, preprocessor: AudioPreprocessor):
        self.pre = preprocessor
        self.files = files.reset_index(drop=True)
        self.knocks: Dict[str, np.ndarray] = {p: preprocessor.load_knocks(p) for p in self.files["path"]}

    def by_fruit(self) -> Dict[str, List[Dict]]:
        out: Dict[str, List[Dict]] = {}
        for _, r in self.files.iterrows():
            out.setdefault(r["fruit_id"], []).append({"path": r["path"], "device": r["device"], "file": r["file"]})
        return out

    def mel(self, path: str, train: bool, rng: Optional[np.random.Generator] = None) -> torch.Tensor:
        return self.pre.process_knocks(self.knocks[path], train=train, rng=rng)

    @property
    def output_shape(self):
        return self.pre.output_shape


class AudioDataset(Dataset):
    """Dataset âm thanh đơn phương thức: (mel [1, 64, T], label, fruit_id, {"device": ...})."""

    def __init__(self, bank: AudioKnockBank, label_map: Dict[str, int], train: bool = False, seed: int = 42):
        self.bank = bank
        self.files = bank.files[bank.files["fruit_id"].isin(label_map)].reset_index(drop=True)
        self.label_map = label_map
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        r = self.files.iloc[idx]
        x = self.bank.mel(r["path"], train=self.train, rng=self.rng)
        return x, int(self.label_map[r["fruit_id"]]), r["fruit_id"], {"device": r["device"]}

