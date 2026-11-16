"""Dataset Đa phương thức (RGB + Knock Sound + Sensor) cho Multi-Modal Teacher.

Dự án: Phân loại độ chín sầu riêng bằng Chưng cất tri thức Đa phương thức (CMKD)
Tham chiếu đặc tả: Mục 1.3.1 - mỗi mẫu là tuple 5 thành phần `(img, audio, sensor, label, fruit_id)`.

Hợp thành (composition) từ 3 dataset con:
- image_dataset.py  : transform ảnh (CLAHE, 384x384, ImageNet norm) giống nhánh vision unimodal.
- audio_dataset.py  : AudioPreprocessor của Member 2 -> log-Mel [1, 64, 306].
- sensor_dataset.py : 21 đặc trưng C_integrated19 + mask, preprocessor fit trên Train.

Quy tắc ghép cặp (zero fruit-level leakage): mọi thành phần của 1 mẫu đều thuộc CÙNG một quả.
- Mỗi ẢNH RGB của quả là một mẫu (6-24 ảnh / quả).
- Âm thanh: Train chọn ngẫu nhiên 1 file của quả mỗi lần lấy mẫu; Val/Test xoay vòng tất định
  (ảnh thứ k dùng file k % n) để cả Android và iPhone đều được chấm.
- Cảm biến: vector của quả (giống nhau cho mọi mẫu của quả).

Thiếu phương thức (manifest: một số quả không có âm thanh / ảnh): tensor tương ứng = 0.
MultiModalTeacher coi input toàn 0 là "vắng mặt" và che đặc trưng nhánh đó (cũng là cách
eval_teacher.py làm zero-masking ablation).

Phần tử thứ 6 (dict) mang thông tin phụ `camera_angle`, `device` để eval_teacher.py tách metric
(nếu chỉ cần 5 phần tử, đặt `return_extras=False`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, get_worker_info

from src.datasets.audio_dataset import AudioKnockBank, build_audio_preprocessor, list_audio_files
from src.datasets.image_dataset import IMAGE_SIZE, get_eval_transforms, get_train_transforms, list_rgb_images, load_rgb
from src.datasets.sensor_dataset import (
    OFFICIAL_SENSOR_COLS,
    SENSOR_PREPROCESSOR_PATH,
    SENSOR_RESULTS_DIR,
    build_sensor_table,
    load_or_fit_sensor_preprocessor,
    sensor_vectors,
)

SPLIT_DIRS = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}


def split_dir_name(split: str) -> str:
    if split not in SPLIT_DIRS:
        raise ValueError(f"split phải thuộc {list(SPLIT_DIRS)}, nhận được {split!r}")
    return SPLIT_DIRS[split]


class MultimodalDataset(Dataset):
    """Dataset đa phương thức đồng bộ theo fruit_id.

    Hỗ trợ 2 cách khởi tạo:
    1. Trực tiếp (tiện dụng như trong tmp.md):
       `MultimodalDataset(manifest_path, data_dir, split="train", is_training=True)`
    2. Qua factory `build_multimodal_datasets(...)` (dùng chung preprocessor giữa các split):
       `MultimodalDataset(split, manifest_df, data_dir, sensor_vecs=..., audio_bank=..., train=...)`
    """

    def __init__(
        self,
        split_or_manifest: Union[str, Path],
        manifest_or_data_dir: Union[pd.DataFrame, str, Path],
        data_dir: Optional[Union[str, Path]] = None,
        sensor_vecs: Optional[Dict[str, torch.Tensor]] = None,
        audio_bank: Optional[AudioKnockBank] = None,
        train: Optional[bool] = None,
        image_size: int = IMAGE_SIZE,
        sensor_dim: int = len(OFFICIAL_SENSOR_COLS),
        seed: int = 42,
        *,
        split: Optional[str] = None,
        img_size: Optional[int] = None,
        is_training: Optional[bool] = None,
        return_extras: bool = True,
        sensor_cache_dir: Union[str, Path] = SENSOR_RESULTS_DIR,
        sensor_preprocessor_path: Optional[Union[str, Path]] = SENSOR_PREPROCESSOR_PATH,
    ):
        if img_size is not None:
            image_size = img_size

        # Phân biệt 2 kiểu gọi constructor
        if isinstance(manifest_or_data_dir, pd.DataFrame):
            # Kiểu 2: MultimodalDataset(split, manifest_df, data_dir, sensor_vecs, audio_bank, ...)
            resolved_split = split_dir_name(str(split_or_manifest))
            manifest = manifest_or_data_dir
            resolved_data_dir = Path(data_dir)
            resolved_train = bool(train) if train is not None else (
                bool(is_training) if is_training is not None else (resolved_split == "train")
            )
        else:
            # Kiểu 1: MultimodalDataset(manifest_path, data_dir, split="train", is_training=True)
            manifest_path = Path(split_or_manifest)
            resolved_data_dir = Path(manifest_or_data_dir)
            resolved_split = split_dir_name(split or "train")
            resolved_train = (
                bool(is_training) if is_training is not None
                else (bool(train) if train is not None else (resolved_split == "train"))
            )
            manifest = pd.read_csv(manifest_path)

        if sensor_vecs is None:
            train_table = build_sensor_table("train", manifest, resolved_data_dir, sensor_cache_dir)
            sensor_pre = load_or_fit_sensor_preprocessor(
                train_table, OFFICIAL_SENSOR_COLS, sensor_preprocessor_path
            )
            table = (
                train_table if resolved_split == "train"
                else build_sensor_table(resolved_split, manifest, resolved_data_dir, sensor_cache_dir)
            )
            sensor_vecs = sensor_vectors(sensor_pre, table)

        if audio_bank is None:
            audio_pre = build_audio_preprocessor(seed)
            files = list_audio_files(resolved_data_dir / resolved_split / "sound")
            audio_bank = AudioKnockBank(files, audio_pre)

        self.split = resolved_split
        self.train = resolved_train
        self.is_training = resolved_train
        self.data_dir = resolved_data_dir
        self.sensor_vecs = sensor_vecs
        self.audio_bank = audio_bank
        self.sensor_dim = sensor_dim
        self.image_size = image_size
        self.audio_shape = tuple(audio_bank.output_shape)
        self.transform = get_train_transforms(image_size) if self.train else get_eval_transforms(image_size)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.return_extras = return_extras

        fruits = manifest[manifest["Split"] == self.split].sort_values("fruit_id")
        self.label_map: Dict[str, int] = dict(zip(fruits["fruit_id"], fruits["label_idx"].astype(int)))
        images = list_rgb_images(self.data_dir / self.split / "rgb")
        img_by_fruit = {fid: g.to_dict("records") for fid, g in images.groupby("fruit_id")}
        audio_by_fruit = audio_bank.by_fruit()

        self.samples: List[Dict] = []
        for fid in self.label_map:
            imgs = img_by_fruit.get(fid) or [None]  # quả không có ảnh vẫn giữ 1 mẫu (ảnh = 0)
            audios = audio_by_fruit.get(fid, [])
            for k, img in enumerate(imgs):
                self.samples.append({
                    "fruit_id": fid,
                    "label": self.label_map[fid],
                    "img_path": img["path"] if img else None,
                    "camera_angle": img["camera_angle"] if img else "NONE",
                    "audios": audios,
                    "eval_audio_idx": k % len(audios) if audios else -1,
                })

        n_fr = len(self.label_map)
        self.stats = {
            "split": self.split,
            "n_fruits": n_fr,
            "n_samples": len(self.samples),
            "fruits_without_rgb": sum(1 for f in self.label_map if f not in img_by_fruit),
            "fruits_without_audio": sum(1 for f in self.label_map if not audio_by_fruit.get(f)),
            "fruits_without_sensor": sum(1 for f in self.label_map if f not in sensor_vecs),
        }

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    @property
    def labels(self) -> List[int]:
        return [s["label"] for s in self.samples]

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        if s["img_path"] is not None:
            img = self.transform(load_rgb(s["img_path"]))
        else:
            img = torch.zeros(3, self.image_size, self.image_size)

        if s["audios"]:
            j = int(self.rng.integers(len(s["audios"]))) if self.train else s["eval_audio_idx"]
            a = s["audios"][j]
            audio = self.audio_bank.mel(a["path"], train=self.train, rng=self.rng)
            device = a["device"]
        else:
            audio = torch.zeros(self.audio_shape)
            device = "none"

        sensor = self.sensor_vecs.get(s["fruit_id"])
        if sensor is None:
            sensor = torch.zeros(self.sensor_dim)

        if not self.return_extras:
            return img, audio, sensor.float(), s["label"], s["fruit_id"]
        extras = {"camera_angle": s["camera_angle"], "device": device}
        return img, audio, sensor.float(), s["label"], s["fruit_id"], extras


def worker_init_fn(worker_id: int) -> None:
    """Mỗi worker một seed riêng cho augmentation âm thanh (tránh trùng chuỗi ngẫu nhiên)."""
    info = get_worker_info()
    if info is not None and hasattr(info.dataset, "reseed"):
        info.dataset.reseed((torch.initial_seed() + worker_id) % (2 ** 32))


def build_multimodal_datasets(
    manifest_path: Union[str, Path],
    data_dir: Union[str, Path],
    splits: Iterable[str] = ("train", "valid"),
    image_size: int = IMAGE_SIZE,
    sensor_cache_dir: Union[str, Path] = SENSOR_RESULTS_DIR,
    sensor_preprocessor_path: Optional[Union[str, Path]] = SENSOR_PREPROCESSOR_PATH,
    sensor_feature_cols=OFFICIAL_SENSOR_COLS,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, MultimodalDataset]:
    """Dựng MultimodalDataset cho các split yêu cầu (train => augmentation bật).

    Preprocessor cảm biến luôn được nạp/fit theo tập Train (kể cả khi chỉ yêu cầu Test).
    """
    manifest = pd.read_csv(manifest_path)
    data_dir = Path(data_dir)

    train_table = build_sensor_table("train", manifest, data_dir, sensor_cache_dir)
    sensor_pre = load_or_fit_sensor_preprocessor(train_table, sensor_feature_cols, sensor_preprocessor_path)
    audio_pre = build_audio_preprocessor(seed)

    out: Dict[str, MultimodalDataset] = {}
    for split in splits:
        sp = split_dir_name(split)
        table = train_table if sp == "train" else build_sensor_table(sp, manifest, data_dir, sensor_cache_dir)
        vecs = sensor_vectors(sensor_pre, table)
        files = list_audio_files(data_dir / sp / "sound")
        if verbose:
            print(f"[data] {sp}: đang cắt tiếng gõ của {len(files)} file âm thanh ...", flush=True)
        bank = AudioKnockBank(files, audio_pre)
        ds = MultimodalDataset(sp, manifest, data_dir, vecs, bank, train=(sp == "train"),
                               image_size=image_size, sensor_dim=len(sensor_feature_cols), seed=seed)
        if ds.stats["n_samples"] == 0:
            raise RuntimeError(f"Split {sp} không có mẫu nào. Kiểm tra {data_dir / sp}.")
        if verbose:
            print(f"[data] {ds.stats}", flush=True)
        out[split] = ds
    return out
