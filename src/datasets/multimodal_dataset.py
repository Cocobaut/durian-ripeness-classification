"""Multimodal Dataset (Multi-View RGB Bag + Knock Sound + Sensor) for the Multi-Modal Teacher & Student.

Project: Durian Ripeness Classification via Cross-Modal Knowledge Distillation (CMKD)
Specification reference: Section 1.3.1 - each sample is a 5-tuple `(img_bag, audio, sensor, label, fruit_id)`
(plus optional 6th element `extras` dict).

Composed from 3 sub-datasets:
- image_dataset.py  : Multi-View RGB Fruit Bag (`[N_views, 3, 384, 384]`, CLAHE, ImageNet norm) matching
                      the improved Multi-View ConvNeXt V2-Base + Lightweight Gated Attention branch.
- audio_dataset.py  : Member 2's AudioPreprocessor -> log-Mel `[1, 64, 306]`.
- sensor_dataset.py : 21 features C_integrated19 + mask, preprocessor fit on Train.

Pairing rule (zero fruit-level leakage): all components of 1 sample belong to the SAME fruit (`fruit_id`).
- Vision: All RGB views belonging to `fruit_id` are grouped into a Multi-View Bag `{I_1, ..., I_N}`.
  + During training: randomly subsamples `4..6` views per fruit per epoch (Random View Subsampling / View Dropout).
  + During Val/Test: loads all available views of the fruit so Lightweight Gated Attention pools every angle.
- Audio: Each audio recording of the fruit (`android` / `iphone`) is paired with the fruit's Multi-View RGB Bag
  and Sensor vector so all recordings and both devices are trained and evaluated.
- Sensor: fruit-level feature vector `[21]`.

Missing modality (manifest: some fruits lack audio / RGB): corresponding tensor = 0.
MultiModalTeacher treats an all-zero input as "absent" and masks that branch's features
(which is also how `eval_teacher.py` performs zero-masking ablation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, get_worker_info

from src.datasets.audio_dataset import AudioKnockBank, build_audio_preprocessor, list_audio_files
from src.datasets.image_dataset import (
    IMAGE_SIZE,
    MAX_EVAL_VIEWS,
    MAX_TRAIN_VIEWS,
    MIN_TRAIN_VIEWS,
    get_eval_transforms,
    get_train_transforms,
    list_rgb_images,
    load_multiview_bag_tensor,
)
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
        raise ValueError(f"split must be one of {list(SPLIT_DIRS)}, got {split!r}")
    return SPLIT_DIRS[split]


class MultimodalDataset(Dataset):
    """Multimodal dataset synchronized by `fruit_id` with Multi-View RGB Fruit Bags.

    Supports 2 initialization styles:
    1. Direct (convenient):
       `MultimodalDataset(manifest_path, data_dir, split="train", is_training=True)`
    2. Via factory `build_multimodal_datasets(...)` (shares preprocessor across splits):
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
        min_train_views: int = MIN_TRAIN_VIEWS,
        max_train_views: int = MAX_TRAIN_VIEWS,
        max_eval_views: int = MAX_EVAL_VIEWS,
    ):
        if img_size is not None:
            image_size = img_size

        # Distinguish between the 2 constructor call patterns
        if isinstance(manifest_or_data_dir, pd.DataFrame):
            resolved_split = split_dir_name(str(split_or_manifest))
            manifest = manifest_or_data_dir
            resolved_data_dir = Path(data_dir)
            resolved_train = bool(train) if train is not None else (
                bool(is_training) if is_training is not None else (resolved_split == "train")
            )
        else:
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
        self.image_size = int(image_size)
        self.min_train_views = int(min_train_views)
        self.max_train_views = int(max_train_views)
        self.max_eval_views = int(max_eval_views)
        self.audio_shape = tuple(audio_bank.output_shape)
        self.transform = get_train_transforms(self.image_size) if self.train else get_eval_transforms(self.image_size)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.return_extras = return_extras

        fruits = manifest[manifest["Split"] == self.split].sort_values("fruit_id")
        self.label_map: Dict[str, int] = dict(zip(fruits["fruit_id"], fruits["label_idx"].astype(int)))
        images = list_rgb_images(self.data_dir / self.split / "rgb")
        img_by_fruit = {
            fid: g.sort_values("file").to_dict("records")
            for fid, g in images.groupby("fruit_id")
        }
        audio_by_fruit = audio_bank.by_fruit()

        max_views_found = max((len(v) for v in img_by_fruit.values()), default=0)
        if max_views_found == 0:
            self.pad_views = 1
        elif self.train:
            self.pad_views = min(self.max_train_views, max_views_found)
        else:
            self.pad_views = min(self.max_eval_views, max_views_found)

        self.samples: List[Dict] = []
        for fid in self.label_map:
            img_list = img_by_fruit.get(fid, [])
            img_paths = [img["path"] for img in img_list]
            audios = audio_by_fruit.get(fid, [])
            audio_entries = audios if audios else [None]
            for a in audio_entries:
                self.samples.append({
                    "fruit_id": fid,
                    "label": self.label_map[fid],
                    "img_paths": img_paths,
                    "img_path": img_paths[0] if img_paths else None,
                    "camera_angle": "MULTI_VIEW" if img_paths else "NONE",
                    "num_views": len(img_paths),
                    "audio": a,
                    "audios": audios,
                })

        n_fr = len(self.label_map)
        self.stats = {
            "split": self.split,
            "n_fruits": n_fr,
            "n_samples": len(self.samples),
            "n_rgb_views": int(len(images)),
            "pad_views": int(self.pad_views),
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

        if s["img_paths"]:
            img, _num_valid = load_multiview_bag_tensor(
                s["img_paths"],
                transform=self.transform,
                image_size=self.image_size,
                is_train=self.train,
                min_train_views=self.min_train_views,
                max_train_views=self.max_train_views,
                max_eval_views=self.max_eval_views,
                pad_views=self.pad_views,
                rng=self.rng,
            )
        else:
            img = torch.zeros(self.pad_views, 3, self.image_size, self.image_size, dtype=torch.float32)

        if s["audio"] is not None:
            a = s["audio"]
            audio = self.audio_bank.mel(a["path"], train=self.train, rng=self.rng)
            device = a["device"]
        elif s["audios"]:
            j = int(self.rng.integers(len(s["audios"])))
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
    """Assign each worker a distinct seed for view subsampling and audio augmentation."""
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
    """Build MultimodalDataset for the requested splits (train => view subsampling + augmentation enabled).

    The sensor preprocessor is always loaded/fit on the Train split (even when only Test is requested).
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
            print(f"[data] {sp}: extracting knocks from {len(files)} audio files ...", flush=True)
        bank = AudioKnockBank(files, audio_pre)
        ds = MultimodalDataset(
            sp,
            manifest,
            data_dir,
            vecs,
            bank,
            train=(sp == "train"),
            image_size=image_size,
            sensor_dim=len(sensor_feature_cols),
            seed=seed,
        )
        if ds.stats["n_samples"] == 0:
            raise RuntimeError(f"Split {sp} has 0 samples. Check {data_dir / sp}.")
        if verbose:
            print(f"[data] {ds.stats}", flush=True)
        out[split] = ds
    return out
