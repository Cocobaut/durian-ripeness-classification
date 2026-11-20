"""Unimodal and multimodal PyTorch Datasets."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "AudioDataset": ("src.datasets.audio_dataset", "AudioDataset"),
    "AudioKnockBank": ("src.datasets.audio_dataset", "AudioKnockBank"),
    "build_audio_preprocessor": ("src.datasets.audio_dataset", "build_audio_preprocessor"),
    "list_audio_files": ("src.datasets.audio_dataset", "list_audio_files"),
    "CLAHETransform": ("src.datasets.image_dataset", "CLAHETransform"),
    "IMAGE_SIZE": ("src.datasets.image_dataset", "IMAGE_SIZE"),
    "ImageDataset": ("src.datasets.image_dataset", "ImageDataset"),
    "get_eval_transforms": ("src.datasets.image_dataset", "get_eval_transforms"),
    "get_train_transforms": ("src.datasets.image_dataset", "get_train_transforms"),
    "list_rgb_images": ("src.datasets.image_dataset", "list_rgb_images"),
    "MultimodalDataset": ("src.datasets.multimodal_dataset", "MultimodalDataset"),
    "build_multimodal_datasets": ("src.datasets.multimodal_dataset", "build_multimodal_datasets"),
    "worker_init_fn": ("src.datasets.multimodal_dataset", "worker_init_fn"),
    "OFFICIAL_SENSOR_COLS": ("src.datasets.sensor_dataset", "OFFICIAL_SENSOR_COLS"),
    "SensorDataset": ("src.datasets.sensor_dataset", "SensorDataset"),
    "build_sensor_table": ("src.datasets.sensor_dataset", "build_sensor_table"),
    "load_or_fit_sensor_preprocessor": ("src.datasets.sensor_dataset", "load_or_fit_sensor_preprocessor"),
    "sensor_vectors": ("src.datasets.sensor_dataset", "sensor_vectors"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))

