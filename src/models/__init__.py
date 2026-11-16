"""Các kiến trúc mạng nơ-ron (Backbones, Fusion & Teacher/Student Models)."""

from src.models.audio_backbone import AudioBackbone
from src.models.sensor_backbone import SensorBackbone
from src.models.student_model import (
    DEFAULT_STUDENT_BACKBONE,
    STUDENT_BACKBONE_ALIASES,
    ProjectionHead,
    StudentConfig,
    VisionStudent,
    build_student_from_checkpoint,
    resolve_student_backbone_name,
    student_checkpoint_dict,
)
from src.models.teacher_model import (
    MODALITIES,
    MultiModalTeacher,
    MultimodalFusion,
    TeacherConfig,
    build_teacher_from_checkpoint,
    teacher_checkpoint_dict,
)
from src.models.vision_backbone import VISION_MODEL_NAME, VisionBackbone

__all__ = [
    "AudioBackbone",
    "DEFAULT_STUDENT_BACKBONE",
    "MODALITIES",
    "MultiModalTeacher",
    "MultimodalFusion",
    "ProjectionHead",
    "STUDENT_BACKBONE_ALIASES",
    "SensorBackbone",
    "StudentConfig",
    "TeacherConfig",
    "VISION_MODEL_NAME",
    "VisionBackbone",
    "VisionStudent",
    "build_student_from_checkpoint",
    "build_teacher_from_checkpoint",
    "resolve_student_backbone_name",
    "student_checkpoint_dict",
    "teacher_checkpoint_dict",
]
