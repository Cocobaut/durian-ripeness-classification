"""Module tiền xử lý âm thanh gõ sầu riêng (Audio Modality).
Phụ trách: Member 2 (Nhánh Âm thanh)

Mục tiêu:
- Định nghĩa class AudioPreprocessor để chuẩn hóa file âm thanh (.wav, .m4a) thành Mel-Spectrogram Tensor.
- Member 1 chỉ cần gọi class này ở bước '03_create_final_dataset' và khi nạp mô hình.
"""

# import librosa
# import numpy as np
# import torch


class AudioPreprocessor:
    """Class tiền xử lý cho nhánh âm thanh gõ sầu riêng (Audio Modality)."""

    def __init__(self, target_sr=22050, duration=2.0, n_mels=64):
        # TODO: Khởi tạo các tham số âm thanh (sample rate, thời lượng cắt/pad cố định, số dải Mel)
        self.target_sr = target_sr
        self.duration = duration
        self.n_mels = n_mels
        pass