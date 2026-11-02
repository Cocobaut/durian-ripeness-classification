"""Module tiền xử lý âm thanh gõ quả sầu riêng cho nhánh âm học (Audio Modality).

Phụ trách: Member 2 (Nhánh Âm thanh)

Module chỉ gồm một class duy nhất `AudioPreprocessor`, gom toàn bộ các bước đã được
kiểm chứng trong notebook `01_eda_and_preprocess_audio.ipynb`:
- Đọc file `.wav` (Android, PCM 48 kHz stereo) và `.m4a` (iPhone, ALAC 48 kHz mono) bằng ffmpeg,
  đưa về mono và resample về `target_sr`. Cả hai thiết bị đi qua cùng một bộ giải mã/resample
  để không tạo ra khác biệt nhân tạo giữa thiết bị.
- Lọc nhiễu tần thấp (high-pass) và chuẩn hóa biên độ đỉnh trước khi dò tiếng gõ.
- Mỗi file dài khoảng 30 giây và chứa khoảng 30 tiếng gõ. Thay vì cắt khoảng lặng rồi
  pad/truncate cả file, class dò onset của từng tiếng gõ (loại tiếng gõ quá yếu và tiếng gõ
  bị cụt ở biên file), cắt mỗi tiếng gõ thành một đoạn dài cố định và ghép K tiếng gõ thành một mẫu.
- Mel-Spectrogram, chuyển sang log-dB rồi chuẩn hóa từng mẫu (mean 0, std 1). Bước này loại bỏ
  chênh lệch độ lớn giữa hai thiết bị và giữa các lần gõ mạnh, nhẹ.
- SpecAugment (Time Masking + Frequency Masking), chỉ dùng khi train.
- Đọc nhãn từ tên file và lập bảng chỉ mục có gắn cờ nhãn nghi sai (`label_suspect`); việc lọc
  được quyết định ở bước tạo MultimodalDataset. Class không tự lọc file nào.

Ví dụ sử dụng:
    preprocessor = AudioPreprocessor(random_state=42)
    mel = preprocessor.process("durian_002_unripe_iphone.m4a", train=True)   # torch.float32 [1, 64, 306]
    fruit_id, label, device = AudioPreprocessor.parse_filename("durian_002_unripe_iphone.m4a")

    # Bảng chỉ mục toàn bộ file kèm cờ nhãn nghi sai, sau đó lấy danh sách file sạch
    index = preprocessor.build_file_index("data/split_data")
    train_files = AudioPreprocessor.get_clean_files(index, split="train")

    # Dùng cache khi một file được đọc nhiều lần (nhiều epoch)
    preprocessor = AudioPreprocessor(cache=True, random_state=42)
"""

import hashlib
import io
import os
import re
import shutil
import subprocess
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import scipy.signal as sps
import soundfile as sf
import torch


class AudioPreprocessor:
    """Bộ tiền xử lý âm thanh gõ quả sầu riêng thành Mel-Spectrogram cố định kích thước.

    Class cung cấp:
    - `process()`: đường dẫn file -> tensor float32 [1, n_mels, T] (train có augmentation).
    - `load()`, `denoise()`, `detect_knocks()`, `load_knocks()`, `prepare_knocks()`,
      `knocks_to_logmel()`, `normalize()`, `spec_augment()`: các bước con, dùng để trực quan hóa
      hoặc để dựng pipeline khác (vd feature extractor của AST) trên cùng cách chọn tiếng gõ.
    - `parse_filename()`: tên file -> (fruit_id, label, device).
    - `build_file_index()` / `get_clean_files()`: bảng chỉ mục file kèm cờ nhãn nghi sai.
    """

    LABELS = ("unripe", "ripe", "overripe")
    LABEL_TO_IDX = {name: idx for idx, name in enumerate(LABELS)}
    DEVICES = ("android", "iphone")
    FILENAME_PATTERN = re.compile(
        r"^durian_(\d{3})_(unripe|ripe|overripe)_(android|iphone)\.(wav|m4a)$", re.IGNORECASE
    )
    # Hậu tố nhãn trong tên file gốc của iPhone, vd "8-RI.m4a", "162-2-or.m4a"
    IPHONE_SUFFIX_PATTERN = re.compile(r"-(un|ri|or)\.m4a$", re.IGNORECASE)
    SUFFIX_TO_LABEL = {"un": "unripe", "ri": "ripe", "or": "overripe"}

    def __init__(
        self,
        target_sr: int = 16000,
        n_knocks: int = 6,
        knock_duration: float = 0.5,
        pre_onset: float = 0.05,
        highpass_hz: float = 60.0,
        min_knock_gap: float = 0.3,
        knock_floor_db: float = 15.0,
        knock_prominence_db: float = 10.0,
        knock_rel_db: float = 12.0,
        n_fft: int = 1024,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 64,
        fmin: float = 50.0,
        fmax: Optional[float] = None,
        top_db: float = 80.0,
        time_shift: float = 0.02,
        freq_mask_param: int = 8,
        time_mask_param: int = 20,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        random_state: Optional[int] = None,
        cache: bool = False,
    ):
        """Khởi tạo bộ tiền xử lý.

        Args:
            target_sr: Tần số lấy mẫu sau resample (Hz). 16 kHz giữ được dải tới 8 kHz, bao trọn
                gần như toàn bộ năng lượng tiếng gõ, và trùng với tần số AST được huấn luyện.
            n_knocks: Số tiếng gõ K ghép thành một mẫu.
            knock_duration: Độ dài mỗi đoạn tiếng gõ (giây), tính từ `pre_onset` trước đỉnh.
            pre_onset: Lùi điểm bắt đầu đoạn cắt về trước đỉnh năng lượng (giây) để giữ trọn pha tấn công.
            highpass_hz: Tần số cắt bộ lọc high-pass (loại tiếng ù, rung tay cầm máy); 0 để tắt.
            min_knock_gap: Khoảng cách tối thiểu giữa hai tiếng gõ (giây) khi dò đỉnh.
            knock_floor_db: Đỉnh phải cao hơn nền nhiễu ít nhất bấy nhiêu dB mới được tính là tiếng gõ.
            knock_prominence_db: Độ nổi bật tối thiểu (dB) của đỉnh so với vùng lân cận.
            knock_rel_db: Loại tiếng gõ yếu hơn trung vị mức đỉnh các tiếng gõ trong file quá bấy nhiêu dB.
            n_fft: Số điểm FFT.
            win_length: Độ dài cửa sổ STFT (mẫu), 400 mẫu = 25 ms ở 16 kHz.
            hop_length: Bước nhảy STFT (mẫu), 160 mẫu = 10 ms ở 16 kHz.
            n_mels: Số dải Mel.
            fmin: Tần số thấp nhất của bộ lọc Mel (Hz).
            fmax: Tần số cao nhất của bộ lọc Mel (Hz), mặc định bằng target_sr / 2.
            top_db: Ngưỡng cắt động của log-dB so với giá trị lớn nhất.
            time_shift: Biên độ dịch ngẫu nhiên điểm bắt đầu mỗi tiếng gõ khi train (giây).
            freq_mask_param: Độ rộng tối đa của mỗi dải che tần số (số dải Mel).
            time_mask_param: Độ rộng tối đa của mỗi dải che thời gian (số khung).
            num_freq_masks: Số dải che tần số.
            num_time_masks: Số dải che thời gian.
            random_state: Seed cho bộ sinh số ngẫu nhiên nội bộ (chọn tiếng gõ, dịch, SpecAugment).
            cache: Lưu các đoạn tiếng gõ đã cắt của từng file trong bộ nhớ để các epoch sau không phải giải mã lại.
        """
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("Không tìm thấy ffmpeg. Cài bằng `brew install ffmpeg` (macOS) hoặc `apt install ffmpeg`.")
        if n_knocks < 1:
            raise ValueError(f"n_knocks phải >= 1, nhận được: {n_knocks}")

        self.target_sr = target_sr
        self.n_knocks = n_knocks
        self.knock_duration = knock_duration
        self.pre_onset = pre_onset
        self.highpass_hz = highpass_hz
        self.min_knock_gap = min_knock_gap
        self.knock_floor_db = knock_floor_db
        self.knock_prominence_db = knock_prominence_db
        self.knock_rel_db = knock_rel_db
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else target_sr / 2
        self.top_db = top_db
        self.time_shift = time_shift
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

        self.knock_samples = int(round(knock_duration * target_sr))
        self.shift_samples = int(round(time_shift * target_sr))
        self.frames_per_knock = 1 + self.knock_samples // hop_length
        self.rng = np.random.default_rng(random_state)
        self.cache = cache
        self._cache: Dict[str, np.ndarray] = {}
        self.short_files: Dict[str, int] = {}  # Các file dò được ít hơn K tiếng gõ -> số tiếng gõ thực tế

    @property
    def output_shape(self) -> Tuple[int, int, int]:
        """Shape cố định của tensor đầu ra `process()`: (1, n_mels, K * frames_per_knock)."""
        return 1, self.n_mels, self.n_knocks * self.frames_per_knock

    def reseed(self, seed: int) -> None:
        """Đặt lại seed. Gọi trong `worker_init_fn` khi DataLoader có nhiều worker,
        nếu không các worker sẽ sinh ra cùng một chuỗi augmentation."""
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Đọc và làm sạch tín hiệu
    # ------------------------------------------------------------------
    def load(self, path: str, sr: Optional[int] = None, mono: bool = True) -> np.ndarray:
        """Đọc file âm thanh ở tần số `sr` (mặc định `target_sr`), kiểu float32.

        Dùng ffmpeg vì libsndfile (soundfile/librosa) không đọc được ALAC trong file .m4a của iPhone.
        ffmpeg chỉ giải mã và resample; việc trộn về mono là lấy trung bình các kênh (bộ trộn mặc định
        của ffmpeg nhân 0.707 cho mỗi kênh nên biên độ có thể vượt 1).

        Returns:
            Mảng [n_samples] nếu `mono=True`, ngược lại [n_samples, n_channels].
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy file âm thanh: {path}")
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-ar", str(sr or self.target_sr),
               "-c:a", "pcm_f32le", "-f", "wav", "-"]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg không giải mã được {path}: {result.stderr.decode(errors='ignore').strip()}")
        y, _ = sf.read(io.BytesIO(result.stdout), dtype="float32", always_2d=True)
        if y.size == 0:
            raise RuntimeError(f"File âm thanh rỗng: {path}")
        return y.mean(axis=1) if mono else y

    def denoise(self, y: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """Lọc high-pass Butterworth bậc 4 rồi chuẩn hóa biên độ đỉnh về 1.

        Chuẩn hóa đỉnh giúp ngưỡng dò tiếng gõ (tính theo dB so với đỉnh) ổn định giữa iPhone và Android.
        """
        sr = sr or self.target_sr
        if self.highpass_hz > 0:
            sos = sps.butter(4, self.highpass_hz, btype="highpass", fs=sr, output="sos")
            y = sps.sosfiltfilt(sos, y)
        return (y / (np.abs(y).max() + 1e-9)).astype(np.float32)

    def frame_energy_db(self, y: np.ndarray) -> Tuple[np.ndarray, int]:
        """Năng lượng RMS theo khung 20 ms, bước 10 ms, đơn vị dB so với khung lớn nhất.

        Returns:
            Tuple (mảng dB theo khung, bước nhảy tính bằng mẫu).
        """
        frame, hop = int(0.02 * self.target_sr), int(0.01 * self.target_sr)
        if len(y) < frame:
            y = np.pad(y, (0, frame - len(y)))
        rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop, center=False)[0] + 1e-9
        return 20 * np.log10(rms / rms.max()), hop

    def detect_knocks(self, y: np.ndarray, drop_edges: bool = True) -> np.ndarray:
        """Dò vị trí các tiếng gõ trên tín hiệu đã `denoise`.

        Tiếng gõ là đỉnh của đường năng lượng dB thỏa mãn đồng thời:
        - cao hơn nền nhiễu (phân vị 20%) ít nhất `knock_floor_db`, nổi bật ít nhất
          `knock_prominence_db` và cách đỉnh trước ít nhất `min_knock_gap` giây;
        - không yếu hơn trung vị mức đỉnh của các tiếng gõ trong file quá `knock_rel_db`
          (loại tiếng gõ hụt, tiếng chạm nhẹ hoặc tiếng động nền);
        - nếu `drop_edges=True`: cửa sổ cắt (kể cả phần dư để dịch thời gian) nằm trọn trong file,
          tránh tiếng gõ bị cụt ở đầu hoặc cuối bản ghi.

        Returns:
            Mảng chỉ số mẫu (sample index) tại đỉnh năng lượng của từng tiếng gõ.
        """
        db, hop = self.frame_energy_db(y)
        noise_floor = np.percentile(db, 20)
        peaks, props = sps.find_peaks(
            db,
            height=max(noise_floor + self.knock_floor_db, -40.0),
            distance=max(1, int(self.min_knock_gap * self.target_sr / hop)),
            prominence=self.knock_prominence_db,
        )
        if len(peaks):
            peaks = peaks[props["peak_heights"] >= np.median(props["peak_heights"]) - self.knock_rel_db]
        samples = peaks * hop + int(0.01 * self.target_sr)  # Tâm khung 20 ms
        if drop_edges:
            start = samples - int(round(self.pre_onset * self.target_sr)) - self.shift_samples
            inside = (start >= 0) & (start + self.knock_samples + 2 * self.shift_samples <= len(y))
            if inside.any():  # Nếu mọi tiếng gõ đều chạm biên thì giữ nguyên và để extract_knocks pad 0
                samples = samples[inside]
        return samples

    def extract_knocks(self, y: np.ndarray, peaks: np.ndarray) -> np.ndarray:
        """Cắt mỗi tiếng gõ thành đoạn dài `knock_samples + 2 * shift_samples`, pad 0 nếu chạm biên.

        Phần dư `shift_samples` ở hai đầu để `prepare_knocks()` dịch ngẫu nhiên khi train
        mà không cần đọc lại file.
        """
        seg_len = self.knock_samples + 2 * self.shift_samples
        pre = int(round(self.pre_onset * self.target_sr)) + self.shift_samples
        padded = np.pad(y, (seg_len, seg_len))
        starts = np.asarray(peaks, dtype=int) - pre + seg_len
        if len(starts) == 0:
            return np.zeros((0, seg_len), dtype=np.float32)
        return np.stack([padded[s: s + seg_len] for s in starts]).astype(np.float32)

    def load_knocks(self, path: str) -> np.ndarray:
        """Đọc file, lọc nhiễu, dò và cắt toàn bộ tiếng gõ. Trả về mảng [N, seg_len].

        Nếu không dò được tiếng gõ nào, lấy đoạn quanh mẫu có biên độ lớn nhất làm một tiếng gõ.
        """
        if self.cache and path in self._cache:
            return self._cache[path]
        y = self.denoise(self.load(path))
        peaks = self.detect_knocks(y)
        if len(peaks) == 0:
            warnings.warn(f"Không dò được tiếng gõ nào trong {os.path.basename(path)}, dùng đoạn có biên độ lớn nhất.")
            peaks = np.array([int(np.argmax(np.abs(y)))])
        knocks = self.extract_knocks(y, peaks)
        if len(knocks) < self.n_knocks:
            self.short_files[path] = len(knocks)
            warnings.warn(
                f"{os.path.basename(path)} chỉ có {len(knocks)} tiếng gõ (< K={self.n_knocks}), sẽ lặp lại tiếng gõ có sẵn."
            )
        if self.cache:
            self._cache[path] = knocks
        return knocks

    # ------------------------------------------------------------------
    # Chọn tiếng gõ và trích xuất Mel-Spectrogram
    # ------------------------------------------------------------------
    def select_knock_indices(self, n_available: int, train: bool = False,
                             rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Chọn K chỉ số tiếng gõ.

        - Train: chọn ngẫu nhiên K tiếng gõ không lặp (giữ thứ tự thời gian).
        - Val/test: lấy K tiếng gõ cách đều nhau trên toàn file, kết quả tất định.
        - Nếu file có ít hơn K tiếng gõ: lặp vòng các tiếng gõ có sẵn.
        """
        k = self.n_knocks
        if n_available < k:
            return np.resize(np.arange(n_available), k)
        if train:
            rng = rng or self.rng
            return np.sort(rng.choice(n_available, size=k, replace=False))
        return np.round(np.linspace(0, n_available - 1, k)).astype(int)

    def prepare_knocks(self, knocks: np.ndarray, train: bool = False,
                       rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Chọn K tiếng gõ và cắt về đúng `knock_samples` (train: dịch ngẫu nhiên ±time_shift). Trả về [K, knock_samples]."""
        rng = rng or self.rng
        idx = self.select_knock_indices(len(knocks), train=train, rng=rng)
        if train and self.shift_samples > 0:
            offsets = rng.integers(0, 2 * self.shift_samples + 1, size=len(idx))
        else:
            offsets = np.full(len(idx), self.shift_samples)
        return np.stack([knocks[i, o: o + self.knock_samples] for i, o in zip(idx, offsets)])

    def knocks_to_logmel(self, knocks: np.ndarray) -> np.ndarray:
        """Tính log-Mel (dB) cho từng tiếng gõ [K, L] rồi ghép theo trục thời gian thành [n_mels, K * frames_per_knock]."""
        mel = librosa.feature.melspectrogram(
            y=knocks, sr=self.target_sr, n_fft=self.n_fft, win_length=self.win_length,
            hop_length=self.hop_length, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax, power=2.0,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max, top_db=self.top_db)  # [K, n_mels, F]
        return np.concatenate(list(log_mel), axis=1).astype(np.float32)

    @staticmethod
    def normalize(log_mel: np.ndarray) -> np.ndarray:
        """Chuẩn hóa từng mẫu về mean 0, std 1."""
        return ((log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)).astype(np.float32)

    def spec_augment(self, mel: torch.Tensor, rng: Optional[np.random.Generator] = None) -> torch.Tensor:
        """SpecAugment: che ngẫu nhiên `num_freq_masks` dải tần và `num_time_masks` dải thời gian.

        Giá trị che là 0, tức giá trị trung bình sau khi chuẩn hóa. Chỉ dùng khi train.

        Args:
            mel: Tensor [..., n_mels, T].
            rng: Bộ sinh số ngẫu nhiên, mặc định dùng `self.rng`.
        """
        rng = rng or self.rng
        mel = mel.clone()
        n_mels, n_frames = mel.shape[-2], mel.shape[-1]
        for _ in range(self.num_freq_masks):
            width = int(rng.integers(0, self.freq_mask_param + 1))
            start = int(rng.integers(0, max(1, n_mels - width + 1)))
            mel[..., start: start + width, :] = 0.0
        for _ in range(self.num_time_masks):
            width = int(rng.integers(0, self.time_mask_param + 1))
            start = int(rng.integers(0, max(1, n_frames - width + 1)))
            mel[..., :, start: start + width] = 0.0
        return mel

    def process_knocks(self, knocks: np.ndarray, train: bool = False,
                       rng: Optional[np.random.Generator] = None) -> torch.Tensor:
        """Từ mảng tiếng gõ của `load_knocks()` -> tensor float32 [1, n_mels, T]."""
        log_mel = self.normalize(self.knocks_to_logmel(self.prepare_knocks(knocks, train=train, rng=rng)))
        mel = torch.from_numpy(log_mel).unsqueeze(0)
        if train:
            mel = self.spec_augment(mel, rng=rng)
        assert tuple(mel.shape) == self.output_shape and mel.dtype == torch.float32
        return mel

    def process(self, path: str, train: bool = False, rng: Optional[np.random.Generator] = None) -> torch.Tensor:
        """Tiền xử lý trọn gói một file âm thanh thành tensor float32 [1, n_mels, T].

        Args:
            path: Đường dẫn file .wav hoặc .m4a.
            train: True để chọn tiếng gõ ngẫu nhiên, dịch thời gian và áp dụng SpecAugment.
            rng: Bộ sinh số ngẫu nhiên riêng (tùy chọn), mặc định dùng `self.rng`.
        """
        return self.process_knocks(self.load_knocks(path), train=train, rng=rng)

    # ------------------------------------------------------------------
    # Tên file, bảng chỉ mục và kiểm tra nhãn
    # ------------------------------------------------------------------
    @classmethod
    def parse_filename(cls, path: str) -> Tuple[int, str, str]:
        """Trích xuất (fruit_id, label, device) từ tên file dạng `durian_<id>_<label>_<device>.<ext>`.

        Nhãn là trạng thái độ chín tại thời điểm gõ, có thể khác cột Ripeness của quả trong CSV.
        """
        m = cls.FILENAME_PATTERN.match(os.path.basename(path))
        if m is None:
            raise ValueError(f"Tên file không đúng định dạng durian_<id>_<label>_<device>.<ext>: {path}")
        return int(m.group(1)), m.group(2).lower(), m.group(3).lower()

    def onset_envelope(self, path: str) -> np.ndarray:
        """Đường onset strength (bước 20 ms) đã chuẩn hóa z-score, dùng để so khớp nhịp gõ giữa hai file."""
        y = self.load(path)
        env = librosa.onset.onset_strength(y=y / (np.abs(y).max() + 1e-9), sr=self.target_sr,
                                           hop_length=int(0.02 * self.target_sr))
        return (env - env.mean()) / (env.std() + 1e-9)

    @staticmethod
    def envelope_xcorr(a: np.ndarray, b: np.ndarray, max_lag: int = 1000) -> float:
        """Tương quan chéo lớn nhất giữa hai onset envelope, xét độ trễ tối đa `max_lag` khung (20 s).

        Chỉ xét các độ trễ mà đoạn chồng lấn dài ít nhất một nửa file ngắn hơn. Hai file ghi
        cùng một phiên gõ (vd iPhone và Android ghi đồng thời) cho giá trị gần 1, khác phiên thì dưới 0.5.
        """
        corr = sps.correlate(a, b, mode="full", method="fft")
        lags = sps.correlation_lags(len(a), len(b), mode="full")
        overlap = np.minimum(len(a), len(b) + lags) - np.maximum(0, lags)
        valid = (np.abs(lags) <= max_lag) & (overlap >= min(len(a), len(b)) // 2)
        return float(np.max(corr[valid] / overlap[valid]))

    def build_file_index(
        self,
        data_root: str,
        splits: Sequence[str] = ("train", "val", "test"),
        sound_dirname: str = "dataset_clean_sound",
        session_threshold: float = 0.6,
        same_day_hours: float = 12.0,
        tz_offset_hours: float = 8.0,
        n_jobs: int = 8,
    ) -> pd.DataFrame:
        """Lập bảng chỉ mục toàn bộ file âm thanh và gắn cờ nhãn nghi sai. Không xóa hay di chuyển file.

        Quy trình:
        1. Đọc nhãn/thiết bị từ tên file; tính md5 để phát hiện bản sao y hệt (mọi thiết bị).
        2. Đối chiếu `metadata.csv` theo kích thước file để biết tên gốc và thời điểm ghi thực sự.
           `iphone_date` là giờ UTC (tag creation_time của Voice Memos), `android_date` là giờ địa phương,
           nên giờ iPhone được cộng `tz_offset_hours` trước khi so sánh.
        3. So khớp nội dung: tương quan onset envelope giữa mọi cặp file của cùng một quả;
           các file có tương quan >= `session_threshold` được gộp thành một phiên ghi.
        4. Nhãn iPhone được tin cậy (kiểm tra thêm hậu tố -UN/-RI/-OR của tên gốc). Nhãn của phiên
           lấy theo file iPhone trong phiên; file Android nhận nhãn của phiên chứa nó.
        5. File Android không khớp phiên nào nhưng ghi cùng ngày với file iPhone cùng nhãn được coi là sạch
           (lần gõ khác trong cùng ngày, độ chín không đổi). Các trường hợp còn lại bị gắn cờ.

        Args:
            data_root: Thư mục `data/split_data`.
            splits: Các tập cần quét.
            sound_dirname: Tên thư mục âm thanh trong mỗi split.
            session_threshold: Ngưỡng tương quan để coi hai file cùng phiên ghi.
            same_day_hours: Độ lệch thời gian tối đa (giờ) để coi là ghi cùng ngày.
            tz_offset_hours: Độ lệch múi giờ của Android so với UTC.
            n_jobs: Số luồng đọc file song song.

        Returns:
            DataFrame, mỗi dòng một file, gồm các cột chính: file, path, split, fruit_id, label, device,
            label_corrected, label_suspect, reason, is_duplicate, duplicate_of, session_id,
            xcorr_pair, gap_hours, original_name, md5, fruit_ripeness, matches_fruit_ripeness.
        """
        rows = []
        for split in splits:
            folder = os.path.join(data_root, split, sound_dirname)
            if not os.path.isdir(folder):
                warnings.warn(f"Không tìm thấy thư mục {folder}, bỏ qua.")
                continue
            for name in sorted(os.listdir(folder)):
                if not self.FILENAME_PATTERN.match(name):
                    continue
                fruit_id, label, device = self.parse_filename(name)
                path = os.path.join(folder, name)
                rows.append({"file": name, "path": path, "split": split, "fruit_id": fruit_id,
                             "label": label, "device": device, "size_bytes": os.path.getsize(path)})
        df = pd.DataFrame(rows)
        if df.empty:
            raise FileNotFoundError(f"Không tìm thấy file âm thanh hợp lệ trong {data_root}")

        # 1. md5 trên toàn bộ file (iPhone-iPhone, Android-Android và iPhone-Android)
        with ThreadPoolExecutor(n_jobs) as ex:
            df["md5"] = list(ex.map(self._md5, df["path"]))

        # 2. Đối chiếu metadata.csv
        df = self._attach_metadata(df, data_root, splits, sound_dirname, tz_offset_hours)

        # 3. So khớp nội dung giữa mọi cặp file của cùng một quả
        with ThreadPoolExecutor(n_jobs) as ex:
            envelopes = dict(zip(df["path"], ex.map(self.onset_envelope, df["path"])))
        df["session_id"], df["xcorr_pair"] = self._assign_sessions(df, envelopes, session_threshold)

        # 4-5. Suy ra nhãn đúng và gắn cờ
        df = self._resolve_labels(df, same_day_hours)

        # Bản sao: giữ một file đại diện cho mỗi nhóm md5 (ưu tiên file có nhãn đúng)
        df["is_duplicate"] = False
        df["duplicate_of"] = ""
        for _, group in df.groupby("md5"):
            if len(group) < 2:
                continue
            ok = group[group["label"] == group["label_corrected"]]
            keep = (ok if len(ok) else group).index[0]
            for idx in group.index.drop(keep):
                df.loc[idx, ["is_duplicate", "duplicate_of"]] = [True, df.loc[keep, "file"]]

        # Nhãn của quả trong sample_split_mapping.csv (phục vụ ghép đa phương thức theo fruit_id)
        mapping_path = os.path.join(data_root, "sample_split_mapping.csv")
        if os.path.exists(mapping_path):
            mapping = pd.read_csv(mapping_path)
            ripeness = dict(zip(mapping["sample_id"], mapping["Ripeness"].str.lower()))
            df["fruit_ripeness"] = df["fruit_id"].map(ripeness)
            df["matches_fruit_ripeness"] = df["label_corrected"] == df["fruit_ripeness"]

        columns = ["file", "path", "split", "fruit_id", "label", "device", "label_corrected", "label_suspect",
                   "reason", "is_duplicate", "duplicate_of", "session_id", "xcorr_pair", "gap_hours",
                   "original_name", "record_time_local", "md5", "size_bytes", "fruit_ripeness",
                   "matches_fruit_ripeness"]
        return df[[c for c in columns if c in df.columns]].sort_values(["split", "fruit_id", "label", "device"]) \
            .reset_index(drop=True)

    @staticmethod
    def get_clean_files(index: pd.DataFrame, split: Optional[str] = None,
                        use_corrected: bool = False) -> pd.DataFrame:
        """Lọc danh sách file sạch từ bảng `build_file_index()`.

        Args:
            index: Bảng chỉ mục.
            split: Tên tập cần lấy; None để lấy toàn bộ.
            use_corrected: False chỉ lấy file không bị gắn cờ. True lấy thêm các file bị gắn cờ nhưng
                đã biết nhãn đúng (`label_corrected`) và không phải bản sao.

        Returns:
            DataFrame đã lọc, có thêm cột `target` là nhãn dùng để huấn luyện (chuỗi) và `y` (chỉ số).
        """
        df = index if split is None else index[index["split"] == split]
        keep = ~df["label_suspect"]
        if use_corrected:
            keep |= df["label_corrected"].notna() & ~df["is_duplicate"]
        out = df[keep & ~df["is_duplicate"]].copy()
        out["target"] = out["label_corrected"]
        out["y"] = out["target"].map(AudioPreprocessor.LABEL_TO_IDX)
        return out.reset_index(drop=True)

    @staticmethod
    def _md5(path: str) -> str:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _attach_metadata(self, df: pd.DataFrame, data_root: str, splits: Iterable[str], sound_dirname: str,
                         tz_offset_hours: float) -> pd.DataFrame:
        """Tìm dòng metadata khớp kích thước file thật để lấy tên gốc và thời điểm ghi (giờ địa phương)."""
        records = []
        for split in splits:
            meta_path = os.path.join(data_root, split, sound_dirname, "metadata.csv")
            if not os.path.exists(meta_path):
                warnings.warn(f"Thiếu {meta_path}, không kiểm tra được thời điểm ghi của tập {split}.")
                continue
            meta = pd.read_csv(meta_path)
            for device in self.DEVICES:
                part = meta[[f"{device}_output", f"{device}_original", f"{device}_date", f"{device}_size_bytes"]]
                part.columns = ["file", "original_name", "record_time", "size_bytes"]
                part = part.assign(split=split).drop_duplicates()
                time = pd.to_datetime(part["record_time"], errors="coerce")
                if hasattr(time.dt, "tz") and time.dt.tz is not None:
                    time = time.dt.tz_localize(None)
                if device == "iphone":
                    time = time + pd.Timedelta(hours=tz_offset_hours)
                records.append(part.assign(record_time_local=time))
        if not records:
            df["original_name"], df["record_time_local"] = np.nan, pd.NaT
            return df
        meta = pd.concat(records).drop_duplicates(["split", "file", "size_bytes"])
        return df.merge(meta[["split", "file", "size_bytes", "original_name", "record_time_local"]],
                        on=["split", "file", "size_bytes"], how="left")

    @staticmethod
    def _assign_sessions(df: pd.DataFrame, envelopes: Dict[str, np.ndarray],
                         threshold: float) -> Tuple[List[str], List[float]]:
        """Gộp các file cùng quả có nhịp gõ trùng nhau thành phiên ghi (thành phần liên thông).

        Trả về mã phiên và tương quan lớn nhất của mỗi file với một file thiết bị kia trong cùng quả.
        """
        session = {}
        best_cross = {}
        for fruit_id, group in df.groupby("fruit_id"):
            paths = group["path"].tolist()
            devices = dict(zip(group["path"], group["device"]))
            parent = {p: p for p in paths}

            def find(p):
                while parent[p] != p:
                    parent[p] = parent[parent[p]]
                    p = parent[p]
                return p

            for i, a in enumerate(paths):
                for b in paths[i + 1:]:
                    c = AudioPreprocessor.envelope_xcorr(envelopes[a], envelopes[b])
                    if devices[a] != devices[b]:
                        best_cross[a] = max(best_cross.get(a, -1.0), c)
                        best_cross[b] = max(best_cross.get(b, -1.0), c)
                    if c >= threshold:
                        parent[find(a)] = find(b)
            roots = sorted({find(p) for p in paths})
            for p in paths:
                session[p] = f"{fruit_id:03d}-S{roots.index(find(p)) + 1}"
        return [session[p] for p in df["path"]], [round(best_cross.get(p, np.nan), 3) for p in df["path"]]

    def _resolve_labels(self, df: pd.DataFrame, same_day_hours: float) -> pd.DataFrame:
        """Suy ra `label_corrected`, `label_suspect`, `reason` cho từng file."""
        df = df.copy()
        df["label_corrected"] = df["label"]
        df["label_suspect"] = False
        df["reason"] = ""
        df["gap_hours"] = np.nan

        # iPhone: kiểm tra hậu tố nhãn trong tên gốc
        for idx, row in df[df["device"] == "iphone"].iterrows():
            m = self.IPHONE_SUFFIX_PATTERN.search(str(row["original_name"]))
            if m and self.SUFFIX_TO_LABEL[m.group(1).lower()] != row["label"]:
                df.loc[idx, ["label_corrected", "label_suspect", "reason"]] = [
                    self.SUFFIX_TO_LABEL[m.group(1).lower()], True,
                    f"Tên gốc '{row['original_name']}' mang nhãn khác tên file",
                ]

        # Nhãn của phiên = nhãn (đã sửa) của file iPhone trong phiên
        iphone = df[df["device"] == "iphone"]
        session_label = iphone.groupby("session_id")["label_corrected"].agg(
            lambda s: s.iloc[0] if s.nunique() == 1 else None)

        for idx, row in df[df["device"] == "android"].iterrows():
            same_label_iphone = iphone[(iphone["fruit_id"] == row["fruit_id"]) & (iphone["label"] == row["label"])]
            if len(same_label_iphone):
                gap = (row["record_time_local"] - same_label_iphone["record_time_local"].iloc[0]).total_seconds() / 3600
                df.loc[idx, "gap_hours"] = round(gap, 2) if pd.notna(gap) else np.nan

            sess_label = session_label.get(row["session_id"])
            if sess_label is not None:  # Android khớp nội dung với một file iPhone
                if sess_label != row["label"]:
                    df.loc[idx, ["label_corrected", "label_suspect", "reason"]] = [
                        sess_label, True, f"Nội dung trùng phiên ghi của file iPhone nhãn '{sess_label}'"]
                # Kế thừa cờ nếu file iPhone cùng phiên bị sửa nhãn
                ip_in_session = iphone[iphone["session_id"] == row["session_id"]]
                if ip_in_session["label_suspect"].any() and not df.loc[idx, "label_suspect"]:
                    df.loc[idx, ["label_corrected", "label_suspect", "reason"]] = [
                        sess_label, True, "Cùng phiên ghi với file iPhone bị sửa nhãn"]
                continue

            gap = df.loc[idx, "gap_hours"]
            if pd.notna(gap) and abs(gap) < same_day_hours:
                df.loc[idx, "reason"] = "Ghi chú: không trùng nội dung nhưng ghi cùng ngày với file iPhone cùng nhãn"
            else:
                gap_txt = f"lệch {gap / 24:+.1f} ngày" if pd.notna(gap) else "không rõ thời điểm"
                df.loc[idx, ["label_corrected", "label_suspect", "reason"]] = [
                    np.nan, True, f"Không trùng nội dung với file iPhone nào và ghi khác ngày ({gap_txt})"]
        return df
