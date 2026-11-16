"""Durian knock audio preprocessing module for the Acoustic Modality.

Responsible: Member 2 (Audio Track)

This module defines a single class `AudioPreprocessor`, consolidating all steps
validated in `01_eda_and_preprocess_audio.ipynb`:
- Decode `.wav` (Android, PCM 48 kHz stereo) and `.m4a` (iPhone, ALAC 48 kHz mono) via ffmpeg,
  downmix to mono, and resample to `target_sr`. Both devices pass through the exact same
  decoder/resampler to avoid introducing artificial device differences.
- Apply high-pass filtering and peak amplitude normalization prior to knock detection.
- Each recording is ~30 s long and contains ~30 knocks. Rather than trimming silence and
  padding/truncating the entire file, this class detects each knock onset (filtering out weak
  knocks and boundary-truncated knocks), slices fixed-length segments, and concatenates K knocks.
- Compute Mel-Spectrogram, convert to log-dB, and normalize per sample (zero mean, unit std).
  This removes loudness differences between devices and across stronger/lighter knocks.
- Apply SpecAugment (Time Masking + Frequency Masking) during training only.
- Parse labels from filenames and build an index table flagging suspected labels (`label_suspect`);
  filtering is deferred to MultimodalDataset creation. This class never discards files itself.

Usage example:
    preprocessor = AudioPreprocessor(random_state=42)
    mel = preprocessor.process("durian_002_unripe_iphone.m4a", train=True)   # torch.float32 [1, 64, 306]
    fruit_id, label, device = AudioPreprocessor.parse_filename("durian_002_unripe_iphone.m4a")

    # Build full file index with suspect-label flags, then retrieve clean file list
    index = preprocessor.build_file_index("data/split_data")
    train_files = AudioPreprocessor.get_clean_files(index, split="train")

    # Enable caching when a file is read repeatedly across epochs
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
    """Preprocessor converting durian knock audio into fixed-size Mel-Spectrograms.

    Provides:
    - `process()`: file path -> float32 tensor [1, n_mels, T] (with augmentation when train=True).
    - `load()`, `denoise()`, `detect_knocks()`, `load_knocks()`, `prepare_knocks()`,
      `knocks_to_logmel()`, `normalize()`, `spec_augment()`: sub-steps exposed for visualization
      or building custom pipelines (e.g., AST feature extractor) with identical knock selection.
    - `parse_filename()`: filename -> (fruit_id, label, device).
    - `build_file_index()` / `get_clean_files()`: file index table with suspect-label flags.
    """

    LABELS = ("unripe", "ripe", "overripe")
    LABEL_TO_IDX = {name: idx for idx, name in enumerate(LABELS)}
    DEVICES = ("android", "iphone")
    FILENAME_PATTERN = re.compile(
        r"^durian_(\d{3})_(unripe|ripe|overripe)_(android|iphone)\.(wav|m4a)$", re.IGNORECASE
    )
    # Ripeness suffix in original iPhone filenames, e.g., "8-RI.m4a", "162-2-or.m4a"
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
        """Initialize the audio preprocessor.

        Args:
            target_sr: Resampled sample rate (Hz). 16 kHz preserves frequencies up to 8 kHz, capturing
                virtually all knock energy and matching AST's pretraining sample rate.
            n_knocks: Number of knock segments K concatenated into a single sample.
            knock_duration: Duration of each knock segment (seconds), starting `pre_onset` before peak.
            pre_onset: Lead time before energy peak (seconds) to preserve the full attack transient.
            highpass_hz: High-pass cutoff frequency (removes hum and handling noise); 0 disables filtering.
            min_knock_gap: Minimum spacing between two detected knocks (seconds).
            knock_floor_db: Minimum dB above noise floor required for a valid knock peak.
            knock_prominence_db: Minimum peak prominence (dB) relative to local neighborhood.
            knock_rel_db: Reject knocks weaker than the file's median peak level by more than this dB.
            n_fft: FFT size.
            win_length: STFT window length (samples); 400 samples = 25 ms at 16 kHz.
            hop_length: STFT hop length (samples); 160 samples = 10 ms at 16 kHz.
            n_mels: Number of Mel filterbank bins.
            fmin: Lowest Mel filterbank frequency (Hz).
            fmax: Highest Mel filterbank frequency (Hz); defaults to target_sr / 2.
            top_db: Dynamic range cutoff for log-dB relative to peak.
            time_shift: Random start-time jitter amplitude per knock during training (seconds).
            freq_mask_param: Maximum width of each frequency mask (Mel bins).
            time_mask_param: Maximum width of each time mask (frames).
            num_freq_masks: Number of frequency masks.
            num_time_masks: Number of time masks.
            random_state: Seed for internal RNG (knock sampling, jitter, SpecAugment).
            cache: Cache extracted knock segments per file in RAM across epochs.
        """
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found. Install via `brew install ffmpeg` (macOS) or `apt install ffmpeg`.")
        if n_knocks < 1:
            raise ValueError(f"n_knocks must be >= 1, got: {n_knocks}")

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
        self.short_files: Dict[str, int] = {}  # Files with fewer than K detected knocks -> actual count

    @property
    def output_shape(self) -> Tuple[int, int, int]:
        """Fixed output tensor shape from `process()`: (1, n_mels, K * frames_per_knock)."""
        return 1, self.n_mels, self.n_knocks * self.frames_per_knock

    def reseed(self, seed: int) -> None:
        """Reset RNG seed. Call inside `worker_init_fn` for multi-worker DataLoaders
        so workers do not generate identical augmentation sequences."""
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Signal loading and denoising
    # ------------------------------------------------------------------
    def load(self, path: str, sr: Optional[int] = None, mono: bool = True) -> np.ndarray:
        """Decode audio file at `sr` Hz (default `target_sr`) as float32.

        Uses ffmpeg because libsndfile (soundfile/librosa) cannot decode ALAC in iPhone .m4a files.
        ffmpeg decodes and resamples; mono downmix averages channels (ffmpeg's default mixer
        scales by 0.707 per channel, which can cause amplitudes > 1).

        Returns:
            Array of shape [n_samples] if `mono=True`, otherwise [n_samples, n_channels].
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-ar", str(sr or self.target_sr),
               "-c:a", "pcm_f32le", "-f", "wav", "-"]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to decode {path}: {result.stderr.decode(errors='ignore').strip()}")
        y, _ = sf.read(io.BytesIO(result.stdout), dtype="float32", always_2d=True)
        if y.size == 0:
            raise RuntimeError(f"Empty audio file: {path}")
        return y.mean(axis=1) if mono else y

    def denoise(self, y: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """Apply 4th-order Butterworth high-pass filter and peak-normalize to 1.

        Peak normalization keeps dB knock-detection thresholds consistent across iPhone and Android.
        """
        sr = sr or self.target_sr
        if self.highpass_hz > 0:
            sos = sps.butter(4, self.highpass_hz, btype="highpass", fs=sr, output="sos")
            y = sps.sosfiltfilt(sos, y)
        return (y / (np.abs(y).max() + 1e-9)).astype(np.float32)

    def frame_energy_db(self, y: np.ndarray) -> Tuple[np.ndarray, int]:
        """Compute RMS energy in 20 ms frames (10 ms hop) in dB relative to maximum frame.

        Returns:
            Tuple (per-frame dB array, hop size in samples).
        """
        frame, hop = int(0.02 * self.target_sr), int(0.01 * self.target_sr)
        if len(y) < frame:
            y = np.pad(y, (0, frame - len(y)))
        rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop, center=False)[0] + 1e-9
        return 20 * np.log10(rms / rms.max()), hop

    def detect_knocks(self, y: np.ndarray, drop_edges: bool = True) -> np.ndarray:
        """Detect knock positions on a denoised signal.

        A knock is a peak in the dB energy curve that simultaneously satisfies:
        - higher than the noise floor (20th percentile) by at least `knock_floor_db`, prominent by at
          least `knock_prominence_db`, and separated from the previous peak by at least `min_knock_gap` s;
        - not weaker than the file's median knock peak level by more than `knock_rel_db`
          (rejects missed knocks, light touches, or background thuds);
        - if `drop_edges=True`: the slice window (including time-shift padding) fits strictly inside
          the recording, avoiding truncated knocks at file boundaries.

        Returns:
            Array of sample indices at the energy peak of each detected knock.
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
        samples = peaks * hop + int(0.01 * self.target_sr)  # Center of 20 ms frame
        if drop_edges:
            start = samples - int(round(self.pre_onset * self.target_sr)) - self.shift_samples
            inside = (start >= 0) & (start + self.knock_samples + 2 * self.shift_samples <= len(y))
            if inside.any():  # If all knocks hit boundaries, keep all and let extract_knocks zero-pad
                samples = samples[inside]
        return samples

    def extract_knocks(self, y: np.ndarray, peaks: np.ndarray) -> np.ndarray:
        """Slice each knock into length `knock_samples + 2 * shift_samples`, zero-padding boundaries.

        Extra `shift_samples` margins allow `prepare_knocks()` to apply random time shifts during
        training without re-decoding the file.
        """
        seg_len = self.knock_samples + 2 * self.shift_samples
        pre = int(round(self.pre_onset * self.target_sr)) + self.shift_samples
        padded = np.pad(y, (seg_len, seg_len))
        starts = np.asarray(peaks, dtype=int) - pre + seg_len
        if len(starts) == 0:
            return np.zeros((0, seg_len), dtype=np.float32)
        return np.stack([padded[s: s + seg_len] for s in starts]).astype(np.float32)

    def load_knocks(self, path: str) -> np.ndarray:
        """Load file, denoise, detect and slice all knocks. Returns array [N, seg_len].

        If no knocks are detected, falls back to a segment centered at the peak amplitude sample.
        """
        if self.cache and path in self._cache:
            return self._cache[path]
        y = self.denoise(self.load(path))
        peaks = self.detect_knocks(y)
        if len(peaks) == 0:
            warnings.warn(f"No knocks detected in {os.path.basename(path)}; using segment around peak amplitude.")
            peaks = np.array([int(np.argmax(np.abs(y)))])
        knocks = self.extract_knocks(y, peaks)
        if len(knocks) < self.n_knocks:
            self.short_files[path] = len(knocks)
            warnings.warn(
                f"{os.path.basename(path)} has only {len(knocks)} knocks (< K={self.n_knocks}); cycling available knocks."
            )
        if self.cache:
            self._cache[path] = knocks
        return knocks

    # ------------------------------------------------------------------
    # Knock selection and Mel-Spectrogram extraction
    # ------------------------------------------------------------------
    def select_knock_indices(self, n_available: int, train: bool = False,
                             rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Select K knock indices.

        - Train: sample K distinct knocks randomly (preserving chronological order).
        - Val/Test: select K evenly spaced knocks across the file (deterministic).
        - If fewer than K knocks exist: cycle through available knocks.
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
        """Select K knocks and crop to `knock_samples` (train: random ±time_shift). Returns [K, knock_samples]."""
        rng = rng or self.rng
        idx = self.select_knock_indices(len(knocks), train=train, rng=rng)
        if train and self.shift_samples > 0:
            offsets = rng.integers(0, 2 * self.shift_samples + 1, size=len(idx))
        else:
            offsets = np.full(len(idx), self.shift_samples)
        return np.stack([knocks[i, o: o + self.knock_samples] for i, o in zip(idx, offsets)])

    def knocks_to_logmel(self, knocks: np.ndarray) -> np.ndarray:
        """Compute log-Mel (dB) per knock [K, L] and concatenate along time axis to [n_mels, K * frames_per_knock]."""
        mel = librosa.feature.melspectrogram(
            y=knocks, sr=self.target_sr, n_fft=self.n_fft, win_length=self.win_length,
            hop_length=self.hop_length, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax, power=2.0,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max, top_db=self.top_db)  # [K, n_mels, F]
        return np.concatenate(list(log_mel), axis=1).astype(np.float32)

    @staticmethod
    def normalize(log_mel: np.ndarray) -> np.ndarray:
        """Normalize each sample to zero mean and unit standard deviation."""
        return ((log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)).astype(np.float32)

    def spec_augment(self, mel: torch.Tensor, rng: Optional[np.random.Generator] = None) -> torch.Tensor:
        """SpecAugment: randomly mask `num_freq_masks` frequency bands and `num_time_masks` time intervals.

        Mask fill value is 0 (the post-normalization mean). Used during training only.

        Args:
            mel: Tensor [..., n_mels, T].
            rng: Optional RNG; defaults to `self.rng`.
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
        """Convert pre-extracted knocks from `load_knocks()` -> float32 tensor [1, n_mels, T]."""
        log_mel = self.normalize(self.knocks_to_logmel(self.prepare_knocks(knocks, train=train, rng=rng)))
        mel = torch.from_numpy(log_mel).unsqueeze(0)
        if train:
            mel = self.spec_augment(mel, rng=rng)
        assert tuple(mel.shape) == self.output_shape and mel.dtype == torch.float32
        return mel

    def process(self, path: str, train: bool = False, rng: Optional[np.random.Generator] = None) -> torch.Tensor:
        """End-to-end preprocessing of an audio file into a float32 tensor [1, n_mels, T].

        Args:
            path: Path to .wav or .m4a file.
            train: If True, randomly sample knocks, apply time shift, and apply SpecAugment.
            rng: Optional RNG; defaults to `self.rng`.
        """
        return self.process_knocks(self.load_knocks(path), train=train, rng=rng)

    # ------------------------------------------------------------------
    # Filename parsing, file indexing, and label verification
    # ------------------------------------------------------------------
    @classmethod
    def parse_filename(cls, path: str) -> Tuple[int, str, str]:
        """Extract (fruit_id, label, device) from filename `durian_<id>_<label>_<device>.<ext>`.

        Note: the filename label reflects ripeness at knock recording time, which may differ from
        the fruit's Ripeness column in the CSV.
        """
        m = cls.FILENAME_PATTERN.match(os.path.basename(path))
        if m is None:
            raise ValueError(f"Filename does not match durian_<id>_<label>_<device>.<ext>: {path}")
        return int(m.group(1)), m.group(2).lower(), m.group(3).lower()

    def onset_envelope(self, path: str) -> np.ndarray:
        """Z-score normalized onset strength envelope (20 ms hop) for cross-recording matching."""
        y = self.load(path)
        env = librosa.onset.onset_strength(y=y / (np.abs(y).max() + 1e-9), sr=self.target_sr,
                                           hop_length=int(0.02 * self.target_sr))
        return (env - env.mean()) / (env.std() + 1e-9)

    @staticmethod
    def envelope_xcorr(a: np.ndarray, b: np.ndarray, max_lag: int = 1000) -> float:
        """Maximum cross-correlation between two onset envelopes up to `max_lag` frames (20 s).

        Considers only lags where overlap is at least half the shorter file. Recordings from the
        same knocking session (e.g., iPhone and Android recorded simultaneously) approach 1.0,
        whereas separate sessions are below 0.5.
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
        """Build an index of all audio files and flag suspect labels without deleting or moving files.

        Workflow:
        1. Parse label/device from filename; compute md5 to detect exact duplicates across devices.
        2. Match `metadata.csv` by file size to recover original names and recording timestamps.
           `iphone_date` is UTC (Voice Memos creation_time tag), while `android_date` is local time,
           so `tz_offset_hours` is added to iPhone timestamps before comparison.
        3. Content matching: compute onset-envelope correlation across all file pairs of each fruit;
           pairs with correlation >= `session_threshold` are grouped into one recording session.
        4. iPhone labels are trusted (further verified against -UN/-RI/-OR suffix in original name).
           Session label is inherited from the iPhone file; Android files receive their session's label.
        5. Android files not matching any session but recorded on the same day as an iPhone file with
           the same label are considered clean (different session on same day). Remaining cases are flagged.

        Args:
            data_root: Path to `data/split_data`.
            splits: Splits to scan.
            sound_dirname: Audio directory name inside each split.
            session_threshold: Correlation threshold to consider two recordings from the same session.
            same_day_hours: Maximum time delta (hours) to count as same-day recording.
            tz_offset_hours: Android local timezone offset relative to UTC.
            n_jobs: Number of worker threads for reading files.

        Returns:
            DataFrame with one row per file and columns: file, path, split, fruit_id, label, device,
            label_corrected, label_suspect, reason, is_duplicate, duplicate_of, session_id,
            xcorr_pair, gap_hours, original_name, md5, fruit_ripeness, matches_fruit_ripeness.
        """
        rows = []
        for split in splits:
            folder = os.path.join(data_root, split, sound_dirname)
            if not os.path.isdir(folder):
                warnings.warn(f"Directory {folder} not found; skipping.")
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
            raise FileNotFoundError(f"No valid audio files found in {data_root}")

        # 1. md5 across all files (iPhone-iPhone, Android-Android, and iPhone-Android)
        with ThreadPoolExecutor(n_jobs) as ex:
            df["md5"] = list(ex.map(self._md5, df["path"]))

        # 2. Match against metadata.csv
        df = self._attach_metadata(df, data_root, splits, sound_dirname, tz_offset_hours)

        # 3. Content matching across all file pairs of each fruit
        with ThreadPoolExecutor(n_jobs) as ex:
            envelopes = dict(zip(df["path"], ex.map(self.onset_envelope, df["path"])))
        df["session_id"], df["xcorr_pair"] = self._assign_sessions(df, envelopes, session_threshold)

        # 4-5. Infer corrected labels and assign flags
        df = self._resolve_labels(df, same_day_hours)

        # Duplicates: keep one representative file per md5 group (preferring files with correct labels)
        df["is_duplicate"] = False
        df["duplicate_of"] = ""
        for _, group in df.groupby("md5"):
            if len(group) < 2:
                continue
            ok = group[group["label"] == group["label_corrected"]]
            keep = (ok if len(ok) else group).index[0]
            for idx in group.index.drop(keep):
                df.loc[idx, ["is_duplicate", "duplicate_of"]] = [True, df.loc[keep, "file"]]

        # Fruit ripeness label in sample_split_mapping.csv (for multimodal pairing by fruit_id)
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
        """Filter clean files from the `build_file_index()` table.

        Args:
            index: File index DataFrame.
            split: Split name to filter; None returns all splits.
            use_corrected: If False, keeps only unflagged files. If True, also includes flagged
                files whose corrected label (`label_corrected`) is known and that are not duplicates.

        Returns:
            Filtered DataFrame with added `target` (string label) and `y` (integer class index) columns.
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
        """Find matching metadata row by exact file size to recover original filename and local timestamp."""
        records = []
        for split in splits:
            meta_path = os.path.join(data_root, split, sound_dirname, "metadata.csv")
            if not os.path.exists(meta_path):
                warnings.warn(f"Missing {meta_path}; cannot verify recording timestamps for split {split}.")
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
        """Group recordings of the same fruit with matching knock patterns into sessions (connected components).

        Returns session IDs and each file's maximum cross-correlation with an opposite-device file of the same fruit.
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
        """Infer `label_corrected`, `label_suspect`, and `reason` for each file."""
        df = df.copy()
        df["label_corrected"] = df["label"]
        df["label_suspect"] = False
        df["reason"] = ""
        df["gap_hours"] = np.nan

        # iPhone: check label suffix in original filename
        for idx, row in df[df["device"] == "iphone"].iterrows():
            m = self.IPHONE_SUFFIX_PATTERN.search(str(row["original_name"]))
            if m and self.SUFFIX_TO_LABEL[m.group(1).lower()] != row["label"]:
                df.loc[idx, ["label_corrected", "label_suspect", "reason"]] = [
                    self.SUFFIX_TO_LABEL[m.group(1).lower()], True,
                    f"Original name '{row['original_name']}' has a different label than filename",
                ]

        # Session label = (corrected) label of the iPhone file in the session
        iphone = df[df["device"] == "iphone"]
        session_label = iphone.groupby("session_id")["label_corrected"].agg(
            lambda s: s.iloc[0] if s.nunique() == 1 else None)

        for idx, row in df[df["device"] == "android"].iterrows():
            same_label_iphone = iphone[(iphone["fruit_id"] == row["fruit_id"]) & (iphone["label"] == row["label"])]
            if len(same_label_iphone):
                gap = (row["record_time_local"] - same_label_iphone["record_time_local"].iloc[0]).total_seconds() / 3600
                df.loc[idx, "gap_hours"] = round(gap, 2) if pd.notna(gap) else np.nan

            sess_label = session_label.get(row["session_id"])
            if sess_label is not None:  # Android content matches an iPhone file
                if sess_label != row["label"]:
                    df.loc[idx, ["label_corrected", "label_suspect", "reason"]] = [
                        sess_label, True, f"Content matches recording session of iPhone file labeled '{sess_label}'"]
                # Inherit flag if the paired iPhone file in the same session had its label corrected
                ip_in_session = iphone[iphone["session_id"] == row["session_id"]]
                if ip_in_session["label_suspect"].any() and not df.loc[idx, "label_suspect"]:
                    df.loc[idx, ["label_corrected", "label_suspect", "reason"]] = [
                        sess_label, True, "Same recording session as iPhone file with corrected label"]
                continue

            gap = df.loc[idx, "gap_hours"]
            if pd.notna(gap) and abs(gap) < same_day_hours:
                df.loc[idx, "reason"] = "Note: content does not match, but recorded on the same day as iPhone file with same label"
            else:
                gap_txt = f"diff {gap / 24:+.1f} days" if pd.notna(gap) else "unknown timestamp"
                df.loc[idx, ["label_corrected", "label_suspect", "reason"]] = [
                    np.nan, True, f"Content does not match any iPhone file and recorded on a different day ({gap_txt})"]
        return df
