"""
Helpers for the rhythm-game frame classification pipeline (multi-class note type + background).

Frame i is centered on sample i * hop_size (after symmetric zero-padding). Features are
log(mel_power + eps), not raw linear mel power.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import librosa
except ImportError:
    librosa = None

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object

# Constants
SAMPLE_RATE = 44100
HOP_SIZE = 512
WINDOW_SIZES: Tuple[int, ...] = (512, 1024, 2048)
MAX_WINDOW = max(WINDOW_SIZES)
# How much the track is zero-padded by so that the max window fits at the edges of the track
CENTER_PAD_SAMPLES = MAX_WINDOW // 2
N_MELS = 80
CONTEXT_FRAMES = 15
CONTEXT_HALF = CONTEXT_FRAMES // 2
SUPPORTED_AUDIO_TYPES = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
# Same rule as data/src/labels.ts: parent folder basename, then invalid chars → "_"
_LABEL_JSON_STEM_SANITIZE_RE = re.compile(r'[\\/:"*?<>| ]+')


class NoteType(str, Enum):
    Don = "don"
    Ka = "ka"
    BigDon = "bigDon"
    BigKa = "bigKa"
    Drumroll = "drumroll"
    BigDrumroll = "bigDrumroll"
    Balloon = "balloon"


NOTE_TYPE_TO_ID: Dict[str, int] = {
    NoteType.Don.value: 1,
    NoteType.Ka.value: 2,
    NoteType.BigDon.value: 3,
    NoteType.BigKa.value: 4,
    NoteType.Drumroll.value: 5,
    NoteType.BigDrumroll.value: 6,
    NoteType.Balloon.value: 7,
}


def _require_librosa() -> None:
    if librosa is None:
        raise ImportError(
            "spectrogram pipeline requires librosa (pip install librosa)."
        )


def _mel_filter_matrix(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """
    Mel filterbank mapping first (n_fft // 2) linear FFT power bins -> n_mels.
    Shape: (n_mels, n_fft // 2). Rows normalized so typical STFT energy is preserved.
    """
    _require_librosa()
    fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=sr / 2.0)
    # throw away mirror half
    fb = np.asarray(fb[:, : n_fft // 2], dtype=np.float64)
    s = fb.sum(axis=1, keepdims=True)
    fb = fb / (s + 1e-10)
    return fb


# Cache filterbanks per (sr, n_fft, n_mels)
_MEL_FB_CACHE: Dict[Tuple[int, int, int], np.ndarray] = {}


def _get_mel_fb(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    key = (sr, n_fft, n_mels)
    if key not in _MEL_FB_CACHE:
        _MEL_FB_CACHE[key] = _mel_filter_matrix(sr, n_fft, n_mels)
    return _MEL_FB_CACHE[key]


_HANN_WIN_CACHE: Dict[int, np.ndarray] = {}


def _get_hann(window_size: int) -> np.ndarray:
    if window_size not in _HANN_WIN_CACHE:
        _HANN_WIN_CACHE[window_size] = np.hanning(window_size).astype(np.float64)
    return _HANN_WIN_CACHE[window_size]


def _natural_mel_frames_centered(audio_length: int, hop_size: int) -> int:
    """
    Number of hop-centered frames for audio of length audio_length (samples).
    Frame i is centered on sample i * hop_size; symmetric padding of at least window//2
    per side is assumed so the window is always in range for all WINDOW_SIZES.
    """
    if audio_length <= 0 or hop_size <= 0:
        return 0
    return audio_length // hop_size + 1


def _compute_mel_spectrogram_nfr(
    padded_audio: np.ndarray,
    center_offset: int,
    window_size: int,
    nfr: int,  # Number of frames
    sample_rate: int,
    hop_size: int,
    n_mels: int,
) -> np.ndarray:
    """
    Centered STFT-style log-mel: frame i uses samples centered at (center_offset + i * hop).

    padded_audio length must be len(original) + 2 * center_offset; caller must ensure
    nfr (num ber of frames) is consistent with _natural_mel_frames_centered(len(original), hop_size).
    """
    padded_audio = np.asarray(padded_audio, dtype=np.float64).ravel()
    if nfr <= 0:
        return np.zeros((0, n_mels), dtype=np.float32)

    fb = _get_mel_fb(sample_rate, window_size, n_mels)
    win = _get_hann(window_size)
    out = np.empty((nfr, n_mels), dtype=np.float32)
    half_fft = window_size // 2

    for i in range(nfr):
        center = center_offset + i * hop_size
        start = center - window_size // 2
        end = start + window_size
        frame = padded_audio[start:end] * win
        spec = np.fft.rfft(frame, n=window_size)
        power = (np.abs(spec[:half_fft]) ** 2).astype(np.float64)
        mel = fb @ power
        out[i] = np.log(np.maximum(mel, 1e-10)).astype(np.float32)

    return out


def compute_mel_spectrogram(
    audio: np.ndarray,
    window_size: int,
    num_frames: int,
    sample_rate: int = SAMPLE_RATE,
    hop_size: int = HOP_SIZE,
    n_mels: int = N_MELS,
) -> np.ndarray:
    """
    Returns:
      log-mel float32 (nfr, n_mels), nfr <= requested num_frames.
      Centered framing with symmetric zero-pad of window_size//2 per side; Hann window; log mel.
    """
    audio = np.asarray(audio, dtype=np.float64).ravel()
    L = len(audio)
    if L <= 0:
        return np.zeros((0, n_mels), dtype=np.float32)

    half_pad = window_size // 2
    padded = np.pad(audio, (half_pad, half_pad), mode="constant")
    natural = _natural_mel_frames_centered(L, hop_size)
    nfr = min(int(num_frames), natural)
    return _compute_mel_spectrogram_nfr(
        padded, half_pad, window_size, nfr, sample_rate, hop_size, n_mels
    )


def common_num_frames(audio_length: int, hop_size: int = HOP_SIZE) -> int:
    """Shared time-axis length for all WINDOW_SIZES (centered + padded STFT)."""
    return _natural_mel_frames_centered(audio_length, hop_size)


def compute_multi_resolution_mel(
    audio: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    hop_size: int = HOP_SIZE,
    n_mels: int = N_MELS,
) -> Tuple[List[np.ndarray], int]:
    """
    Multi-resolution log-mel for WINDOW_SIZES on one hop grid.

    Frame i is centered on sample i * hop_size in the original waveform; audio is zero-padded
    by CENTER_PAD_SAMPLES on each side so every window fits.

    Returns:
      [mel_512, mel_1024, mel_2048], each (num_frames, n_mels) with num_frames = L // hop + 1.
    """
    audio = np.asarray(audio, dtype=np.float64).ravel()
    L = len(audio)
    if hop_size <= 0:
        raise ValueError(f"hop_size must be positive, got {hop_size}")
    if L <= 0:
        raise ValueError("compute_multi_resolution_mel requires non-empty audio")
    n_common = common_num_frames(L, hop_size)

    padded = np.pad(audio, (CENTER_PAD_SAMPLES, CENTER_PAD_SAMPLES), mode="constant")

    specs: List[np.ndarray] = []
    for w in WINDOW_SIZES:
        specs.append(
            _compute_mel_spectrogram_nfr(
                padded,
                CENTER_PAD_SAMPLES,
                w,
                n_common,
                sample_rate,
                hop_size,
                n_mels,
            )
        )
    return specs, n_common


def _note_time_ms(note: dict) -> Optional[float]:
    """Milliseconds from time_ms or timeMs; None if missing."""
    raw = note.get("time_ms") if note.get("time_ms") is not None else note.get("timeMs")
    if raw is None:
        return None
    return float(raw)


def build_multiclass_labels(
    notes: Sequence[dict],
    num_frames: int,
    class_ids: Dict[str, int],
    *,
    sample_rate: int = SAMPLE_RATE,
    hop_size: int = HOP_SIZE,
) -> np.ndarray:
    """
    Multi-class note-type labels per hop frame (0 = background).

    - Classes are defined by class_ids: mapping note type string -> integer ID.
    - If multiple notes fall into the same frame, the last one in `notes` wins.

    Frame i is centered on time i * hop_size / sample_rate (see mel pipeline).
    frame_index = round(time_seconds * sample_rate / hop_size).

    Args:
      notes: JSON-style objects with time_ms or timeMs, and type
      num_frames: length of label array (must match mel spec time axis)
      class_ids: mapping from note type string -> class index (1..K)
    """
    labels = np.zeros(int(num_frames), dtype=np.int64)
    if num_frames <= 0 or not class_ids:
        return labels

    for note in notes:
        t_ms = _note_time_ms(note)
        if t_ms is None:
            continue
        note_type = str(note.get("type", ""))
        if note_type not in class_ids:
            continue
        t_sec = t_ms / 1000.0
        fi = int(round(t_sec * sample_rate / hop_size))
        if 0 <= fi < num_frames:
            labels[fi] = int(class_ids[note_type])
    return labels


def extract_windows(
    mel_specs: Sequence[np.ndarray],
    labels: np.ndarray,
    *,
    rng: Optional[np.random.Generator] = None,
    negative_ratio: Optional[float] = 1.0,
    max_negatives: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build training samples: X (N, 3, 15, 80), y (N,) multi-class.

    For each valid center i in [CONTEXT_HALF, num_frames - CONTEXT_HALF):
      patch_res = mel_spec[i-7:i+8]  (15, 80)
      X = stack along axis 0 -> (3, 15, 80)
      y = labels[i]

    Balancing: include every i with y[i]!=0 (any beat class); sample background
    frames (y==0) to match negative_ratio * num_pos (default 1:1). If
    negative_ratio is None, keep all background frames.
    """
    if len(mel_specs) != 3:
        raise ValueError("mel_specs must contain 3 spectrograms (512, 1024, 2048).")
    for s in mel_specs:
        if s.ndim != 2 or s.shape[1] != N_MELS:
            raise ValueError(f"Each mel spec must be (T, {N_MELS}), got {s.shape}")

    n_frames = mel_specs[0].shape[0]
    for s in mel_specs[1:]:
        if s.shape[0] != n_frames:
            raise ValueError("All mel spectrograms must share the same num_frames.")

    labels = np.asarray(labels, dtype=np.int64).ravel()
    if labels.shape[0] != n_frames:
        raise ValueError(f"labels length {labels.shape[0]} != num_frames {n_frames}")

    i_lo = CONTEXT_HALF
    i_hi = n_frames - CONTEXT_HALF
    if i_hi <= i_lo:
        return np.zeros((0, 3, CONTEXT_FRAMES, N_MELS), dtype=np.float32), np.zeros(
            (0,), dtype=np.int64
        )

    valid = np.arange(i_lo, i_hi, dtype=np.int64)
    pos_mask = labels[valid] != 0
    pos_idx = valid[pos_mask]
    neg_idx = valid[~pos_mask]

    rng = rng or np.random.default_rng()

    # TODO: right now, we're picking negatives randomly. Model may not be able to
    # classify for harder cases where it's given a frame that's close to an onset
    if negative_ratio is None:
        neg_pick = neg_idx
    else:
        n_pos = len(pos_idx)
        target_neg = (
            int(np.ceil(n_pos * float(negative_ratio))) if n_pos > 0 else len(neg_idx)
        )
        if max_negatives is not None:
            target_neg = min(target_neg, max_negatives)
        if len(neg_idx) == 0:
            neg_pick = neg_idx
        elif target_neg >= len(neg_idx):
            neg_pick = neg_idx
        else:
            neg_pick = rng.choice(neg_idx, size=target_neg, replace=False)

    centers = np.concatenate([pos_idx, neg_pick])
    rng.shuffle(centers)

    X = np.empty((len(centers), 3, CONTEXT_FRAMES, N_MELS), dtype=np.float32)
    y = np.empty((len(centers),), dtype=np.int64)

    for k, i in enumerate(centers):
        i = int(i)
        for r, spec in enumerate(mel_specs):
            X[k, r] = spec[i - CONTEXT_HALF : i + CONTEXT_HALF + 1]
        y[k] = labels[i]

    return X, y


def notes_json_to_onset_times_sec(
    notes: List[dict],
    allowed_types: Optional[Sequence[Union[str, NoteType]]] = None,
) -> np.ndarray:
    """Extract onset times in seconds from rhythm-game JSON (time_ms or timeMs)."""
    if allowed_types is not None:
        allowed = {
            t.value if isinstance(t, NoteType) else str(t) for t in allowed_types
        }
    else:
        allowed = None
    times: List[float] = []
    for note in notes:
        note_type = str(note.get("type", ""))
        if allowed is not None and note_type not in allowed:
            continue
        t_ms = _note_time_ms(note)
        if t_ms is None:
            continue
        times.append(t_ms / 1000.0)
    return np.asarray(times, dtype=np.float64)


@dataclass
class OnsetPipelineConfig:
    sample_rate: int = SAMPLE_RATE
    hop_size: int = HOP_SIZE
    n_mels: int = N_MELS
    negative_ratio: Optional[float] = 1.0  # ~1:1 vs positives; None = all negatives
    seed: int = 0


def pipeline_from_audio(
    audio: np.ndarray,
    notes: List[dict],
    class_ids: Dict[str, int],
    cfg: Optional[OnsetPipelineConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full pipeline: converts audio + note JSON labels to X (N, 3, 15, 80), y (N,) multi-class note type ids.
    """
    cfg = cfg or OnsetPipelineConfig()
    rng = rng or np.random.default_rng(cfg.seed)
    # Create 3 mel spectrograms
    mel_specs, nfr = compute_multi_resolution_mel(
        audio,
        sample_rate=cfg.sample_rate,
        hop_size=cfg.hop_size,
        n_mels=cfg.n_mels,
    )
    labels = build_multiclass_labels(
        notes,
        nfr,
        class_ids=class_ids,
        sample_rate=cfg.sample_rate,
        hop_size=cfg.hop_size,
    )
    return extract_windows(
        mel_specs,
        labels,
        rng=rng,
        negative_ratio=cfg.negative_ratio,
    )


if torch is not None:

    class OnsetSpectrogramDataset(Dataset):
        """
        PyTorch Dataset for frame-wise note-type classification (multi-class; 0 = background).

        __getitem__ returns:
          x: FloatTensor (3, 15, 80) log-mel context
          y: LongTensor scalar class id
        """

        def __init__(self, X: np.ndarray, y: np.ndarray):
            if not torch:
                raise ImportError("OnsetSpectrogramDataset requires PyTorch.")

            if X.ndim != 4 or X.shape[1:] != (3, CONTEXT_FRAMES, N_MELS):
                raise ValueError(
                    f"X must be (N, 3, {CONTEXT_FRAMES}, {N_MELS}), got {X.shape}"
                )
            self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
            self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))

        def __len__(self) -> int:
            return self.X.shape[0]

        def __getitem__(self, idx: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
            return self.X[idx], self.y[idx]
else:

    class OnsetSpectrogramDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError("OnsetSpectrogramDataset requires PyTorch.")


def load_audio(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    _require_librosa()
    y, _ = librosa.load(path, sr=sample_rate, mono=True)
    return np.asarray(y, dtype=np.float32)


def process_song(
    audio_path: str,
    json_path: str,
    cfg: OnsetPipelineConfig,
    rng: np.random.Generator,
    allowed_types: List[NoteType],
) -> Tuple[np.ndarray, np.ndarray]:
    """Process one song -> (X, y) with shapes (N, 3, 15, 80), (N,)."""
    audio = load_audio(audio_path, sample_rate=cfg.sample_rate)
    with open(json_path, "r", encoding="utf-8") as f:
        notes = json.load(f)

    # Class-id mapping for the requested beat types
    class_ids = {t.value: NOTE_TYPE_TO_ID[t.value] for t in allowed_types}
    return pipeline_from_audio(audio, notes, class_ids=class_ids, cfg=cfg, rng=rng)


def parse_tja_title_wave(path: str) -> Tuple[Optional[str], Optional[str]]:
    """Read TITLE and WAVE from TJA header (before #START)."""
    title: Optional[str] = None
    wave: Optional[str] = None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip("\r\n")
                if s.startswith("//"):
                    continue
                if s.startswith("#START"):
                    break
                if ":" not in s:
                    continue
                key, _, val = s.partition(":")
                key_u = key.strip().upper()
                val = val.strip()
                if key_u == "TITLE":
                    title = val
                elif key_u == "WAVE":
                    wave = val
    except OSError:
        pass
    return title, wave


def _audio_filenames_in_folder(folder: str) -> List[str]:
    names: List[str] = []
    for name in os.listdir(folder):
        _, ext = os.path.splitext(name)
        if ext.lower() in SUPPORTED_AUDIO_TYPES:
            names.append(name)
    names.sort()
    return names


def _tja_paths_in_folder(folder: str) -> List[str]:
    paths: List[str] = []
    for name in os.listdir(folder):
        if name.lower().endswith(".tja"):
            paths.append(os.path.join(folder, name))
    paths.sort()
    return paths


def label_file(folder: str) -> str:
    """Finds the label file given a folder name; matches naming convention for labels.ts (uses parent folder name of .tja)."""
    folder_basename = os.path.basename(folder)
    return _LABEL_JSON_STEM_SANITIZE_RE.sub("_", folder_basename).strip()


def get_song_folders(audio_root: str) -> List[str]:
    """Return every directory under audio_root (at any depth) that contains a supported audio file."""
    folders: List[str] = []
    if not os.path.isdir(audio_root):
        return folders
    audio_root = os.path.abspath(audio_root)
    for dirpath, _dirnames, filenames in os.walk(audio_root):
        if any(
            os.path.splitext(name)[1].lower() in SUPPORTED_AUDIO_TYPES
            for name in filenames
        ):
            folders.append(dirpath)
    folders.sort()
    return folders


def get_audio_from_folder(folder: str) -> str:
    names = _audio_filenames_in_folder(folder)
    if not names:
        raise FileNotFoundError(f"No audio file found in folder {folder}")
    if len(names) == 1:
        return os.path.join(folder, names[0])
    tjas = _tja_paths_in_folder(folder)
    if len(tjas) == 1:
        _title, wave = parse_tja_title_wave(tjas[0])
        if wave:
            wbase = os.path.basename(wave).lower()
            for name in names:
                if name.lower() == wbase:
                    return os.path.join(folder, name)
    return os.path.join(folder, names[0])
