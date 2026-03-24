"""
Onset detection dataset pipeline for CNN training. Saves batches of a specified size
to out_path/batch_n.npz, and a metadata file out_path/metadata.json containing information
about the dataset.

Each training example:
  - Input: multi-resolution mel spectrogram windows centered at a frame
    X.shape = (3, 15, 80) — stack of [512, 1024, 2048] FFT mel patches
  - Output: binary label (1 = onset at center frame, 0 = no onset)

Audio: mono, 44100 Hz. FTT-style frames: hop 512, window sizes [512, 1024, 2048].

Args:
    audio_dir (str): Root directory containing song folders with audio files.
    json_dir (str): Directory containing JSON label files (one per song).
    out_path (str): Path to output .npz file (will be created/overwritten).
    cfg (OnsetPipelineConfig): Pipeline configuration (sample_rate, hop_size, n_mels, negative_ratio, seed).
    allowed_types (List[NoteType]): Note types to include in training (e.g., [NoteType.Don, NoteType.Ka]).

Usage (from repo root):
    python -m data.src.spectrogram \\
        --audio_dir data/tracks \\
        --json_dir data/preprocessed/labels \\
        --out_path data/preprocessed/train_data

Tensor shapes (L = len(audio), hop = 512, W in {512,1024,2048}):
  - Per resolution: num_frames_W = (L - W) // hop + 1; aligned num_frames = (L - 2048) // hop + 1
  - mel_spec_W: (num_frames, 80)
  - labels: (num_frames,) binary, frame i <-> time i * hop / sr sec; onset at t -> frame int(t*sr/hop)
  - Per sample: X (3, 15, 80), y scalar {0,1}; batch X (N, 3, 15, 80)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

from .spectrogram_utils import NOTE_TYPE_TO_ID

import numpy as np
import json
from math import ceil

from data.src.spectrogram_utils import (
    CONTEXT_FRAMES,
    HOP_SIZE,
    N_MELS,
    NoteType,
    OnsetPipelineConfig,
    SAMPLE_RATE,
    WINDOW_SIZES,
    get_song_folders,
    get_audio_from_folder,
    process_song,
)


def export_batch(
    batch_X: List[np.ndarray],
    batch_Y: List[np.ndarray],
    batch_num: int,
    out_path: str,
    sample_to_song: List[int],
    song_names: List[str],
):
    X_all = np.concatenate(batch_X, axis=0)
    y_all = np.concatenate(batch_Y, axis=0)
    song_index_arr = np.asarray(sample_to_song, dtype=np.int64)

    # Export batch to .npz
    file_path = f"{out_path}/batch_{batch_num}"
    np.savez_compressed(
        file=file_path,
        X=X_all,
        y=y_all,
        sample_to_song=song_index_arr,  # Global sample index to song index in song_names
        song_names=song_names,
    )
    print(
        f"Saved batch {batch_num} to {file_path}, X={X_all.shape} y={y_all.shape} songs={len(song_names)}"
    )


def preprocess_dataset(
    audio_dir: str,
    json_dir: str,
    out_path: str,
    batch_size: int,
    cfg: OnsetPipelineConfig,
    allowed_types: List[NoteType],
) -> None:
    rng = np.random.default_rng(cfg.seed)
    song_folders = get_song_folders(audio_dir)
    if not song_folders:
        raise RuntimeError(f"No song folders found in {audio_dir}")

    # X shape: float32 (N, 3, 15, 80)
    # N samples, 3
    batch_X: List[np.ndarray] = []
    # y shape: int64 (N,), (beat classes)
    batch_Y: List[np.ndarray] = []
    batch_sample_to_song: List[int] = []
    batch_song_names: List[str] = []

    all_song_names: List[str] = []
    class_ids = {t.value: NOTE_TYPE_TO_ID[t] for t in allowed_types}
    n_samples, n_songs = 0, 0

    # Make output directory if it doesn't exist
    os.makedirs(out_path, exist_ok=True)

    for song_id, folder in enumerate(song_folders):
        base = os.path.basename(folder)
        try:
            audio_path = get_audio_from_folder(folder)
        except FileNotFoundError as e:
            print(f"Skipping {base}: {e}")
            continue
        json_path = os.path.join(json_dir, f"{base}.json")
        if not os.path.exists(json_path):
            print(f"Skipping {base}: missing JSON {json_path}")
            continue

        print(f"Processing {base}...")
        # TODO: can this be concurrent?
        X, y = process_song(audio_path, json_path, cfg, rng, allowed_types)
        if X.shape[0] == 0:
            print(f"No samples for {base}, skipping.")
            continue
        batch_X.append(X)
        batch_Y.append(y)
        batch_sample_to_song.extend([song_id] * X.shape[0])
        batch_song_names.append(base)

        n_samples += X.shape[0]
        n_songs += 1

        # Batch on every nth song, and on the last song
        if ((song_id + 1) % batch_size == 0) or song_id == len(song_folders) - 1:
            batch_num = ceil(song_id / batch_size)
            if not batch_X:
                raise RuntimeError("No training samples generated.")
            export_batch(
                batch_num=batch_num,
                batch_X=batch_X,
                batch_Y=batch_Y,
                out_path=out_path,
                sample_to_song=batch_sample_to_song,
                song_names=batch_song_names,
            )

            all_song_names.append(batch_song_names)
            # Reset all batch-related data
            batch_X, batch_Y, batch_sample_to_song, batch_song_names = [], [], [], []

    # Save metadata
    dataset_info = {
        "sample_rate": SAMPLE_RATE,
        "hop_size": HOP_SIZE,
        "window_sizes": list(WINDOW_SIZES),
        "n_mels": N_MELS,
        "context_frames": CONTEXT_FRAMES,
        "X_shape": "(N, 3, 15, 80)",
        "y_shape": "(N,) multi-class beat label at center frame (0=background)",
        "classes": {"0": "background", **{str(v): k for k, v in class_ids.items()}},
    }
    metadata = {
        "config": dataset_info,
        "n_samples": n_samples,
        "n_songs": n_songs,
        "note_type_to_id": class_ids,
    }
    with open(f"{out_path}/metadata.json", "w") as file:
        json.dump(metadata, file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build onset dataset: X (N,3,15,80), y binary."
    )
    parser.add_argument("--audio_dir", type=str, default="data/tracks")
    parser.add_argument(
        "--negative_ratio",
        type=float,
        default=1.0,
        help="Target negatives per positive (~1:1). Use -1 for all negatives.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=50)
    allowed = [n.value for n in NoteType]

    parser.add_argument(
        "--note_types",
        type=str,
        required=True,
        help=f"Comma-separated onset types. Allowed: {allowed}",
    )
    parser.add_argument(
        "--out_path",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--json_dir",
        required=True,
        type=str,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = [t.strip() for t in args.note_types.split(",") if t.strip()]
    if raw:
        try:
            allowed_types = [NoteType(t) for t in raw]
        except ValueError as e:
            raise ValueError(
                f"Unknown note type. Valid: {[n.value for n in NoteType]}"
            ) from e
    else:
        allowed_types = [NoteType.Don, NoteType.Ka]

    neg_ratio: Optional[float]
    if args.negative_ratio < 0:
        neg_ratio = None
    else:
        neg_ratio = args.negative_ratio

    cfg = OnsetPipelineConfig(negative_ratio=neg_ratio, seed=args.seed)
    preprocess_dataset(
        audio_dir=args.audio_dir,
        json_dir=args.json_dir,
        out_path=args.out_path,
        cfg=cfg,
        allowed_types=allowed_types,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
