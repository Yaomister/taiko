"""
Onset detection dataset pipeline for CNN training. Saves batches of a specified size
to out_path/batch_n.npz, and a metadata file out_path/metadata.json containing information
about the dataset.

Tensor shapes: X (3, 15, 80), y scalar {0,1}; batch X (N, 3, 15, 80)

Usage:
    --out_path data/preprocessed/train_data \\
    --note_types "Don,Ka" \\
    --batch_size 50 \\
    --negative_ratio 1.0 \\
    --seed 0

Arguments:
    --audio_dir (str): Path to directory containing song folders with audio files.
    Default: "data/tracks"
    
    --json_dir (str): Path to directory containing JSON label files (required).
    Each JSON file should correspond to an audio file with the same base name.
    
    --out_path (str): Path to output directory for saving batch files and metadata (required).
    Creates batch_0.npz, batch_1.npz, ... and metadata.json
    
    --note_types (str): Comma-separated list of onset types to include (required).
    Valid values: Don, Ka, Shaker
    Example: "Don,Ka"
    
    --batch_size (int): Number of songs per batch before saving.
    Default: 50
    
    --negative_ratio (float): Target ratio of negative to positive samples per song.
    Use -1 to include all negative samples.
    Default: 1.0
    
    --seed (int): Random seed for reproducibility.
    Default: 0
    
    --diff (str): Difficulty level of the songs.
"""

from __future__ import annotations
from tqdm import tqdm

import argparse
import json
import os
from typing import List, Optional

import numpy as np
from collections import Counter
import psutil

from data.src.spectrogram_utils import (
    CONTEXT_FRAMES,
    HOP_SIZE,
    ID_TO_NOTE_TYPE,
    N_MELS,
    NOTE_TYPE_TO_ID,
    NoteType,
    OnsetPipelineConfig,
    SAMPLE_RATE,
    WINDOW_SIZES,
    get_song_folders,
    get_audio_from_folder,
    process_song,
)


def export_and_clear_batch(
    batch_X: List[np.ndarray],
    batch_Y: List[np.ndarray],
    batch_num: int,
    out_path: str,
):
    """
    Exports a batch to a given output path. Note that this function clears batch_X and batch_Y
    to save memory.
    """
    X_all = np.concatenate(batch_X, axis=0)
    batch_X.clear()
    y_all = np.concatenate(batch_Y, axis=0)
    batch_Y.clear()

    # Export batch to .npz
    file_path = f"{out_path}/batch_{batch_num}"
    np.savez_compressed(
        file=file_path,
        X=X_all,
        y=y_all,
    )


def preprocess_dataset(
    audio_dir: str,
    json_dir: str,
    out_path: str,
    diff: str,
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
    batch_n_songs = 0

    class_cnts = Counter()  # Count of appearances per class in the dataset
    class_ids = {
        NoteType.Background.value: 0,
        **{t.value: NOTE_TYPE_TO_ID[t] for t in allowed_types},
    }
    n_samples, n_songs = 0, 0
    batch_num = 0

    # Make output directory if it doesn't exist
    os.makedirs(out_path, exist_ok=True)

    pbar = tqdm(song_folders)
    for song_id, folder in enumerate(pbar):
        proc = psutil.Process(os.getpid())
        pbar.set_description_str(f"Mem: {proc.memory_info().rss / 1e6:.1f} MB")

        base = os.path.basename(folder)

        try:
            audio_path = get_audio_from_folder(folder)
        except FileNotFoundError as e:
            # print(f"Skipping {base}: {e}")
            continue
        json_path = os.path.join(json_dir, f"{base}.json")
        if not os.path.exists(json_path):
            # print(f"Skipping {base}: missing JSON {json_path}")
            continue

        X, y = process_song(audio_path, json_path, cfg, rng, allowed_types)
        if X.shape[0] == 0:
            # print(f"No samples for {base}, skipping.")
            continue

        # Print class distribution. When smoothing is on (smooth_radius > 0), recover hard labels from
        # the integer class_ids mapping by rounding, so the display shows real note types.
        id_to_name = {v: k for k, v in class_ids.items()}
        hard_y = np.round(y).astype(np.int64) if y.dtype.kind == "f" else y
        unique, counts = np.unique(hard_y, return_counts=True)
        class_cnts.update(dict(zip(unique.tolist(), counts.tolist())))
        total = sum(class_cnts.values())
        dist = {
            id_to_name.get(int(k), str(k)): f"{v / total:.2%}"
            for k, v in class_cnts.items()
        }
        pbar.set_postfix(dist)

        batch_X.append(X)
        batch_Y.append(y)
        batch_n_songs += 1

        n_samples += X.shape[0]
        n_songs += 1

        # Batch on every nth song, and on the last song
        if batch_n_songs >= batch_size or song_id == len(song_folders) - 1:
            if not batch_X:
                continue

            export_and_clear_batch(
                batch_num=batch_num,
                batch_X=batch_X,
                batch_Y=batch_Y,
                out_path=out_path,
            )
            batch_num += 1
            batch_n_songs = 0

    # Export any remaining samples (if the last song got skipped)
    if batch_X:
        export_and_clear_batch(
            batch_num=batch_num,
            batch_X=batch_X,
            batch_Y=batch_Y,
            out_path=out_path,
        )

    # Save metadata
    metadata = {
        "n_samples": n_samples,
        "n_songs": n_songs,
        "batch_size": batch_size,
        "diff": diff,
        "smooth_radius": cfg.smooth_radius,
        "classes": {str(v): k for k, v in class_ids.items()},
        "class_counts": {
            ID_TO_NOTE_TYPE[int(k)]: int(v) for k, v in class_cnts.items()
        },
        "negative_ratio": cfg.negative_ratio,
        "seed": cfg.seed,
        "sample_rate": SAMPLE_RATE,
        "hop_size": HOP_SIZE,
        "window_sizes": list(WINDOW_SIZES),
        "n_mels": N_MELS,
        "per_window_context_frames": CONTEXT_FRAMES,
        "X_shape": "(N, 3, 15, 80)",
        "y_shape": "(N,) beat label at center frame (0=background)",
    }
    with open(f"{out_path}/metadata.json", "w") as file:
        json.dump(metadata, file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build onset dataset: X (N,3,15,80), y binary."
    )
    parser.add_argument("--audio_dir", type=str, default="data/tracks")
    parser.add_argument(
        "--diff",
        type=str,
        help="Difficulty level of songs in this dataset.",
    )
    parser.add_argument(
        "--negative_ratio",
        type=float,
        default=1.0,
        help="Target negatives per positive (~1:1). Use -1 for all negatives.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument(
        "--hard_negative_radius",
        type=int,
        default=60,
        help="Sample negatives within this many frames of a note event (~0.7s at 44100/512). Set to -1 to disable.",
    )
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
    parser.add_argument(
        "--smooth_radius",
        type=int,
        default=3,
        help=(
            "Half-width (frames) of the Gaussian soft-label halo around each onset. "
            "Weight at distance d is exp(-0.5*(d/sigma)^2) with sigma=radius/sqrt(2*ln(10)), "
            "giving ~0.1 at the boundary. Default 3 (~35 ms each side at 44100/512)."
        ),
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

    hard_neg_radius: Optional[int] = (
        None if args.hard_negative_radius < 0 else args.hard_negative_radius
    )
    cfg = OnsetPipelineConfig(
        negative_ratio=neg_ratio,
        seed=args.seed,
        hard_negative_radius=hard_neg_radius,
        smooth_radius=args.smooth_radius,
    )
    preprocess_dataset(
        audio_dir=args.audio_dir,
        json_dir=args.json_dir,
        out_path=args.out_path,
        cfg=cfg,
        diff=args.diff,
        allowed_types=allowed_types,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
