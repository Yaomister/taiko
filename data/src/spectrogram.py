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

from .spectrogram_utils import NOTE_TYPE_TO_ID, ID_TO_NOTE_TYPE

import numpy as np
from math import ceil
from collections import defaultdict
import psutil, os

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

def export_and_clear_batch(
    batch_X: List[np.ndarray],
    batch_Y: List[np.ndarray],
    batch_num: int,
    out_path: str,
    sample_to_song: List[int],
    song_names: List[str],
):
    """
    Exports a batch to a given output path. Note that this function clears batch_X and batch_Y
    to save memory.
    """
    X_all = np.concatenate(batch_X, axis=0)
    batch_X.clear() 
    y_all = np.concatenate(batch_Y, axis=0)
    batch_Y.clear()
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
    # print(
    #     f"Saved batch {batch_num} to {file_path}, X={X_all.shape} y={y_all.shape} songs={len(song_names)}"
    # )


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
    batch_sample_to_song: List[int] = []
    batch_song_names: List[str] = []

    class_cnts = defaultdict(int) # Count of appearances per class in the dataset
    class_ids = {NoteType.Background.value: 0, **{t.value: NOTE_TYPE_TO_ID[t] for t in allowed_types}}
    n_samples, n_songs = 0, 0

    # Make output directory if it doesn't exist
    os.makedirs(out_path, exist_ok=True)

    pbar = tqdm(song_folders)
    for song_id, folder in enumerate(pbar):
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
        proc = psutil.Process(os.getpid())
        pbar.set_description_str(f"Mem: {proc.memory_info().rss / 1e6:.1f} MB")

        if X.shape[0] == 0:
            # print(f"No samples for {base}, skipping.")
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
            
            for arr in batch_Y:
                for id in arr.flat:
                    class_cnts[id] += 1

            # Display class distribution
            total = sum(class_cnts.values())
            dist = {ID_TO_NOTE_TYPE[int(k)]: f"{v/total:.2%}" for k, v in class_cnts.items()}
            pbar.set_postfix(dist)

            # This clears batch_X and batch_Y, so print the class distribution before
            export_and_clear_batch(
                batch_num=batch_num,
                batch_X=batch_X,
                batch_Y=batch_Y,
                out_path=out_path,
                sample_to_song=batch_sample_to_song,
                song_names=batch_song_names,
            )

            # Reset all batch-related data
            batch_sample_to_song.clear()
            batch_song_names.clear()


    # Save metadata
    metadata = {
        "n_samples": n_samples,
        "n_songs": n_songs,
        "batch_size": batch_size,
        "diff": diff,
        "classes": {str(v): k for k, v in class_ids.items()},
        "class_counts": {ID_TO_NOTE_TYPE[int(k)]: int(v) for k, v in class_cnts.items()},
        "negative_ratio": cfg.negative_ratio,
        "seed": cfg.seed,
        "sample_rate": SAMPLE_RATE,
        "hop_size": HOP_SIZE,
        "window_sizes": list(WINDOW_SIZES),
        "n_mels": N_MELS,
        "per_window_context_frames": CONTEXT_FRAMES,
        "X_shape": "(N, 3, 15, 80)",
        "y_shape": "(N,) multi-class beat label at center frame (0=background)",
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
        diff=args.diff,
        allowed_types=allowed_types,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
