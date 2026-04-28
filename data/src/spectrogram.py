"""
OSU standard (mode 0) dataset pipeline for CNN training. Reads song folders containing .osu
files and audio, processes each into spectrogram windows with hit type and position labels,
and saves batches to out_path/batch_n.npz plus a metadata file out_path/metadata.json.

Tensor shapes: X (N, 3, 15, 80), y (N,) hit type class id, y_pos (N, 2) normalized (x, y),
weights (N,) per-sample loss weights.

Usage:
    --audio_dir data/tracks \\
    --json_dir data/labels/insane \\
    --out_path data/preprocessed/insane \\
    --note_types "circle,slider,spinner" \\
    --diff insane \\
    --batch_size 50 \\
    --negative_percentage 0.5 \\
    --seed 0

Arguments:
    --audio_dir (str): Path to directory containing song folders with audio files.
    Default: "data/tracks"

    --json_dir (str): Path to directory containing JSON label files produced by parse_osu.py
    (required). Each JSON file is named after the song folder and contains hit_objects with
    time_ms, type, x, y fields.

    --out_path (str): Path to output directory for saving batch files and metadata (required).
    Creates batch_0.npz, batch_1.npz, ... and metadata.json

    --note_types (str): Comma-separated hit object types to include (required).
    Valid values: circle, slider, spinner
    Example: "circle,slider,spinner"

    --diff (str): Difficulty tier label (e.g. "insane"). Used only for metadata — filter
    your .osu files by difficulty before building the dataset, one difficulty per dataset.

    --batch_size (int): Number of songs to accumulate before flushing to disk.
    Default: 50

    --negative_percentage (float): Fraction of total samples that are background (0.33 = 33%).
    Use -1 to include all background samples.
    Default: 0.5

    --seed (int): Random seed for reproducibility.
    Default: 0

    --hard_negative_radius (int): Prefer background samples within this many frames of a hit.
    Set to -1 to disable. Default: 60

    --onset_weight_radius (int): Background frames within this many frames of a hit onset get
    linearly reduced loss weight (weight = dist / radius). Hit frames always get weight 1.0.
    Set to 0 to disable. Default: 4
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
    batch_W: List[np.ndarray],
    batch_Y_pos: List[np.ndarray],
    batch_num: int,
    out_path: str,
):
    """
    Exports a batch to a given output path. Note that this function clears all batch lists
    to save memory.
    """
    X_all = np.concatenate(batch_X, axis=0)
    batch_X.clear()
    y_all = np.concatenate(batch_Y, axis=0)
    batch_Y.clear()
    w_all = np.concatenate(batch_W, axis=0)
    batch_W.clear()
    y_pos_all = np.concatenate(batch_Y_pos, axis=0)
    batch_Y_pos.clear()

    # Export batch to .npz
    file_path = f"{out_path}/batch_{batch_num}"
    np.savez_compressed(
        file=file_path,
        X=X_all,
        y=y_all,
        weights=w_all,
        y_pos=y_pos_all,
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

    # Accumulators for the current batch; flushed to .npz every batch_size songs
    batch_X: List[np.ndarray] = []      # (N, 3, 15, 80) spectrogram windows
    batch_Y: List[np.ndarray] = []      # (N,) hit type class ids
    batch_W: List[np.ndarray] = []      # (N,) per-sample loss weights
    batch_Y_pos: List[np.ndarray] = []  # (N, 2) normalized (x, y) positions
    batch_n_songs = 0

    class_cnts = Counter()  # running count of samples per class across all songs processed so far
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

        X, y, weights, y_pos = process_song(audio_path, json_path, cfg, rng, allowed_types)
        if X.shape[0] == 0:
            # print(f"No samples for {base}, skipping.")
            continue

        id_to_name = {v: k for k, v in class_ids.items()}
        unique, counts = np.unique(y, return_counts=True)
        class_cnts.update(dict(zip(unique.tolist(), counts.tolist())))
        total = sum(class_cnts.values())
        dist = {
            id_to_name.get(int(k), str(k)): f"{v / total:.2%}"
            for k, v in class_cnts.items()
        }
        pbar.set_postfix(dist)

        batch_X.append(X)
        batch_Y.append(y)
        batch_W.append(weights)
        batch_Y_pos.append(y_pos)
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
                batch_W=batch_W,
                batch_Y_pos=batch_Y_pos,
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
            batch_W=batch_W,
            batch_Y_pos=batch_Y_pos,
            out_path=out_path,
        )

    # Save metadata
    metadata = {
        "n_samples": n_samples,
        "n_songs": n_songs,
        "batch_size": batch_size,
        "diff": diff,
        "classes": {str(v): k for k, v in class_ids.items()},
        "class_counts": {
            ID_TO_NOTE_TYPE[int(k)]: int(v) for k, v in class_cnts.items()
        },
        "negative_percentage": cfg.negative_percentage,
        "hard_negative_radius": cfg.hard_negative_radius,
        "onset_weight_radius": cfg.onset_weight_radius,
        "seed": cfg.seed,
        "sample_rate": SAMPLE_RATE,
        "hop_size": HOP_SIZE,
        "window_sizes": list(WINDOW_SIZES),
        "n_mels": N_MELS,
        "per_window_context_frames": CONTEXT_FRAMES,
        "X_shape": "(N, 3, 15, 80)",
        "y_shape": "(N,) hit type at center frame (0=background, 1=circle, 2=slider, 3=spinner)",
        "y_pos_shape": "(N, 2) normalized (x, y) position; (0,0) for background frames",
    }
    with open(f"{out_path}/metadata.json", "w") as file:
        json.dump(metadata, file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build OSU standard dataset: X (N,3,15,80), y hit type, y_pos (N,2) position."
    )
    parser.add_argument("--audio_dir", type=str, default="data/tracks")
    parser.add_argument(
        "--diff",
        type=str,
        help="Difficulty level of songs in this dataset.",
    )
    parser.add_argument(
        "--negative_percentage",
        type=float,
        default=0.5,
        help="Fraction of total samples that are background (0.33 = 33%%). Use -1 for all negatives.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument(
        "--hard_negative_radius",
        type=int,
        default=60,
        help="Sample negatives within this many frames of a note event (~0.7s at 44100/512). Set to -1 to disable.",
    )
    parser.add_argument(
        "--onset_weight_radius",
        type=int,
        default=4,
        help="Background frames within this many frames of a note onset get linearly reduced loss weight (weight = dist / radius). Set to 0 to disable.",
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
        allowed_types = [NoteType.Circle, NoteType.Slider, NoteType.Spinner]

    neg_ratio: Optional[float]
    if args.negative_percentage < 0:
        neg_ratio = None
    else:
        neg_ratio = args.negative_percentage

    hard_neg_radius: Optional[int] = (
        None if args.hard_negative_radius < 0 else args.hard_negative_radius
    )
    cfg = OnsetPipelineConfig(
        negative_percentage=neg_ratio,
        seed=args.seed,
        hard_negative_radius=hard_neg_radius,
        onset_weight_radius=args.onset_weight_radius,
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
