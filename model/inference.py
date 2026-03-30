"""
Runsthe trained Taiko CNN or MLP on an audio file and writes a .tja chart.
 
Usage:
    python inference.py \\
        --audio path/to/song.mp3 \\
        --bpm 140 \\
        --model path/to/model.pth \\
        --out path/to/output.tja \\
        --title "My Song" \\
        --offset 0.0 \\
        --threshold 0.5
 
Arguments:
    --audio (str): Path to input audio file (required)
    --bpm (float): BPM of the song (required). Songs with beats per minute changes will produce inaccurate charts.
    --model (str): Path to trained model checkpoint .pth file (required)
    --out (str): Path to write output .tja file (required)
    --title (str): Song title in TJA header. Default is "Untitled"
    --offset (float): Seconds of silence before music starts. Default is 0.0
    --threshold (float): Minimum confidence to count as a note (0-1). Default is 0.5
                         Higher = fewer notes, fewer false positives.
                         Lower  = more notes, more false positives.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data", "src"))

from spectrogram_utils import (
    load_audio,
    compute_multi_resolution_mel,
    SAMPLE_RATE,
    HOP_SIZE,
    N_MELS,
    CONTEXT_FRAMES,
    CONTEXT_HALF,
)

from cnn import CNN
from mlp import MLP

IN_FEATURES = 3 * 15 * 80  # 3600
SUBDIVISIONS = 16
BEATS_PER_MEASURE = 4
TJA_SINGLE = {1: "1", 2: "2", 3: "3", 4: "4"}
TJA_SPAN_START = {5: "7", 6: "9", 7: "5"}
TJA_SPAN_END = "8"


def load_model(path: str, device: torch.device):
    """
    Loads a trained model.

    Args:
        path: path to .pth file
        device: cpu or cuda

    Returns:
        model: loaded model in eval mode
        model_type: 'cnn' or 'mlp'
    """
    info = torch.load(path, map_location=device, weights_only=False)
    state_dict = info["state_dict"]
    n_classes = info["n_classes"]
    args = info.get("args", {})
    dropout = args.get("dropout", 0.5)

    if "in_features" in info:
        model_type = "mlp"
        in_features = info["in_features"]
        model = MLP(in_features=in_features, out_degree=n_classes, dropout=dropout)
    else:
        model_type = "cnn"
        model = CNN(in_degree=3, out_degree=n_classes, dropout=dropout)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded {model_type} model: {n_classes} classes")
    return model, model_type


def predict_frames(
    model: nn.Module,
    model_type: str,
    audio: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
):
    """
    Runs the model on every frame of the audio and returns class probabilities.

    Args:
        model: loaded model in eval mode
        model_type: 'cnn' or 'mlp'
        audio: raw audio samples as numpy array
        device: cpu or cuda
        batch_size: number of windows to process at once. Default: 64

    Returns:
        all_probs: probabilities for every frame
        centers: list of frame indices corresponding to each row in all_probs
    """

    mel_specs, n_frames = compute_multi_resolution_mel(audio)
    centers = list(range(CONTEXT_HALF, n_frames - CONTEXT_HALF))
    X = np.empty((len(centers), 3, CONTEXT_FRAMES, N_MELS), dtype=np.float32)
    for k, i in enumerate(centers):
        for r, spec in enumerate(mel_specs):
            X[k, r] = spec[i - CONTEXT_HALF : i + CONTEXT_HALF + 1]

    all_probs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            chunk = torch.from_numpy(X[start : start + batch_size]).to(device)
            if model_type == "mlp":
                chunk = chunk.view(chunk.size(0), 1, -1)
            logits = model(chunk)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)
    return all_probs, centers


def postprocess(probs, frame_indices, threshold=0.5, min_gap_frames=3):
    """
    Converts per-frame probabilities into a list of note events.

    Single notes (don, ka, bigDon, bigKa): keeps the first detection in each
    cluster, enforcing a minimum gap of min_gap_frames between same-class hits.

    Held notes (drumroll, bigDrumroll, balloon): merges consecutive frames of
    the same class into one event with a start and end time.

    Args:
        probs: probabilities from predict_frames
        frame_indices: list of frame indices from predict_frames
        threshold: minimum confidence to count as a note. Default is 0.5
        min_gap_frames: minimum frames between detections of the same class. Default is 3

    Returns:
        events: list of dicts sorted by time_ms. Single notes have keys
                {time_ms, type}. Held notes also have {end_time_ms}.
    """
    predictions = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    events = []
    # Don, Ka, BigDon, BigKa
    for cls in [1, 2, 3, 4]:
        mask = (predictions == cls) & (confidences >= threshold)
        idxs = np.where(mask)[0]
        filtered = []
        last = -min_gap_frames - 1
        for i in idxs:
            if i - last >= min_gap_frames:
                filtered.append(i)
                last = i
        for i in filtered:
            frame = frame_indices[i]
            t_ms = frame * HOP_SIZE / SAMPLE_RATE * 1000.0
            events.append({"time_ms": t_ms, "type": cls})

    # Drumroll, BigDrumroll, Balloon
    for cls in [5, 6, 7]:
        in_span = False
        span_start = None
        for k, frame in enumerate(frame_indices):
            if predictions[k] == cls and confidences[k] >= threshold:
                if not in_span:
                    in_span = True
                    span_start = frame
            else:
                if in_span:
                    span_end = frame_indices[k - 1]
                    t_start = span_start * HOP_SIZE / SAMPLE_RATE * 1000.0
                    t_end = span_end * HOP_SIZE / SAMPLE_RATE * 1000.0
                    events.append(
                        {"time_ms": t_start, "end_time_ms": t_end, "type": cls}
                    )
                    in_span = False
        if in_span:
            span_end = frame_indices[-1]
            t_start = span_start * HOP_SIZE / SAMPLE_RATE * 1000.0
            t_end = span_end * HOP_SIZE / SAMPLE_RATE * 1000.0
            events.append({"time_ms": t_start, "end_time_ms": t_end, "type": cls})
    events.sort(key=lambda e: e["time_ms"])
    return events


def write_tja(events, bpm, title, wave, offset, out_path):
    """
    Converts note mapping into a .tja chart file.

    Args:
        events: list of note events from postprocess
        bpm: song BPM used to compute the subdivision grid
        title: song title written into the TJA header
        wave: audio filename written into the TJA header
        offset: seconds of silence before music starts
        out_path: path to write the .tja file
    """
    ms_per_beat = 60000.0 / bpm
    ms_per_sub = ms_per_beat / SUBDIVISIONS
    subs_per_measure = SUBDIVISIONS * BEATS_PER_MEASURE

    grid = {}
    for ev in events:
        cls = ev["type"]
        sub_idx = int(round(ev["time_ms"] / ms_per_sub))
        if cls in TJA_SINGLE:
            grid[sub_idx] = TJA_SINGLE[cls]
        elif cls in TJA_SPAN_START:
            grid[sub_idx] = TJA_SPAN_START[cls]
            end_sub = int(round(ev["end_time_ms"] / ms_per_sub))
            grid[end_sub] = TJA_SPAN_END

    lines = [
        f"TITLE:{title}",
        f"WAVE:{wave}",
        f"BPM:{bpm:.2f}",
        f"OFFSET:{offset:.3f}",
        "",
        "COURSE:Oni",
        "LEVEL:8",
        "",
        "#START",
    ]
    last_sub = max(grid.keys()) if grid else 0
    last_measure = last_sub // subs_per_measure
    for m in range(last_measure + 1):
        measure_str = ""
        for s in range(subs_per_measure):
            sub = m * subs_per_measure + s
            measure_str += grid.get(sub, "0")
        lines.append(measure_str + ",")
    lines.append("#END")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Written to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--bpm", required=True, type=float, help="BPM of the song")
    parser.add_argument("--model", required=True, help="Path to .pth model file")
    parser.add_argument("--out", required=True, help="Output .tja path")
    parser.add_argument("--title", default="Untitled")
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type = load_model(args.model, device)
    audio = load_audio(args.audio)
    print(f"Running inference on {args.audio}...")
    probs, frame_indices = predict_frames(model, model_type, audio, device)
    events = postprocess(probs, frame_indices, threshold=args.threshold)
    print(f"Found {len(events)} note events")
    wave = os.path.basename(args.audio)
    write_tja(events, args.bpm, args.title, wave, args.offset, args.out)


if __name__ == "__main__":
    main()
