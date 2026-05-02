"""
Runs the trained OSU standard CNN on an audio file and writes a .osu beatmap.

Usage:
    python inference.py \\
        --audio path/to/song.mp3 \\
        --model path/to/model.pt \\
        --out path/to/output.osu \\
        --title "My Song" \\
        --diff "Insane" \\
        --threshold 0.5

Arguments:
    --audio (str): Path to input audio file (required)
    --model (str): Path to trained model checkpoint .pt file (required)
    --out (str): Path to write output .osu file (required)
    --title (str): Song title in .osu metadata. Default is "Untitled"
    --diff (str): Difficulty name in .osu metadata. Default is "Normal"
    --threshold (float): Minimum hit confidence to place a note (0-1). Default is 0.5
    --min_gap_frames (int): Minimum frames between notes to avoid double triggers. Default is 5
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


def load_model(path: str, device: torch.device):
    """Load a trained model checkpoint and return it in eval mode."""
    info = torch.load(path, map_location=device, weights_only=False)
    state_dict = info["state_dict"]
    n_classes = info["n_classes"]
    args = info.get("args", {})
    dropout = args.get("dropout", 0.5)

    model = CNN(in_degree=3, out_degree=1, dropout=dropout)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded model: {n_classes} classes, dropout={dropout}")
    return model


def predict_frames(
    model: nn.Module,
    audio: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
):
    """
    Runs the model on every valid frame of the audio.

    Returns:
        hit_probs:  (N,)    sigmoid confidence that each frame contains a note
        type_probs: (N, 3)  softmax probabilities over [circle, slider, spinner]
        positions:  (N, 2)  predicted normalized (x, y) position for each frame
        frames:     list of frame indices corresponding to each row above
    """
    mel_specs, n_frames = compute_multi_resolution_mel(audio)
    frames = list(range(CONTEXT_HALF, n_frames - CONTEXT_HALF))
    X = np.empty((len(frames), 3, CONTEXT_FRAMES, N_MELS), dtype=np.float32)
    for i, frame in enumerate(frames):
        for r, spec in enumerate(mel_specs):
            X[i, r] = spec[frame - CONTEXT_HALF : frame + CONTEXT_HALF + 1]

    all_hit, all_type, all_pos, all_curve_type, all_curve_cp, all_combo = [], [], [], [], [], []
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            chunk = torch.from_numpy(X[start : start + batch_size]).to(device)
            logits_hit, logits_type, pos, logits_curve_type, logits_curve_directions, logits_combo = model(chunk)
            all_hit.append(torch.sigmoid(logits_hit.squeeze(-1)).cpu().numpy())
            all_type.append(torch.softmax(logits_type, dim=1).cpu().numpy())
            all_pos.append(pos.clamp(0, 1).cpu().numpy())
            all_curve_type.append(logits_curve_type.argmax(dim=1).cpu().numpy())
            all_curve_cp.append(logits_curve_directions.cpu().numpy())
            all_combo.append(torch.sigmoid(logits_combo.squeeze(-1)).cpu().numpy())

    hit_probs = np.concatenate(all_hit, axis=0)
    type_probs = np.concatenate(all_type, axis=0)
    positions = np.concatenate(all_pos, axis=0)
    curve_types = np.concatenate(all_curve_type, axis=0)
    curve_cps = np.concatenate(all_curve_cp, axis=0)
    combo_probs = np.concatenate(all_combo, axis=0)
    return hit_probs, type_probs, positions, curve_types, curve_cps, combo_probs, frames


def postprocess(hit_probs, type_probs, positions, curve_types, curve_cps, combo_probs, centers, threshold=0.5, min_gap_frames=5, seed=42):
    """
    Converts per-frame model outputs into a list of note events.

    Positions use a random walk so consecutive notes are reachable from each other
    rather than teleporting across the screen.

    Returns:
        events: list of dicts sorted by time_ms, each with keys:
                {time_ms (float), type (str: circle/slider/spinner), x (int), y (int)}
    """
    rng = np.random.default_rng(seed)
    events = []
    last_kept_idx = -min_gap_frames - 1
    candidate_idxs = np.where(hit_probs > threshold)[0]

    # Start near center of playfield
    cur_x, cur_y = 256, 192
    prev_time_ms = 0.0
    cursor_speed = 0.5  # px/ms — scales jump with time gap, keeps notes reachable

    for idx in candidate_idxs:
        if idx - last_kept_idx < min_gap_frames:
            continue
        last_kept_idx = idx
        time_ms = centers[idx] * HOP_SIZE / SAMPLE_RATE * 1000.0
        type_idx = int(type_probs[idx].argmax())
        type_str = ["circle", "slider", "spinner"][type_idx]

        # Jump distance proportional to time since last note so cursor can physically reach it
        dt_ms = time_ms - prev_time_ms
        max_jump = min(dt_ms * cursor_speed, 200)
        min_jump = max_jump * 0.3
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(min_jump, max_jump)
        cur_x = int(np.clip(cur_x + dist * np.cos(angle), 30, 482))
        cur_y = int(np.clip(cur_y + dist * np.sin(angle), 30, 354))
        prev_time_ms = time_ms

        events.append({"time_ms": time_ms, "type": type_str, "x": cur_x, "y": cur_y,
                        "curve_type": int(curve_types[idx]), "curve_cp": curve_cps[idx],
                        "new_combo": bool(combo_probs[idx] > 0.5)})
    events.sort(key=lambda e: e["time_ms"])
    return events


def write_osu(events, title, audio_filename, diff, out_path):
    """
    Writes a minimal .osu file from note events.

    .osu HitObject line format:
        x,y,time,type,hitSound,extras
        type bits: 1=circle, 2=slider, 8=spinner
    """
    type_bit = {"circle": 1, "slider": 2, "spinner": 8}
    lines = [
        "osu file format v14",
        "",
        "[General]",
        f"AudioFilename: {audio_filename}",
        "Mode: 0",
        "",
        "[Metadata]",
        f"Title: {title}",
        f"Version: {diff}",
        "",
        "[Difficulty]",
        "HPDrainRate:5",
        "CircleSize:4",
        "OverallDifficulty:5",
        "ApproachRate:9",
        "SliderMultiplier:1.4",
        "SliderTickRate:1",
        "",
        "[TimingPoints]",
        "0,500,4,1,0,100,1,0",
        "",
        "[HitObjects]",
    ]
    for event in events:
        t = int(round(event["time_ms"]))
        x, y, typ = event["x"], event["y"], event["type"]
        type_val = type_bit[typ] | (4 if event["new_combo"] else 0)
        if typ == "circle":
            lines.append(f"{x},{y},{t},{type_val},0,0:0:0:0:")
        elif typ == "slider":
            curve_letter = ["L", "B", "P", "C"][event["curve_type"]]
            cp = event["curve_cp"]
            cp_x = int(np.clip(x + cp[0] * 512, 0, 512))
            cp_y = int(np.clip(y + cp[1] * 384, 0, 384))
            end_x = int(np.clip(x + 80, 0, 512))
            if curve_letter == "L":
                lines.append(f"{x},{y},{t},{type_val},0,L|{end_x}:{y},1,80,0|0,0:0|0:0,0:0:0:0:")
            else:
                lines.append(f"{x},{y},{t},{type_val},0,{curve_letter}|{cp_x}:{cp_y}|{end_x}:{y},1,80,0|0,0:0|0:0,0:0:0:0:")
        elif typ == "spinner":
            lines.append(f"256,192,{t},{type_val},0,{t + 500},0:0:0:0:")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {len(events)} note events to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--model", required=True, help="Path to .pt model file")
    parser.add_argument("--out", required=True, help="Output .osu path")
    parser.add_argument("--title", default="Untitled")
    parser.add_argument("--diff", default="Normal")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min_gap_frames", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model, device)
    audio = load_audio(args.audio)

    print(f"Running inference on {args.audio}...")
    hit_probs, type_probs, positions, curve_types, curve_cps, combo_probs, centers = predict_frames(model, audio, device)

    events = postprocess(hit_probs, type_probs, positions, curve_types, curve_cps, combo_probs, centers, args.threshold, args.min_gap_frames)
    print(f"Found {len(events)} note events")

    audio_filename = os.path.basename(args.audio)
    write_osu(events, args.title, audio_filename, args.diff, args.out)


if __name__ == "__main__":
    main()
