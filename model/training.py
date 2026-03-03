import json 
import glob
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from spectrogram import LogMelSpectrogram, load_audio, calculate_ms_per_frame
from cnn import CNN

from note_types import notes_label_to_id, notes_id_to_label



@dataclass
class SpectrogramParameters:
    sr: int = 44100
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 80
    patch_frames: int = 64  # width of the CNN


def load_audio_file(dir: str) -> str:
    extensions = ["wav", "mp3", "ogg", "flac", "m4a"]
    for e in extensions:
        hits = glob.glob(os.path.join(dir), f"*.{e}")
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No audio file found in {dir}")


def load_json_file(dir: str) -> List[dict]:
    with open(dir, "r", encoding="utf-8") as f:
        return json.load(f)

    


def get_ground_truth(notes: List[dict], T: int, ms_per_frame: float) -> torch.Tensor:
    mapping = torch.zeros(T, dtype=torch.long)
    for note in notes:
        start_time_ms = note["timeMs"]
        note_type = str(note["type"])
        if not note_type in notes_label_to_id:
            continue
        frame_index = int(round(start_time_ms / ms_per_frame))
        if 0 <= frame_index < T:
            mapping[frame_index] = notes_label_to_id[note_type]
        return mapping
    
def slice_spectrogram(spectrogram: torch.Tensor, center: int, patch_frames: int) -> torch.Tensor:
    # patch_frames = # of columns of the spectrogram taken at once
    n_mels, T = spectrogram.shape
    half = patch_frames //2
    start = center -half
    end = center + half

    patch = torch.zeros((n_mels, patch_frames), dtype=spectrogram.dtype)

    # clamp the start and end
    src_start = max(0, start)
    src_end = min(T, end)

    # fill with zeroes if go out of bounds
    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)

    if src_end > src_start:
        patch[:, dst_start:dst_end] = spectrogram[:, dst_start: dst_end]
    return patch.unsqueeze(0)






def train(audio_files_dir: str, json_files_dir: str, epochs: int = 10, batch_size: int = 64, learning_rate = 1e-3, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    parameters = SpectrogramParameters()

    model = CNN().to(device)

