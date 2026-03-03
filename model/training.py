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

from typing import Dict, List, Tuple

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



class TrainingDataset(Dataset):

    def __init__(
        self,
        audio_file_dir: str,
        json_dir: str,
        spectrogram_parameters: SpectrogramParameters,
        examples_per_song: int = 4000,
        hit_fraction: float = 0.5,
    ):
        self.tracks_dir = audio_file_dir
        self.json_dir = json_dir
        self.p = spectrogram_parameters
        self.examples_per_song = examples_per_song
        self.hit_fraction = hit_fraction

        self.spec = LogMelSpectrogram(sr=self.p.sr, n_ftt=self.p.n_fft, hop_length=self.p.hop_length, n_mels=self.p.n_mels)

        self.song_folders = [
            os.path.join(self.tracks_dir, d) for d in os.listdir(self.tracks_dir) if os.path.isdir(os.path.join(self.tracks_dir, d))
        ]

        self.songs = List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.song_names = List[str]

        ms_per_frame = calculate_ms_per_frame(self.p.sr, self.p.hop_length)

        for folder in self.song_folders:
            audio_path = load_audio_file(folder)

            base = os.path.basename(folder)
            json_path = os.path.join(json_dir, f"{base}.json")
            if not os.path.exists(json_path):
                # the json file doesnt exist
                continue

            wav = load_audio(audio_path, target_sr=self.p.sr)
            with torch.no_grad():
                spectrogram = self.spec(wav)
            
            T = spectrogram.shape[1]
            notes = load_json_file(json_path)

            mapping = get_ground_truth(notes=notes, T=T, ms_per_frame=ms_per_frame)

            self.songs.append((spectrogram, mapping))
            self.song_names.append(base)

        if not self.songs:
            raise RuntimeError("no songs loaded")
        

        # build an index of (song_id, center_frame) samples

        self.sample_indexes: List[Tuple[int, int]] = []
        g = torch.Generator().manual_seed(0)


        for index, (_spectrogram, _mapping) in enumerate(self.songs):
            T = _spectrogram.shape[1]
            hit_frames = torch.nonzero(_mapping != 0).squeeze(0).tolist()
            all_frames = torch.zeros(len(_mapping))

            n = self.examples_per_song
            n_hit = int(n * self.hit_fraction)
            n_no_hit = n - n_hit


            if hit_frames:
                hit_choices = torch.randint(0, len(hit_frames), (n_hit, ), generator=g)
                for c in hit_choices:
                    self.sample_indexes.append((index,hit_frames[c]))
            else:
                n_no_hit = n


            tries = 0
            added = 0
            while added < n_no_hit and tries < n_no_hit * 10:
                center = int(torch.randint(0, T, (1,), generator=g).item())
                if int(_mapping[center].item() == 0):
                    self.sample_indexes.append((index, center))
                    added  = added + 1

                tries = tries + 1

            perm = torch.randperm(len(self.sample_indexes), generator=g).tolist()
            self.sample_indexes = [self.sample_indexes[i] for i in perm]


    def __len__(self) -> int:
        return len(self.sample_indexes)
    
    def __getitem__(self, index: int):
        song_index, center = self.sample_indexes[index]
        spectrogram, mapping = self.songs[song_index]
        slice = slice_spectrogram(spectrogram=spectrogram, center=center, patch_frames=self.p.patch_frames)
        label = mapping[center]
        return slice.float(), label
    



def train(
        track_dir: str,
        json_dir: str,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str = "cude" if  torch.cuda.is_available() else "cpu"

):
    spectrogram_parameters = SpectrogramParameters()
    ds = TrainingDataset(track_dir=track_dir, json_dir=json_dir, spectrogram_parameters=spectrogram_parameters, examples_per_song=4000, hit_fraction=0.5)
    loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    model = CNN(in_degree=1, out_degree=8).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            loss, logits = model.backprop(x, y, optimizer=optimizer, criterion=criterion)
            running_loss += x.size(0) *  loss


            preds = logits.arg_max(dim = 1)
            correct += (preds == y).sum().item()
            total += y.numel()

            print(f"epoch {epoch}, loss {running_loss/total:.4f}, acc {correct/total:.4f}, samples {total}")

    return model