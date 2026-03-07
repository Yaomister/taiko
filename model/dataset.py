import torch
import os
from spectrogram import LogMelSpectrogram
from typing import Dict, List, Tuple
from utils import calculate_ms_per_frame, load_audio_file, load_audio_wav, load_json_file, get_ground_truth, slice_spectrogram
from torch.utils.data import Dataset
from config import SpectrogramParameters

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

        self.songs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.song_names : List[str] = []

        ms_per_frame = calculate_ms_per_frame(self.p.sr, self.p.hop_length)

        for folder in self.song_folders:
            audio_path = load_audio_file(folder)

            base = os.path.basename(folder)
            json_path = os.path.join(json_dir, f"{base}.json")
            if not os.path.exists(json_path):
                # the json file doesnt exist
                continue

            wav = load_audio_wav(audio_path, target_sr=self.p.sr)
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
            hit_frames = torch.nonzero(_mapping != 0).squeeze(1).tolist()

            # the proportion of hits vs no hits we're training on
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
    