import json 
import glob
import os
from dataclasses import dataclass

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from spectrogram import LogMelSpectrogram, load_audio, calculate_ms_per_frame
from cnn import CNN

from note_types import notes_label_to_id, notes_id_to_label



@dataclass
class SpectrogramParameterss:
    sr: int = 44100
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 80
    patch_frames: int = 64  # width of the CNN

    

