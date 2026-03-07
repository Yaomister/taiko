from dataclasses import dataclass

# Standard CD sample rate
DEFAULT_SAMPLE_RATE = 44100 


@dataclass
class SpectrogramParameters:
    sr: int = 44100
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 80
    patch_frames: int = 64  # width of the CNN