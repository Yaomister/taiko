from dataclasses import dataclass

# standard CD sample rate
DEFAULT_SAMPLE_RATE = 44100 


@dataclass
class SpectrogramParameters:
    sr: int = DEFAULT_SAMPLE_RATE
    # the size of the window to take for the fourier transformation (in samples)
    n_fft: int = 2048
    # the distance between each window (in samples)
    hop_length: int = 512
    # the number of frequency bins in the spectrograms (rows)
    n_mels: int = 80
    # the number of time frames in the spectrogram (columns)
    patch_frames: int = 64  # width of the CNN