import torch
import torchaudio


# Standard CD sample rate
DEFAULT_SAMPLE_RATE = 44100 

# sample rate = how many sampels exists per second
# n_ftt = how many audio samples you look at at once to analyze frequencies (window length)
# hop_length = how far you move each time you analyze (how far the window moves)
# n_mels = how many rows the spectrogram have 


def load_audio(path: str, target_sr: int = DEFAULT_SAMPLE_RATE):
    """Returns waveform (1, N) at the sample rate"""
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        # make sure its mono channel and keep dimensions
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


class LogMelSpectrogram(torch.nn.Module):
    def __init__(
            self, 
            sr: int = DEFAULT_SAMPLE_RATE,
            n_ftt: int  = 2048,
            hop_length: int = 512,
            n_mels: int = 80,
            f_min: float = 20.0,
            f_max: float | None = None,
    ):
        super().__init__()
        self.sr = sr
        self.hop_length = hop_length

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_ftt,
            n_mels=n_mels,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max if f_max is not None else sr / 2,
            power=2.0
        )
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)


    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # (1, N) -> (n_mels, T)
        m = self.mel(waveform)
        m_db = self.db(m)
        return m_db.squeeze(0)
    
def calculate_ms_per_frame(sr: int, hop_length: int) -> float:
    return (hop_length / sr) * 1000.0




        






