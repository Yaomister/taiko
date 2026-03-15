import torch
import torchaudio

from config import DEFAULT_SAMPLE_RATE

class LogMelSpectrogram(torch.nn.Module):
    """
    Converts a raw audio .wav into a log mel spectrogram.

    Args:
        sr: sample rate of the audio in hz
        n_ftt: number of audio samples per fft window
        hop_length: number of samples to move for the next window
        n_mels: number of mel frequency bins
        f_min: lowest frequency to include in hz 
        f_max: highest frequency to include in hz
    """
    def __init__(
            self, 
            sr: int = DEFAULT_SAMPLE_RATE,
            n_ftt: int  = 2048,
            hop_length: int = 512,
            n_mels: int = 80,
            # there's no point in including anything under 20 Hz
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
        # convert the sound from linear amplitude to log scale decibles
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)


    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Converts a tensor to a log mel spectrogram.

        Args:
            wav: audio tensor of shape

        Returns:
            log-mel spectrogram of shape (n_mels, T) where T is the number of frames/windows
        """
        # (1, N) -> (n_mels, T)
        # T is the number of frames for the entire song
        # N is the number of sammples
        m = self.mel(wav)
        m_db = self.db(m)
        return m_db.squeeze(0)
    







        






