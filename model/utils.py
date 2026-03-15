import torchaudio
import glob
import json
import torch
import os
from typing import Dict, List
from config import DEFAULT_SAMPLE_RATE

# the mapping of beat names to indices
notes_label_to_id: Dict[str, int] = {
    "no_hit": 0,
    "don": 1,
    "ka": 2,
    "bigDon": 3,
    "bigKa": 4,
    "drumroll": 5,
    "bigDrumroll": 6,
    "balloon": 7,
}

# the mapping of beat indices to names
notes_id_to_label = {v : k for v, k in notes_label_to_id.items()}


def load_audio_wav(path: str, target_sr: int = DEFAULT_SAMPLE_RATE):
    """
    Load the .wav audio file as a tensor at the target sample rate

    Args:
        path: absolute path to the audio file
        target_sr: the target sample rate

    Returns:
        the audio file as a tensor
    """
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        # make sure its mono channel and keep dimensions
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def calculate_ms_per_frame(sr: int, hop_length: int) -> float:
    """
    Calculates how many ms each spectrogram frame represents

    Args:
        sr: the sample rate in hz
        hop_length: number of samples between each frame

    Returns:
        duration of each frame in ms
    """
    return (hop_length / sr) * 1000.0



def load_audio_file(dir: str) -> str:
    """
    Finds and returns the path to the first .wav or .mp3 file in a directory.

    Args:
        dir: absolute path to directory containing the audio file

    Returns:
       the path to the first audio file found
    """
    extensions = ["wav", "mp3"]
    for e in extensions:
        hits = glob.glob(os.path.join(dir, f"*.{e}"))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No audio file found in {dir}")


def get_ground_truth(notes: List[dict], T: int, ms_per_frame: float) -> torch.Tensor:
    """
    Builds a tensor mapping each frame to the beat type in its center

    Args:
        notes: a list of note dicts, each with a timeMs and a type field
        T: the total number of frames in the song spectrogram
        ms_per_frame: the duration of each frame in ms

    Returns:
        label tensor with frame level beat indices 
    """
    mapping = torch.zeros(T, dtype=torch.long)
    for note in notes:
        start_time_ms = note["timeMs"]
        note_type = str(note["type"])
        if not note_type in notes_label_to_id:
            continue
        # find the frame which the beat appears in
        frame_index = int(round(start_time_ms / ms_per_frame))
        if 0 <= frame_index < T:
            mapping[frame_index] = notes_label_to_id[note_type]
    return mapping
    
def slice_spectrogram(spectrogram: torch.Tensor, center: int, patch_frames: int) -> torch.Tensor:
    """
    Slices a patch from the whole spectrogram centered on a given frame.

    Args:
        spectrogram: the spectrogram of the whole song
        center: index of the center frame to slice around
        patch_frames: width of the patch in frames

    Returns:
        the sliced spectrogram patch
    """
    # patch_frames = # of columns of the spectrogram taken at once
    n_mels, T = spectrogram.shape
    half = patch_frames //2
    start = center - half
    end = center + half

    # load empty tensor of a patch of the spectrogram
    patch = torch.zeros((n_mels, patch_frames), dtype=spectrogram.dtype)

    # clamp the start and end to prevent the indices from going out of bounds
    src_start = max(0, start)
    src_end = min(T, end)

    # fill with zeroes if go out of bounds
    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)

    if src_end > src_start:
        patch[:, dst_start:dst_end] = spectrogram[:, src_start: src_end]
    return patch.unsqueeze(0)

