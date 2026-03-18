import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spectrogram import LogMelSpectrogram
from mlp import MLP  # different: MLP instead of CNN

from dataset import TrainingDataset
from config import SpectrogramParameters


def train(
        track_dir: str,
        json_dir: str,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"

):
    """
    Train the MLP model on taiko beat data.

    Loads audio files from track_dir and their corresponding beat annotations
    from json_dir, builds a dataset, then trains the MLP for the given number of epochs.

    Args:
        track_dir: path to the folder containing song subfolders with audio files
        json_dir:  path to the folder containing per-song JSON annotation files
        epochs:    number of full passes over the dataset
        batch_size: number of samples per gradient update
        lr:        learning rate
        device:    torch device string; defaults to CUDA if available, else CPU

    Returns:
        The trained MLP model (call model.eval() before inference).
    """
    spectrogram_parameters = SpectrogramParameters()

    # Build the dataset
    ds = TrainingDataset(audio_file_dir=track_dir, json_dir=json_dir, spectrogram_parameters=spectrogram_parameters, examples_per_song=4000, hit_fraction=0.5)
    loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    # MLP with attention: projects the flattened 80x64=5120 spectrogram patch into a
    # 128-dim token, applies multi-head self-attention, then classifies into 8 note types:
    # no_hit, don, ka, bigDon, bigKa, drumroll, bigDrumroll, balloon
    model = MLP(in_features=5120, out_degree=8).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # every epoch is a full pass through the dataset
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        # keep track of stats to report per-batch progress
        correct = 0
        total = 0

        # x is the input data, y is the ground truth label index
        for x, y in loader:
            x = x.to(device)
            # Flatten the channel + frequency + time dims into a single sequence token of length 5120
            x = x.view(x.size(0), 1, -1)
            y = y.to(device)

            # Single forward + backward + weight update step
            loss, logits = model.backprop(x, y, optimizer=optimizer, criterion=criterion)

            # Accumulate sample-weighted loss for a true running average across the epoch
            running_loss += x.size(0) * loss

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()

            print(f"epoch {epoch}, loss {running_loss/total:.4f}, acc {correct/total:.4f}, samples {total}")

    return model


def main():
    model = train(
        track_dir="../data/tracks",
        json_dir="../data/track_data",
        epochs=10
    )

    # Persist only the learned weights (not optimizer state) so the model can be
    # reloaded later with MLP(...).load_state_dict(torch.load("model_mlp.pth"))
    torch.save(model.state_dict(), "model_mlp.pth")

if __name__ == "__main__":
    main()
