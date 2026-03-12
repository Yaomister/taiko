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
    spectrogram_parameters = SpectrogramParameters()
    ds = TrainingDataset(audio_file_dir=track_dir, json_dir=json_dir, spectrogram_parameters=spectrogram_parameters, examples_per_song=4000, hit_fraction=0.5)
    loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    model = MLP(in_features=5120, out_degree=8).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x = x.to(device)
            x = x.view(x.size(0), 1, -1) 
            y = y.to(device)

            loss, logits = model.backprop(x, y, optimizer=optimizer, criterion=criterion)
            running_loss += x.size(0) * loss

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()

            print(f"epoch {epoch}, loss {running_loss/total:.4f}, acc {correct/total:.4f}, samples {total}")

    return model


def main():
    model = train(
        track_dir="data/tracks",
        json_dir="data/track_data",
        epochs=10
    )

    torch.save(model.state_dict(), "model_mlp.pth")

if __name__ == "__main__":
    main()
