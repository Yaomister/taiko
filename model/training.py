import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cnn import CNN
from dataset import TrainingDataset
from config import SpectrogramParameters



def train(
        track_dir: str,
        json_dir: str,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str = "cuda" if  torch.cuda.is_available() else "cpu"

):
    """
    Trains the CNN on spectrogram patches extracted from a dataset of song.

    Builds the dataset and dataloader, initializes the model, optimizer, and loss
    function, then runs the training loop.

    Args:
        track_dir: absolute path to directory with the audio files
        json_dir: absolute path to directory containing .json beat mapping files
        epochs: the number of full passes through the dataset
        batch_size: the number of samples per training batch
        lr: the learning rate
        device: the device to train on, defaults to gpu if available, otherwise cpu

    Returns:
        the trained CNN model
    """
    spectrogram_parameters = SpectrogramParameters()
    # load the training dataset
    ds = TrainingDataset(audio_file_dir=track_dir, json_dir=json_dir, spectrogram_parameters=spectrogram_parameters, examples_per_song=4000, hit_fraction=0.5)
    loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    # load the CNN model
    model = CNN(in_degree=1, out_degree=8).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        # keep track of stats to print when training is done
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            loss, logits = model.backprop(x, y, optimizer=optimizer, criterion=criterion)
            running_loss += x.size(0) *  loss

            preds = logits.argmax(dim = 1)
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

    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    main()