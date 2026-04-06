import torch
import torch.nn as nn
import torch.nn.functional as functional


class CNN(nn.Module):
    """
    CNN for taiko beat detection, adapted from convNet in "Improved musical onset detection with Convolutional Neural Networks".
    https://ieeexplore.ieee.org/document/6854953

    Input: (N, 3, 15, 80)  — 3-channel multi-resolution log-mel spectrogram
    Output: (N, 1) — raw logit; apply sigmoid for beat probability
    """

    def __init__(self, in_channels: int = 3, dropout: float = 0.5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 10, kernel_size=(7, 3)
        )  # rectangular kernel: 7 time frames, 3 freq bins - onset detection cares more about temporal changes
        self.pool1 = nn.MaxPool2d(
            kernel_size=(1, 3)
        )  # pool freq only - time dim is too short to pool

        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3))  # pool freq only

        with (
            torch.no_grad()
        ):  # infer flat size dynamically so it doesn't need to be hardcoded
            dummy = torch.zeros(1, in_channels, 15, 80)
            dummy = self.pool1(functional.relu(self.conv1(dummy)))
            dummy = self.pool2(functional.relu(self.conv2(dummy)))
            flat_size = dummy.flatten(start_dim=1).size(1)

        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(flat_size, 256)
        self.fc2 = nn.Linear(256, 120)
        self.fc3 = nn.Linear(120, 1)  # single logit - sigmoid gives P(beat)

    def forward(self, x) -> torch.Tensor:
        x = self.pool1(functional.relu(self.conv1(x)))
        x = self.pool2(functional.relu(self.conv2(x)))
        x = self.dropout(x.flatten(start_dim=1))
        x = self.dropout(functional.relu(self.fc1(x)))
        x = self.dropout(functional.relu(self.fc2(x)))
        return self.fc3(x)

    def predict(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference and return beat probabilities and binary predictions.

        Args:
            x: input tensor of shape (N, 3, 15, 80)

        Returns:
            probs: (N,) sigmoid probabilities
            preds: (N,) binary predictions (1 if beat, 0 if background)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits).squeeze(1)
            preds = (probs > 0.5).long()
        return probs, preds
