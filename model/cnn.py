import torch
import torch.nn as nn
import torch.nn.functional as functional
from typing import Tuple

class CNN(nn.Module):
    """
    The CNN model.

    Args:
        in_degree: number of input channels
        out_degree: number of output classes
    """
    def __init__(self, in_degree: int = 3, out_degree: int = 3, dropout: float = 0.5):
        super(CNN, self).__init__()

        # Rectangular kernels: onset detection cares more about changes over time than frequency
        self.conv1 = nn.Conv2d(in_channels=in_degree, out_channels=32, kernel_size=(7, 3))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))

        # Compute flattened size dynamically so fc1 doesn't need to be hand-tuned
        with torch.no_grad():
            dummy = torch.zeros(1, in_degree, 15, 80)
            dummy = self.pool1(functional.relu(self.conv1(dummy)))
            dummy = self.pool2(functional.relu(self.conv2(dummy)))
            dummy = self.pool3(functional.relu(self.conv3(dummy)))
            flat_size = dummy.flatten(start_dim=1).size(1)

        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(in_features=flat_size, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=out_degree)

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch_size, in_degree, 15, 80)

        Returns:
            logits of shape (batch_size, out_degree)
        """
        x = self.pool1(functional.relu(self.conv1(x)))
        x = self.pool2(functional.relu(self.conv2(x)))
        x = self.pool3(functional.relu(self.conv3(x)))
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        x = functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def predict(self, x) -> tuple[torch.tensor, torch.tensor]:
        """
        Runs inference and applies softmax to the logits.

        Args:
            x: input tensor of shape (batch_size, in_degree, 15, 80)

        Returns:
            probs: probability distribution over the note types
            preds: predicted note for each sample
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
        return probs, preds
    
