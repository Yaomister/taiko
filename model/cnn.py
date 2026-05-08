import torch
import torch.nn as nn
import torch.nn.functional as functional
from typing import Tuple

class CNN(nn.Module):
    """
    CNN for OSU standard hit object detection and placement.
    3 convolutional blocks followed by 7 output heads:
      fc2                 — hit detection logits (1,)
      fc_type             — object type logits: circle/slider/spinner (3,)
      fc_pos              — predicted (x, y) position normalized to [0, 1] (2,)
      fc_curve_type       — slider curve type logits: L/B/P/C (4,)
      fc_curve_directions — slider control point offset normalized to [-1, 1] (2,)
      fc_combo            — new combo start logit (1,)
      fc_length           — slider length normalized to [0, 1] by LENGTH_NORM=400 (1,)

    Input: (batch, 3, 15, 80)  — 3-channel multi-resolution log-mel spectrogram

    extract_features(x) returns the 256-dim fc1 activation before all output heads,
    used as audio conditioning input for the TransformerRegressor.
    """
    def __init__(self, in_degree: int = 3, out_degree: int = 1, dropout: float = 0.5):
        super(CNN, self).__init__()
        # in the onset detection paper they're using rectangular kernels because we care more about changes over time than frequency
        self.conv1 = nn.Conv2d(in_channels=in_degree, out_channels=32, kernel_size=(7, 3))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))

        # figure out the flat size (easier to finetune)
        with torch.no_grad():
            dummy = torch.zeros(1, in_degree, 15, 80)
            dummy = self.pool1(functional.relu(self.conv1(dummy)))
            dummy = self.pool2(functional.relu(self.conv2(dummy)))
            dummy = self.pool3(functional.relu(self.conv3(dummy)))
            flat_size = dummy.flatten(start_dim=1).size(1)

        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(in_features=flat_size, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=out_degree)
        self.fc_type = nn.Linear(in_features=256, out_features=3)
        self.fc_pos = nn.Linear(in_features=256, out_features=2)
        self.fc_curve_type = nn.Linear(256, 4)   # L / B / P / C
        self.fc_curve_directions = nn.Linear(256, 2)   # normalized control point offset
        self.fc_combo = nn.Linear(256, 1)
        self.fc_length = nn.Linear(256, 1)


    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.pool1(functional.relu(self.conv1(x)))
        x = self.pool2(functional.relu(self.conv2(x)))
        x = self.pool3(functional.relu(self.conv3(x)))
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        x = functional.relu(self.fc1(x))
        x = self.dropout(x)

        logits_hit = self.fc2(x)
        logits_type = self.fc_type(x)

        logits_curve_type = self.fc_curve_type(x)
        logits_curve_directions = self.fc_curve_directions(x)
        logits_combo = self.fc_combo(x)
        logits_length = self.fc_length(x)
        pos = self.fc_pos(x)

        return logits_hit, logits_type, pos, logits_curve_type, logits_curve_directions, logits_combo, logits_length

    def extract_features(self, x) -> torch.Tensor:
        x = self.pool1(functional.relu(self.conv1(x)))
        x = self.pool2(functional.relu(self.conv2(x)))
        x = self.pool3(functional.relu(self.conv3(x)))
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        x = functional.relu(self.fc1(x))
        return x

    def predict(self, x) -> tuple:
        self.eval()
        with torch.no_grad():
            logits_hit, logits_type, pos, logits_curve_type, logits_curve_directions, logits_combo, logits_length = self.forward(x)
            probs_hit = torch.sigmoid(logits_hit)
            probs_type = torch.softmax(logits_type, dim=1)
            preds_hit = (probs_hit > 0.5).long()
            preds_type = probs_type.argmax(dim=1)
            preds_curve_type = logits_curve_type.argmax(dim=1)
            preds_curve_directions = logits_curve_directions
            probs_combo = torch.sigmoid(logits_combo) 
            preds_combo = (probs_combo > 0.5).long().squeeze(-1)

        return probs_hit, probs_type, pos, preds_hit, preds_type, preds_curve_type, preds_curve_directions, preds_combo, probs_combo
