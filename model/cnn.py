

import torch

import torch.nn as nn
import torch.nn.functional as functional


class CNN(nn.Module):
    def __init__(self, in_degree = 1, out_degree = 3):
        # assuming 80 by 64 spectrogram
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_degree, out_channels=32, kernel_size=(5, 5), padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), padding=2)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(in_features= 128 * 10 * 8, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=out_degree)


    def forward(self, x):
        x = self.conv1(x)
        x = functional.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = functional.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = functional.relu(x)
        x = self.pool3(x)

        x = x.view(-1, 128 * 10 * 8)
        x = self.fc1(x)
        x = functional.relu(x)
        
        x = self.fc2(x)

        return x

    # one backprop step
    def backprop(self, x, labels, optimizer, criterion):
        # forward pass
        outputs = self.forward(x)
        loss = criterion(outputs, labels)

        # backprop
        optimizer.zero_grad()
        loss.backward()

        # update weights
        optimizer.step()

        return loss.item(), outputs
