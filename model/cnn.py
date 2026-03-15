import torch
import torch.nn as nn
import torch.nn.functional as functional

class CNN(nn.Module):
    """
    CNN for taiko beat classification
    3 convolutional blocks followed by 2 fully connexcted layers

    Args:
        in_degree: number of input channels (1 for grayscale spectrogram)
        out_degree: number of output classes (8 for note types)
    """
    def __init__(self, in_degree = 1, out_degree = 8):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_degree, out_channels=32, kernel_size=(5, 5), padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), padding=2)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(in_features= 128 * 10 * 8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=out_degree)


    def forward(self, x):
        """
        Forward pass with relu activation functions and pooling

        Args:
            x: the input of the dimensions: batch_size, in_degree, 80, 64
        
        Returns:
            unnormalized prediction values with the dimensions of: batch_size, out_degree
        """
        x = self.conv1(x)
        x = functional.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = functional.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = functional.relu(x)
        x = self.pool3(x)

        # view(-1, ...) figures out the dimension itself
        x = x.view(-1, 128 * 10 * 8)
        x = self.fc1(x)
        x = functional.relu(x)
        
        x = self.fc2(x)

        return x

    def predict(self, x):
        """
        Prediction and normalizes the values (with softmax) in the forward pass
        
        Args:
            x: input tensor of shape (batch_size, 1, 80, 64)

        Returns:
            probs: probability distribution over the note types
            preds: predicted note for each sample
        """
        # Runs inference and applies softmax/normalizes the values in the forward pass
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
        return probs, preds
    

    def backprop(self, x, labels, optimizer, criterion):
        """
        Performs a single training step: forward pass, loss calculation, and weight update via backpropagation

        Args:
            x: input tensor (batch_size, 1, 80, 64)
            labels: ground truth note types
            optimizer: the optimizer
            criterion: the loss function
        
        Returns:
            loss value, predictions
        """
        optimizer.zero_grad()

        # forward pass
        outputs = self.forward(x)
        loss = criterion(outputs, labels)

        # backprop
        loss.backward()
        
        # update weights
        optimizer.step()

        return loss.item(), outputs