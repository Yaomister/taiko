import torch
import torch.nn as nn
import torch.nn.functional as functional

class MLP(nn.Module):
    """
    Creates MLP with the given parameters 

    Args:
        in_features: dimensionality of each input token (default 5120).
        out_degree: number of output classes.
        d_model: internal model dimension after projection.
        n_head: number of attention heads.
        dropout: dropout probability.
    """
    def __init__(self, in_features = 5120, out_degree = 3, d_model = 128, n_head = 4, dropout = 0.1):
        super(MLP, self).__init__()

        self.projection = nn.Linear(in_features, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)

        self.fc1 = nn.Linear(d_model, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, out_degree)

        self.dropout = nn.Dropout(dropout)

    # forward pass
    def forward(self, x):
        """
        Does forward pass with relu activation functions and then returns prediction

        Args:
            x: the input layer with dimensions of: batch, seq_len, in_features
        
        Returns:
            The raw predictions with dimensions of: batch, out_degree
        """
        x = self.norm(self.projection(x))
        x, _ = self.attention(x, x, x)
        x = x[:, -1, :]

        x = functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

    # one simple backpropagation step
    def backprop(self, x, labels, optimizer, criterion):
        """
        complete forward + backward + weight update step.
        
        Args:
            x: input tensor with the shape of (batch, seq_len, in_features)
            labels: ground truth class indices (batch,)
            optimizer: torch.optim optimizer (e.g. Adam)
            criterion: loss function (e.g. Cross Entropy Loss)
        
        Returns:
            loss value, predictions
        """
        # forward pass
        outputs = self.forward(x)
        loss = criterion(outputs, labels)

        # backprop
        optimizer.zero_grad()
        loss.backward()

        # update weights
        optimizer.step()

        return loss.item(), outputs