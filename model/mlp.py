import torch
import torch.nn as nn
import torch.nn.functional as functional

class MLP(nn.Module):
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
        x = self.norm(self.projection(x))
        x, _ = self.attention(x, x, x)
        x = x[:, -1, :]

        x = functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

    # one backpropagation step
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
