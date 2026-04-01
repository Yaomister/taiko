"""
Training script for Taiko CNN note classifier. Loads preprocessed .npz batch files and trains the CNN model.

Usage:
    --data_dir data/preprocessed/train_data \\
    --out_dir models/my_model.pt \\
    --epochs 100 \\
    --lr 0.001 \\
    --batch_size 256


Arguments:
    --data_dir (str): Directory containing batch_0.npz, batch_1.npz, ... and metadata.json (required)

    --out_dir (str): Path to save the trained model weights (required)

    --epochs (int): Number of training epochs. Default is 100

    --lr (float): Learning rate. Default is 0.001

    --batch_size (int): Mini-batch size for training. Default is 256

    --val_split (float): Proportion of data to use for validation. Default is 0.1

    --seed (int): Random seed. Default is 0
    
    --dropout (float): Dropout rate on fully connected layers. Default is 0.5

"""

import os
import random
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cnn import CNN
from typing import Tuple
import glob
from torch.utils.data import TensorDataset


def train(model: CNN, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_function = nn.Module, device = torch.device) -> float:
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = loss_function(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


def evaluate(model: CNN, loader: DataLoader, loss_function: nn.Module, device: torch.device) -> Tuple[float, float]:
    """Returns the average loss and accuracy"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        preds = logits.argmax(dim=1)
        loss = loss_function(logits, y_batch)
        total_loss += loss.item() * len(X_batch)
        correct += (preds == y_batch).sum().item()
        total += len(X_batch)

    return total_loss / total, correct / total, total

    

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CNN on preprocessed .npz data.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with batch_*.npz files")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save the trained model weights")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--split_prop", type=float, default=0.1, help="The proportion of the dataset used for testing")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the data
    meta_path = os.path.join(args.data_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    n_classes = len(meta.get("classes", {})) or 3
    n_samples = meta["n_samples"]
    val_start = int(n_samples * (1 - args.split_prop))
    batch_files = sorted(glob.glob(os.path.join(args.data_dir, "batch_*.npz")))
    print(f"Total samples: {n_samples:,} | Train: {val_start:,} | Val: {n_samples - val_start:,}")

    # Create the model
    model = CNN(in_degree=3, out_degree=n_classes, dropout=args.dropout).to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    loss_function = torch.nn.CrossEntropyLoss()

    # Training loop
    print(f"Training for {args.epochs} epochs")
    for epoch in range(1, args.epochs + 1):
        train_loss_sum, val_loss_sum, correct, total, train_total = 0.0, 0.0, 0, 0, 0
        samples_seen = 0

        for path in batch_files:
            data = np.load(path)
            X = torch.from_numpy(data["X"].astype(np.float32))
            y = torch.from_numpy(data["y"].astype(np.int64))
            n = len(X)

            batch_train_end = max(0, min(n, val_start - samples_seen))
            batch_val_start = batch_train_end

            if batch_train_end > 0:
                loader = DataLoader(TensorDataset(X[:batch_train_end], y[:batch_train_end]),
                                    batch_size=args.batch_size, shuffle=True)
                train_loss_sum += train(model, loader, optimizer, loss_function, device) * batch_train_end
                train_total += batch_train_end

            if batch_val_start < n:
                loader = DataLoader(TensorDataset(X[batch_val_start:], y[batch_val_start:]),
                                    batch_size=args.batch_size, shuffle=False)
                bl, bc, bt = evaluate(model, loader, loss_function, device)
                val_loss_sum += bl * bt
                correct += int(bc * bt)
                total += bt

            samples_seen += n
            del X, y, data

    print(
        f"Epoch {epoch}/{args.epochs}  |  "
        f"train_loss: {train_loss_sum/train_total:.4f}  |  "
        f"val_loss: {val_loss_sum/total:.4f}  |  "
        f"val_acc: {correct/total:.3%}"
    )

    # Save the model
    torch.save({
    "state_dict": model.state_dict(),
    "n_classes": n_classes,
    "args": vars(args), 
    }, args.out_dir)   

     





if __name__ == "__main__":
    main()