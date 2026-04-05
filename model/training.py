"""
Training script for Taiko CNN note classifier. Loads preprocessed .npz batch files and trains the CNN model.

Usage:
    --data_dir data/preprocessed/train_data \\
    --out models/my_model.pt \\
    --epochs 100 \\
    --lr 0.001 \\
    --batch_size 256


Arguments:
    --data_dir (str): Directory containing batch_0.npz, batch_1.npz, ... and metadata.json (required)

    --out (str): File path to save the trained model weights (required)

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
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from cnn import CNN
from typing import Tuple
import glob


def train(model: CNN, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_function: nn.Module, device: torch.device) -> float:
    """Train one epoch, returns average loss."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = loss_function(logits, (y_batch > 0).float().unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


def evaluate(model: CNN, loader: DataLoader, loss_function: nn.Module, device: torch.device) -> Tuple[float, float, int]:
    """Returns average loss, accuracy, and total sample count."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            preds = (logits.squeeze(1) > 0).long()
            loss = loss_function(logits, (y_batch > 0).float().unsqueeze(1))
            total_loss += loss.item() * len(X_batch)
            correct += (preds == (y_batch > 0).long()).sum().item()
            total += len(X_batch)
    return total_loss / total, correct / total, total


def plot_losses(train_losses: list[float], val_losses: list[float], out_path: str) -> None:
    """Save a train/val loss curve to out_path."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CNN on preprocessed .npz data.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with batch_*.npz files and metadata.json")
    parser.add_argument("--out", type=str, required=True, help="File path to save trained model weights to")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--split_prop", type=float, default=0.1, help="Proportion of data reserved for validation")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True) # Create out path if it doesn't exist
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load metadata
    meta_path = os.path.join(args.data_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    n_classes = len(meta.get("classes", {}))

    # Collect batch files and count samples
    batch_files = sorted(glob.glob(os.path.join(args.data_dir, "batch_*.npz")))
    n_samples = meta.get("n_samples")
    val_start = int(n_samples * (1 - args.split_prop))
    print(f"Loaded {n_samples:,} samples from {len(batch_files)} batch files")
    print(f"Train: {val_start:,} samples | Val: {n_samples - val_start:,} samples")

    # Build model
    model = CNN(in_degree=3, out_degree=1, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", factor=0.7, patience=10, min_lr=1e-6
    # )

    # Weighted loss function (pos_weight = negatives / positives)
    class_counts = meta["class_counts"]
    id_to_type = meta["classes"]  # {"0": "background", "1": "don", ...}
    ordered_counts = [class_counts[id_to_type[str(i)]] for i in range(n_classes)]
    neg_count = ordered_counts[0]
    pos_count = sum(ordered_counts[1:])
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    train_losses, val_losses = [], []

    with tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch") as pbar:
        for epoch in pbar:
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

            train_loss = train_loss_sum / train_total
            val_loss = val_loss_sum / total
            val_acc = correct / total

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            pbar.set_postfix({
                "train_loss": f"{train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_acc": f"{val_acc:.1%}",
            })

            # scheduler.step(val_loss)

    # Save model
    torch.save({
        "state_dict": model.state_dict(),
        "n_classes": n_classes,
        "args": vars(args),
    }, args.out)
    print(f"Model saved to {args.out}")

    # Save loss plot
    model_name = os.path.splitext(args.out)[0]
    plot_path = model_name + ".png"
    plot_losses(train_losses, val_losses, plot_path)
    print(f"Loss plot saved to {plot_path}")

if __name__ == "__main__":
    main()