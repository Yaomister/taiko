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
from training_utils import load_all_batches, split_data, make_dataset



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
        total_loss += loss.item() + len(X_batch)
        correct += (preds == y_batch).sum().item()
        total += len(X_batch)

    return total_loss/ total, correct/ total

    

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
    print(f"Loading all saved batches from f{args.data_dir}")
    X, y = load_all_batches(args.data_dir)
    meta_path = os.path.join(args.data_dir, "metadata.json")
    n_classes = 3
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        n_classes = len(meta.get("classes", {})) or n_classes

    # Split the dataset into training and testing sets
    X_train, y_train, X_test, y_test =split_data(X, y, args.split_prop, args.seed)
    print(f"Train: {len(X_train):,} samples, Test: {len(X_test):,} samples")

    train_dataset = make_dataset(X_train, y_train, args.batch_size, True)
    test_dataset = make_dataset(X_test, y_test, args.batch_size, False)

    # Create the model
    model = CNN(in_degree=3, out_degree=n_classes, dropout=args.dropout).to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    loss_function = torch.nn.CrossEntropyLoss()

    # Training loop
    print(f"Training for {args.epochs} epochs")
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_dataset, optimizer, loss_function, device)
        test_loss, test_accuracy = evaluate(model, test_dataset, loss_function, device)
        print(
            f"Epoch {epoch}/{args.epochs}  |  "
            f"train_loss: {train_loss:.4f}  |  "
            f"test_loss: {test_loss:.4f}  |  "
            f"test_acc: {test_accuracy:.3%}"
        )

    # Save the model
    torch.save({
    "state_dict": model.state_dict(),
    "n_classes": n_classes,
    "args": vars(args), 
    }, args.out_dir)    





if __name__ == "__main__":
    main()