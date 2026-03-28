import os
import random
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple
from mlp import MLP
from training_utils import load_all_batches, split_data, make_dataset

IN_FEATURES = 3 * 15 * 80  # 3600 — matches NPZ batch shape (3, 15, 80)


def train(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device).view(x.size(0), 1, -1)
        y = y.to(device)
        loss, _ = model.backprop(x, y, optimizer=optimizer, criterion=criterion)
        total_loss += loss * x.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).view(x.size(0), 1, -1)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
    return total_loss / total, correct / total

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--split_prop", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading all saved batches from {args.data_dir}")
    X, y = load_all_batches(args.data_dir)
    
    n_classes = 8
    meta_path = os.path.join(args.data_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        n_classes = len(meta.get("classes", {})) or n_classes
    
    X_train, y_train, X_test, y_test = split_data(X, y, args.split_prop, args.seed)
    print(f"Train: {len(X_train):,} samples, Test: {len(X_test):,} samples")

    train_dataset = make_dataset(X_train, y_train, args.batch_size, True)
    test_dataset = make_dataset(X_test, y_test, args.batch_size, False)

    model = MLP(in_features=IN_FEATURES, out_degree=n_classes, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    loss_function = torch.nn.CrossEntropyLoss()

    print(f"Training for {args.epochs} epochs")
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_dataset, optimizer, loss_function, device)
        test_loss, test_accuracy = evaluate(model, test_dataset, loss_function, device)
        print(f"Epoch {epoch}/{args.epochs}  |  train_loss: {train_loss:.4f}  |  test_loss: {test_loss:.4f}  |  test_acc: {test_accuracy:.3%}")
    
    torch.save({
        "state_dict": model.state_dict(),
        "n_classes":  n_classes,
        "in_features": IN_FEATURES,
        "args": vars(args),
    }, args.out_path)
    print(f"Saved to {args.out_path}")

if __name__ == "__main__":
    main()