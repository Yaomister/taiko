"""
Training script for OSU standard CNN note classifier. Loads preprocessed .npz batch files and trains the CNN model.

Usage:
    python model/training.py \\
        --data_dir data/preprocessed/insane \\
        --out trained_model/insane.pt

Arguments:
    --data_dir (str): Directory containing batch_0.npz, batch_1.npz, ... and metadata.json (required)

    --out (str): File path to save the trained model weights (required)

    --epochs (int): Number of training epochs. Default is 100

    --lr (float): Learning rate. Default is 0.001

    --batch_size (int): Mini-batch size for training. Default is 256

    --split_prop (float): Proportion of data to use for validation. Default is 0.1

    --seed (int): Random seed. Default is 1

    --dropout (float): Dropout rate on fully connected layers. Default is 0.5

    --patience (int): Early stopping patience in epochs. Default is 10

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


def train(
    model: CNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    BCE: nn.Module,
    CE: nn.Module,
    MSE: nn.Module,
    device: torch.device,
    class_weights: torch.Tensor = None,
) -> float:
    """Train one epoch, returns average loss."""
    model.train()
    total_hit_loss = 0.0
    total_type_loss = 0.0
    total_pos_loss = 0.0
    total_combo_loss = 0.0
    # class_weights covers [background, circle, slider, spinner]; CE uses [circle, slider, spinner]
    weights = class_weights[1:].to(device) if class_weights is not None else None
    for X_batch, y_batch, y_pos_batch, w_batch, y_curve_type_batch, y_curve_cp_batch, y_combo_batch, y_length_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        y_pos_batch = y_pos_batch.to(device)
        w_batch = w_batch.to(device)
        y_curve_type_batch = y_curve_type_batch.to(device)
        y_curve_cp_batch = y_curve_cp_batch.to(device)
        y_combo_batch = y_combo_batch.to(device)
        y_length_batch = y_length_batch.to(device)
        optimizer.zero_grad()
        logits_hit, logits_type, pos, logits_curve_type, logits_curve_directions, logits_combo, logits_length = model(X_batch)
        y_hit = (y_batch != 0).float()
        hit_sample_loss = nn.functional.binary_cross_entropy_with_logits(
            logits_hit.squeeze(-1), y_hit, reduction="none"
        )
        loss = (hit_sample_loss * w_batch).mean()
        total_hit_loss += hit_sample_loss.sum().item()

        hit_mask = y_hit.bool()
        if hit_mask.any():
            type_sample_loss = nn.functional.cross_entropy(
                logits_type[hit_mask], y_batch[hit_mask].long() - 1, weight=weights, reduction="none"
            )
            loss += type_sample_loss.mean()
            total_type_loss += type_sample_loss.sum().item()

            pos_sample_loss = nn.functional.mse_loss(pos[hit_mask], y_pos_batch[hit_mask], reduction="none")
            loss += pos_sample_loss.sum(dim=1).mean()
            total_pos_loss += pos_sample_loss.sum().item()

            combo_loss = nn.functional.binary_cross_entropy_with_logits(
                logits_combo[hit_mask].squeeze(-1), y_combo_batch[hit_mask],
                pos_weight=torch.tensor(4.0, device=device), reduction="none"
            )
            loss += combo_loss.mean()
            total_combo_loss += combo_loss.sum().item()

        slider_mask = (y_batch == 2).bool()
        if slider_mask.any():
            curve_type_sample_loss = nn.functional.cross_entropy(
                logits_curve_type[slider_mask], y_curve_type_batch[slider_mask].long()
            )
            curve_dir_loss = nn.functional.mse_loss(
                logits_curve_directions[slider_mask], y_curve_cp_batch[slider_mask], reduction="none"
            )
            length_loss = nn.functional.mse_loss(
                logits_length[slider_mask].squeeze(-1), y_length_batch[slider_mask]
            )
            loss += curve_type_sample_loss + curve_dir_loss.mean() + length_loss

        loss.backward()
        optimizer.step()

    total_loss = total_hit_loss + total_type_loss + total_pos_loss + total_combo_loss
    return total_loss / len(loader.dataset)


def evaluate(
    model: CNN,
    loader: DataLoader,
    BCE: nn.Module,
    CE: nn.Module,
    MSE: nn.Module,
    device: torch.device,
) -> Tuple[float, int, int]:
    """Returns (average loss, correct predictions, total samples)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch, y_pos_batch, w_batch, y_curve_type_batch, y_curve_cp_batch, y_combo_batch, y_length_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pos_batch = y_pos_batch.to(device)
            w_batch = w_batch.to(device)
            y_curve_type_batch = y_curve_type_batch.to(device)
            y_curve_cp_batch = y_curve_cp_batch.to(device)
            y_combo_batch = y_combo_batch.to(device)
            y_length_batch = y_length_batch.to(device)
            logits_hit, logits_type, pos, logits_curve_type, logits_curve_directions, logits_combo, logits_length = model(X_batch)
            y_hit = (y_batch != 0).float()
            hit_sample_loss = nn.functional.binary_cross_entropy_with_logits(
                logits_hit.squeeze(-1), y_hit, reduction="none"
            )
            loss = (hit_sample_loss * w_batch).mean()
            hit_mask = y_hit.bool()
            if hit_mask.any():
                type_sample_loss = nn.functional.cross_entropy(
                    logits_type[hit_mask], y_batch[hit_mask].long() - 1, reduction="none"
                )
                loss += type_sample_loss.mean()
                pos_sample_loss = nn.functional.mse_loss(
                    pos[hit_mask], y_pos_batch[hit_mask], reduction="none"
                )
                loss += pos_sample_loss.sum(dim=1).mean()
                combo_loss = nn.functional.binary_cross_entropy_with_logits(
                    logits_combo[hit_mask].squeeze(-1), y_combo_batch[hit_mask],
                    pos_weight=torch.tensor(4.0, device=device), reduction="none"
                )
                loss += combo_loss.mean()
            slider_mask = (y_batch == 2)
            if slider_mask.any():
                curve_type_loss = nn.functional.cross_entropy(
                    logits_curve_type[slider_mask], y_curve_type_batch[slider_mask]
                )
                curve_dir_loss = nn.functional.mse_loss(
                    logits_curve_directions[slider_mask], y_curve_cp_batch[slider_mask]
                )
                length_loss = nn.functional.mse_loss(
                    logits_length[slider_mask].squeeze(-1), y_length_batch[slider_mask]
                )
                loss += curve_type_loss + curve_dir_loss + length_loss

            total_loss += loss.item() * len(X_batch)
            preds_hit = (logits_hit.squeeze(-1) > 0)
            correct += (preds_hit == y_hit.bool()).sum().item()
            total += len(X_batch)
    return total_loss / len(loader.dataset), correct, total


def plot_losses(
    train_losses: list[float], val_losses: list[float], out_path: str
) -> None:
    """Save a train/val loss curve to out_path."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the CNN on preprocessed .npz data."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory with batch_*.npz files and metadata.json",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="File path to save trained model weights to",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--split_prop",
        type=float,
        default=0.1,
        help="Proportion of data reserved for validation",
    )
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs without val loss improvement). Default is 10",
    )
    parser.add_argument(
        "--class_weights",
        action="store_true",
        default=False,
        help="Weight cross entropy loss by inverse class frequency. Default is off",
    )
    parser.add_argument(
        "--onset_weights",
        action="store_true",
        default=False,
        help="Use per-sample onset weights from the dataset during training. Default is off",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
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
    n_samples = sum(len(np.load(p)["X"]) for p in batch_files)
    val_start = int(n_samples * (1 - args.split_prop))
    print(f"Loaded {n_samples:,} samples from {len(batch_files)} batch files")
    print(f"Train: {val_start:,} samples | Val: {n_samples - val_start:,} samples")

    # Optionally compute inverse-frequency class weights from training split
    class_weights = None
    if args.class_weights:
        class_counts = torch.zeros(n_classes, dtype=torch.float32)
        samples_seen = 0
        for path in batch_files:
            data = np.load(path)
            y = torch.from_numpy(data["y"].astype(np.int64))
            n = len(y)
            train_end = max(0, min(n, val_start - samples_seen))
            if train_end > 0:
                for c in range(n_classes):
                    class_counts[c] += (y[:train_end] == c).sum().item()
            samples_seen += n
            del y, data
        class_weights = class_counts.sum() / (n_classes * class_counts.clamp(min=1))
        print(
            "Class weights:",
            {i: f"{w:.4f}" for i, w in enumerate(class_weights.tolist())},
        )

    model = CNN(in_degree=3, out_degree=1, dropout=args.dropout).to(device)
    BCE = nn.BCEWithLogitsLoss()
    CE = nn.CrossEntropyLoss()
    MSE = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state_dict = None
    epochs_no_improve = 0

    with tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch") as pbar:
        for epoch in pbar:
            train_loss_sum, val_loss_sum, train_total = 0.0, 0.0, 0
            samples_seen = 0
            correct, total = 0, 0

            for path in batch_files:
                data = np.load(path)
                X = torch.from_numpy(data["X"].astype(np.float32))
                y = torch.from_numpy(data["y"])
                y_pos = torch.from_numpy(data["y_pos"].astype(np.float32))
                y_curve_type = torch.from_numpy(data["y_curve_type"].astype(np.int64))
                y_curve_cp = torch.from_numpy(data["y_curve_cp"].astype(np.float32))
                y_combo = torch.from_numpy(data["y_combo"].astype(np.float32))
                y_length = torch.from_numpy(data["y_length"].astype(np.float32))
                n = len(X)
                w = (
                    torch.from_numpy(data["weights"].astype(np.float32))
                    if "weights" in data and args.onset_weights
                    else torch.ones(n, dtype=torch.float32)
                )

                batch_train_end = max(0, min(n, val_start - samples_seen))
                batch_val_start = batch_train_end

                if batch_train_end > 0:
                    loader = DataLoader(
                        TensorDataset(
                            X[:batch_train_end],
                            y[:batch_train_end],
                            y_pos[:batch_train_end],
                            w[:batch_train_end],
                            y_curve_type[:batch_train_end],
                            y_curve_cp[:batch_train_end],
                            y_combo[:batch_train_end],
                            y_length[:batch_train_end],
                        ),
                        batch_size=args.batch_size,
                        shuffle=True,
                    )
                    batch_loss = train(
                        model, loader, optimizer, BCE, CE, MSE, device, class_weights
                    )
                    train_loss_sum += batch_loss * batch_train_end
                    train_total += batch_train_end

                if batch_val_start < n:
                    loader = DataLoader(
                        TensorDataset(
                            X[batch_val_start:],
                            y[batch_val_start:],
                            y_pos[batch_val_start:],
                            w[batch_val_start:],
                            y_curve_type[batch_val_start:],
                            y_curve_cp[batch_val_start:],
                            y_combo[batch_val_start:],
                            y_length[batch_val_start:],
                        ),
                        batch_size=args.batch_size,
                        shuffle=False,
                    )
                    bl, bcorrect, bt = evaluate(model, loader, BCE, CE, MSE, device)
                    val_loss_sum += bl * bt
                    correct += bcorrect
                    total += bt

                samples_seen += n
                del X, y, y_pos, w, y_curve_type, y_curve_cp, y_combo, y_length, data

            train_loss = train_loss_sum / train_total
            val_loss = val_loss_sum / total
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            accuracy = correct / total if total > 0 else 0.0
            postfix = {
                "train_loss": f"{train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "accuracy": f"{accuracy:.1%}",
            }

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            pbar.set_postfix(postfix)

            if epochs_no_improve >= args.patience:
                print(
                    f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)"
                )
                break

    model.load_state_dict(best_state_dict)

    # Save model
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_classes": n_classes,
            "args": vars(args),
        },
        args.out,
    )
    print(f"Model saved to {args.out}")

    # Save loss plot
    model_name = os.path.splitext(args.out)[0]
    plot_path = model_name + ".png"
    plot_losses(train_losses, val_losses, plot_path)
    print(f"Loss plot saved to {plot_path}")


if __name__ == "__main__":
    main()
