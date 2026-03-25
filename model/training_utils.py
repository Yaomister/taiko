import glob
import os
import numpy as np
from typing import Tuple
from torch.utils.data import TensorDataset, DataLoader
import torch

def load_all_batches(folder_dir : str) -> Tuple[np.ndarray, np.ndarray]:
    batch_files = sorted(glob.glob(os.path.join(folder_dir, "batch_*.npz")))
    if not batch_files:
        raise FileNotFoundError(f"No batch_*.npz files found in {folder_dir}")
    
    X_list = []
    y_list = []

    for path in batch_files:
        data = np.load(path)
        X_list.append(data["X"])
        y_list.append(data['y'])
        print(f"Loaded {os.path.basename(path)}: {data['X'].shape[0]} samples")

    return np.concat(X_list,axis=0), np.concat(y_list, axis=0)


def split_data(X: np.ndarray, y: np.ndarray, split_prop: float, seed: int):
     rng = np.random.default_rng(seed)
     n = len(X)
     split = max(1, int(n * split_prop))
     idx = rng.permutation(len(X))
     test_idx, train_idx = idx[split:], idx[:split]
     return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

def make_dataset(X: np.array, y: np.array, batch_size : int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.int64)))
    
    return DataLoader(dataset=dataset, shuffle=shuffle,batch_size=batch_size)




