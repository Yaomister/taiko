import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch


def make_dataset(
    X: np.array, y: np.array, batch_size: int, shuffle: bool
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.int64))
    )

    return DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size)
