import argparse
import json
from typing import List, Dict
 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cnn import CNN

def load_model(path: str, device: torch.device):
    info = torch.load(path, map_location=device)

    state_dict = info['state_dict']
    n_classes = info["n_classes"]
    args = info.get("args", {})
    dropout = args.get("dropout", 0.5)
    model = CNN(in_degree=3, out_degree=n_classes, dropout=dropout)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Loaded model with f{n_classes} beat types")

    return model, info

