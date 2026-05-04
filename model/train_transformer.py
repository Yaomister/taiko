import torch
import torch.optim as optim
import torch.nn as nn
import glob
import os
from transformer import TransformerRegressor
import json
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NOTE_TYPE_TO_ID = {"circle": 1, "slider": 2, "spinner": 3}
max_seq_len = 200

# Load data — one sequence per map from JSON label files
sequences = []  # list of (X, Y) tensors, one per map
json_files = glob.glob("../data/labels/*/*.json")
for path in json_files:
    with open(path, "r", encoding="utf-8") as f:
        notes = json.load(f)["hit_objects"]
    notes = notes[:max_seq_len]
    if len(notes) < 2:
        continue
    X_seq, Y_seq = [], []
    prev_time = 0.0
    prev_x, prev_y = 0.5, 0.5
    for note in notes:
        t = float(note["time_ms"])
        dt = (t - prev_time) / 1000.0
        type_id = NOTE_TYPE_TO_ID.get(note["type"], 1)
        x, y = float(note["x"]), float(note["y"])
        X_seq.append([dt, type_id, prev_x, prev_y])
        Y_seq.append([x, y])
        prev_time = t
        prev_x, prev_y = x, y
    sequences.append((
        torch.tensor(X_seq, dtype=torch.float32),
        torch.tensor(Y_seq, dtype=torch.float32),
    ))

# Define the loss function and optimizer
criterion = nn.MSELoss()
model = TransformerRegressor(input_dim=4, d_model=64, nhead=4, num_layers=4, output_dim=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

checkpoint_path = "../trained_model/transformer.pt"
start_epoch = 0
if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt["epoch"]
            print(f"Resuming from epoch {start_epoch} with optimizer state")
        else:
            start_epoch = ckpt.get("epoch", 0)
            print(f"Resuming weights from epoch {start_epoch}, optimizer reset")
    else:
        model.load_state_dict(ckpt)
        print("Loaded weights from old-format checkpoint, optimizer reset")

# Training loop
batch_size = 32
num_epochs = 200
for epoch in tqdm(range(start_epoch, start_epoch + num_epochs), desc="Training", unit="epoch"):
    random.shuffle(sequences)
    epoch_loss = 0.0
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        X_batch = pad_sequence([s[0] for s in batch], batch_first=True).to(device)
        Y_batch = pad_sequence([s[1] for s in batch], batch_first=True).to(device)
        lengths = torch.tensor([s[0].shape[0] for s in batch])
        mask = torch.arange(X_batch.shape[1]) >= lengths.unsqueeze(1)
        real = ~mask.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch, src_key_padding_mask=mask.to(device))
        loss = criterion(outputs[real], Y_batch[real])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
 
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{start_epoch + num_epochs}], Loss: {epoch_loss:.4f}')

torch.save({
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "epoch": start_epoch + num_epochs,
}, checkpoint_path)