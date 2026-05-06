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
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data", "src"))
from spectrogram_utils import load_audio, compute_multi_resolution_mel, SAMPLE_RATE, HOP_SIZE, CONTEXT_HALF, CONTEXT_FRAMES, N_MELS, get_audio_from_folder
from cnn import CNN


CNN_FEATURE_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_cnn(path, device):
    info = torch.load(path, map_location=device, weights_only=False)
    model = CNN(in_degree=3, out_degree=1, dropout=0.0)
    model.load_state_dict(info["state_dict"])
    model.to(device)
    model.eval()
    return model

cnn_model = load_cnn("trained_model/hard", device)

NOTE_TYPE_TO_ID = {"circle": 1, "slider": 2, "spinner": 3}
max_seq_len = 200

# Load data — one sequence per map from JSON label files
sequences = []  # list of (X, Y) tensors, one per map
json_files = glob.glob("data/labels/*/*.json")
for path in json_files:
    with open(path, "r", encoding="utf-8") as f:
        notes = json.load(f)["hit_objects"]
    json_stem = os.path.splitext(os.path.basename(path))[0]
    audio_folder = os.path.join("data", "tracks", json_stem)
    try:
        audio_path = get_audio_from_folder(audio_folder)
        audio = load_audio(audio_path)
        mel_specs, n_frames = compute_multi_resolution_mel(audio)
    except Exception:
        continue
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
        frame_idx = int(round(t / 1000.0 * SAMPLE_RATE / HOP_SIZE))
        frame_idx = max(CONTEXT_HALF, min(n_frames - CONTEXT_HALF - 1, frame_idx))
        window = np.empty((1, 3, CONTEXT_FRAMES, N_MELS), dtype=np.float32)
        for r, spec in enumerate(mel_specs):
            window[0, r] = spec[frame_idx - CONTEXT_HALF : frame_idx + CONTEXT_HALF + 1]
        with torch.no_grad():
            feats = cnn_model.extract_features(torch.from_numpy(window).to(device)).squeeze(0).cpu().numpy()
        X_seq.append([dt, type_id, prev_x, prev_y] + feats.tolist())
        x, y = float(note["x"]), float(note["y"])
        Y_seq.append([x, y])
        prev_time = t
        prev_x, prev_y = x, y
    if random.random() < 0.5:
        for row in X_seq:
            row[2] = 1.0 - row[2]
        for row in Y_seq:
            row[0] = 1.0 - row[0]
    if random.random() < 0.5:
        for row in X_seq:
            row[3] = 1.0 - row[3]
        for row in Y_seq:
            row[1] = 1.0 - row[1]
    sequences.append((
        torch.tensor(X_seq, dtype=torch.float32),
        torch.tensor(Y_seq, dtype=torch.float32),
    ))

# Define the loss function and optimizer
criterion = nn.L1Loss()
model = TransformerRegressor(input_dim=260, d_model=64, nhead=4, num_layers=4, output_dim=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

checkpoint_path = "trained_model/transformer.pt"
start_epoch = 0
if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
num_epochs = 1200
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