# AutoOSU

## Prerequisites

Make sure you have the following installed:

- Python 3.8+
- pip (Python package manager)

And required Python packages with:

```
pip install -r requirements.txt
```

torch is included in `requirements.txt`, but depending on your system, you may want to install
torch separately from pytorch.org with the right CUDA version for your system.

## Data pipeline

### Overview

The pipeline runs over several stages:

1. Add song folders containing `.osu` files and audio into `data/tracks/`
2. Generate JSON label files from `.osu` beatmaps using `make_labels.py`
3. Run the spectrogram pipeline
   1. Build 3 log-mel spectrograms for each frame for 3 window sizes
   2. Create labels for each frame using the JSON labels from step 2
   3. Export spectrogram windows and labels to `data/preprocessed/<diff>/`

### Usage

#### 1. Add your songs into `data/tracks/`

- Each song folder should contain an `.osu` file and an audio file.
- Most audio types should be supported. See `data/src/spectrogram_utils.py` for supported audio types.

#### 2. Generate label files

```bash
python data/src/make_labels.py \
  --audio_dir data/tracks \
  --out_dir data/labels/insane \
  --diff insane
```

Supported arguments:

| Argument      | Required | Default | Description                                                                     |
| ------------- | -------- | ------- | ------------------------------------------------------------------------------- |
| `--audio_dir` | Yes      | —       | Directory containing song folders with `.osu` files                             |
| `--out_dir`   | Yes      | —       | Directory to write JSON label files into (created if missing)                   |
| `--diff`      | Yes      | —       | Difficulty substring to match against the `Version:` field (case-insensitive)   |

#### 3. Run the spectrogram pipeline

```bash
python data/src/spectrogram.py \
  --audio_dir data/tracks \
  --json_dir data/labels/insane \
  --out_path data/preprocessed/insane \
  --note_types "circle,slider,spinner" \
  --diff insane
```

Supported arguments:

| Argument               | Required | Default              | Description                                                                                    |
| ---------------------- | -------- | -------------------- | ---------------------------------------------------------------------------------------------- |
| `--audio_dir`          | Yes      | `"data/tracks"`      | Directory containing song folders with audio files                                             |
| `--json_dir`           | Yes      | —                    | Directory containing JSON label files from `make_labels.py`                                    |
| `--out_path`           | Yes      | —                    | Output directory for `batch_*.npz` files and `metadata.json`                                   |
| `--note_types`         | Yes      | —                    | Comma-separated hit object types to include (e.g. `circle,slider,spinner`)                     |
| `--diff`               | No       | —                    | Difficulty label written into metadata                                                         |
| `--batch_size`         | No       | `50`                 | Number of songs per batch file                                                                 |
| `--negative_percentage`| No       | `0.5`                | Fraction of total samples that are background (e.g. `0.33` = 33%)                             |
| `--seed`               | No       | `0`                  | Random seed                                                                                    |

#### 4. Import `.npz` file for each batch

Example:

```python
import numpy as np

data = np.load(file="data/preprocessed/insane/batch_1.npz")
X, y, y_pos, weights = data["X"], data["y"], data["y_pos"], data["weights"]

print(X.shape)      # (N, 3, 15, 80) spectrogram windows
print(y.shape)      # (N,) hit type class id
print(y_pos.shape)  # (N, 2) normalized (x, y) position
print(weights.shape)# (N,) per-sample loss weights
```

Note that there are multiple batch files per dataset. Load them in individually while training.

## Model training

### CNN

Trains the CNN on preprocessed `.npz` batch files produced by the data pipeline.

#### Usage

```bash
python model/training.py \
  --data_dir data/preprocessed/insane \
  --out trained_model/insane
```

#### Arguments

| Argument       | Required | Default | Description                                                                     |
| -------------- | -------- | ------- | ------------------------------------------------------------------------------- |
| `--data_dir`   | Yes      | —       | Directory containing `batch_*.npz` files and `metadata.json`                    |
| `--out`        | Yes      | —       | Path to save the trained model checkpoint                                       |
| `--epochs`     | No       | `100`   | Number of training epochs                                                       |
| `--lr`         | No       | `0.001` | Learning rate                                                                   |
| `--batch_size` | No       | `256`   | Mini-batch size                                                                 |
| `--split_prop` | No       | `0.1`   | Fraction of data held out for validation                                        |
| `--dropout`    | No       | `0.5`   | Dropout rate on fully connected layers                                          |
| `--seed`       | No       | `1`     | Random seed                                                                     |
| `--patience`   | No       | `10`    | Early stopping patience in epochs                                               |

### Transformer (position prediction)

Trains the transformer on note sequences using CNN audio features as conditioning. Requires a trained CNN first.

#### Usage

```bash
python model/train_transformer.py
```

Reads from `data/labels/` and `data/tracks/` and saves to `trained_model/transformer.pt`. Edit the script directly to change the CNN path or number of epochs.

---

## Inference

Runs a trained CNN and optional transformer on an audio file and outputs a playable `.osu` beatmap.

### Usage

```bash
python model/inference.py \
  --audio path/to/song.mp3 \
  --model trained_model/insane \
  --out path/to/output.osu \
  --transformer trained_model/transformer.pt
```

### Arguments

| Argument            | Required | Default      | Description                                                                                                          |
| ------------------- | -------- | ------------ | -------------------------------------------------------------------------------------------------------------------- |
| `--audio`           | Yes      | —            | Path to input audio file                                                                                             |
| `--model`           | Yes      | —            | Path to trained CNN checkpoint                                                                                       |
| `--out`             | Yes      | —            | Path to write output `.osu` file                                                                                     |
| `--title`           | No       | `"Untitled"` | Song title in `.osu` metadata                                                                                        |
| `--diff`            | No       | `"Normal"`   | Difficulty name in `.osu` metadata                                                                                   |
| `--threshold`       | No       | `0.5`        | Minimum hit confidence to place a note (0–1). Increase to reduce false positives, decrease to catch more notes.      |
| `--min_gap_frames`  | No       | `5`          | Minimum frames between notes to avoid double triggers                                                                |
| `--hp`              | No       | `5`          | HP drain rate 0–10. Lower values are more forgiving                                                                  |
| `--transformer`     | No       | `None`       | Path to transformer checkpoint for position prediction                                                               |
