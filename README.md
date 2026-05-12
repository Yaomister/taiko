# Taiko

A machine learning pipeline that generates playable Taiko no Tatsujin (太鼓の達人) drum charts from audio files. It processes songs into log-mel spectrograms, trains a CNN to detect note onsets, and outputs `.tja` chart files.

---

## Prerequisites

| Requirement            | Notes                            |
| ---------------------- | -------------------------------- |
| Node.js (includes npm) | Required for TypeScript compiler |
| Python 3.8+            | Core runtime                     |
| pip                    | Python package manager           |
| TypeScript compiler    | Install globally via npm         |

**Install the TypeScript compiler:**

```bash
npm install -g typescript
```

**Install Python dependencies:**

```bash
pip install -r requirements.txt
```

> **Note:** `torch` is included in `requirements.txt`, but you may want to install it separately from [pytorch.org](https://pytorch.org) with the correct CUDA version for your system.

---

## Data Pipeline

### Overview

The pipeline runs in three stages:

1. **Label creation** — generates note labels per difficulty (`data/preprocessed/labels/<diff>`)
2. **Spectrogram processing** — builds 3 log-mel spectrograms per frame across 3 window sizes, assigns frame labels, and extracts windowed segments
3. **Dataset export** — writes batched `.npz` files to `data/preprocessed/exports/<my_dataset>`

For a more detailed explanation, see the [pipeline documentation](https://docs.google.com/document/d/1nBxzO4Q0O5qYJpeCSY0WN7S8GYsNMCFrZRxRqKWj9QM/edit?tab=t.0) (WIP).

---

### Step 1 — Add songs to `data/tracks/`

- Track sets can be found at [TJA Portal](https://tjaportal.neocities.org/)
- Track folders may be nested; any folder containing a `.tja` file must also contain an audio file
- Most common audio formats are supported — see `data/src/spectrogram_utils.py`

### Step 2 — Run the dataset builder

```bash
./data/src/build_dataset.sh --difficulty easy --folder my_dataset --notes don,ka --batch-size 50 --ratio 0.33
```

**Flags:**

| Flag                     | Required | Description                                                                                                                                                                          |
| ------------------------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--difficulty`           | ✅       | Course difficulty ([supported values](https://jozsefsallai.github.io/tja-js/classes/Difficulty.html))                                                                                |
| `--folder`               | ✅       | Output directory name under `data/preprocessed/exports/`                                                                                                                             |
| `--notes`                | ✅       | Note types, comma-separated (e.g. `don,ka`). See `spectrogram_utils.py` for supported types                                                                                          |
| `--batch-size`           |          | Songs per batch file (default: `50`)                                                                                                                                                 |
| `--clear`                |          | Clears the labels directory for the specified difficulty                                                                                                                             |
| `--ratio`                |          | Fraction of samples that are background (default: `0.5`)                                                                                                                             |
| `--hard-negative-radius` |          | Hard negative radius in frames; negatives sampled within this many frames of a note event (default: `60`, ~0.7s). Set to `-1` to disable                                             |
| `--onset-weight-radius`  |          | Onset weight radius in frames; background frames within this radius get linearly reduced loss weight. Positive frames always have weight `1.0` (default: `4`). Set to `0` to disable |

### Step 3 — Load batches for training

```python
import numpy as np

data = np.load(file="../preprocessed/exports/my_dataset/batch_1.npz")
X, y, W = data["X"], data["y"], data["W"]

print(X.shape)  # Spectrogram windows
print(y.shape)  # Frame labels
print(W.shape)  # Onset weights
```

Each dataset is split across multiple batch files — load them individually during training.

---

## Model Training

Trains a CNN on the `.npz` batch files produced by the data pipeline.

```bash
python model/training.py \
  --data_dir data/preprocessed/exports/my_dataset \
  --out models/my_model.pth
```

**Arguments:**

| Argument          | Required | Default | Description                                                  |
| ----------------- | -------- | ------- | ------------------------------------------------------------ |
| `--data_dir`      | ✅       | —       | Directory containing `batch_*.npz` files and `metadata.json` |
| `--out`           | ✅       | —       | Path to save the trained `.pth` model file                   |
| `--epochs`        |          | `100`   | Number of training epochs                                    |
| `--lr`            |          | `0.001` | Learning rate                                                |
| `--batch_size`    |          | `256`   | Mini-batch size                                              |
| `--split_prop`    |          | `0.1`   | Fraction of data held out for validation                     |
| `--dropout`       |          | `0.5`   | Dropout rate on fully connected layers                       |
| `--seed`          |          | `1`     | Random seed                                                  |
| `--patience`      |          | `10`    | Early stopping patience (epochs)                             |
| `--class_weights` |          | off     | Weight cross-entropy loss by inverse class frequency         |
| `--onset_weights` |          | off     | Use per-sample onset weights from the dataset                |

---

## Inference

Runs a trained model on an audio file and outputs a playable `.tja` chart.

```bash
python model/inference.py \
  --audio path/to/song.mp3 \
  --bpm 140 \
  --model models/my_model.pth \
  --out path/to/output.tja
```

**Arguments:**

| Argument      | Required | Default      | Description                                                                                                         |
| ------------- | -------- | ------------ | ------------------------------------------------------------------------------------------------------------------- |
| `--audio`     | ✅       | —            | Path to input audio file                                                                                            |
| `--bpm`       | ✅       | —            | Song BPM (mid-song BPM changes will produce inaccurate charts)                                                      |
| `--model`     | ✅       | —            | Path to trained `.pth` model file                                                                                   |
| `--out`       | ✅       | —            | Path to write output `.tja` file                                                                                    |
| `--title`     |          | `"Untitled"` | Song title written into the TJA header                                                                              |
| `--offset`    |          | `0.0`        | Seconds of silence before the music starts                                                                          |
| `--threshold` |          | `0.5`        | Minimum model confidence to register a note (0–1). Increase to reduce false positives; decrease to catch more notes |
