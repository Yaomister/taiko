# taiko

## Prerequisites

Make sure you have the following installed:

- Node.js (includes npm)
- Python 3.8+
- pip (Python package manager)
- TypeScript compiler

Install the TypeScript compiler globally with:

```bash
npm install -g typescript
```

And required Python packages with:

```
pip install -r requirements.txt
```

torch is included in `requirements.txt`, but depending on your system, you may want to install
torch separately from pytorch.org with the right CUDA version for your system.

## Data pipeline

### Overview

The pipeline runs over several stages:

1. Create labels using labeller script (`data/preprocessed/labels/<diff>`)
2. Run spectrogram pipeline
   1. Build 3 log-mel spectrograms for each frame for 3 window sizes
   2. Create labels for each frame using labels from step 1
   3. Extract windows from spectrograms based on labels
3. Export dataset in batches to `data/preprocessed/exports/<my_data>`

A more detailed explanation can be found [here](https://docs.google.com/document/d/1nBxzO4Q0O5qYJpeCSY0WN7S8GYsNMCFrZRxRqKWj9QM/edit?tab=t.0) (WIP).

### Usage

#### 1. Add your songs into `data/tracks/`

- Various track sets can be found at [TJA Portal](https://tjaportal.neocities.org/).
- Track folders can be nested, but just make sure that any folder that contains a `.tja` file also has an audio file.
- Most audio types should be supported. See `data/src/spectrogram_utils.py` for supported audio types.

#### 2. Run the dataset builder script

Supported flags:

| Flag | Type     | Description                                                                                                    |
| ---- | -------- | -------------------------------------------------------------------------------------------------------------- |
| `-d` | Required | Course difficulty. See [supported difficulties](https://jozsefsallai.github.io/tja-js/classes/Difficulty.html) |
| `-f` | Required | Output directory name under `<data>/preprocessed/exports/`                                                     |
| `-n` | Required | Note types (comma-separated, e.g. `don,ka`. See `data/src/spectrogram_utils.py` for supported note types.)     |
| `-b` | Optional | Batch size; songs per dataset file (default: `50`)                                                             |
| `-c` | Optional | Clears labels directory for the specified difficulty.                                                          |
| `-r` | Optional | Ratio of negatives over positives (default: `0.5`).                                                            |

Example:

```bash
./data/src/build_dataset.sh -d easy -f my_dataset -n don,ka -b 50 -r 0.25
```

#### 3. Import .npz file for each batch

Example:

```python
import numpy as np

data = np.load(file="../preprocessed/exports/my_dataset/batch_1.npz")
X, y = data["X"], data["y"]

print(X.shape)
print(y.shape)
```

Note that there are multiple batch files per dataset. Load them in individually while training.

## Model training

Trains a CNN on the preprocessed `.npz` batch files produced by the data pipeline.

### Usage

```bash
python model/training.py \
  --data_dir data/preprocessed/exports/my_dataset \
  --out models/my_model.pth
```

### Arguments

| Argument       | Required | Default | Description                                                  |
| -------------- | -------- | ------- | ------------------------------------------------------------ |
| `--data_dir`   | Yes      | —       | Directory containing `batch_*.npz` files and `metadata.json` |
| `--out`        | Yes      | —       | Path to save the trained model `.pth` file                   |
| `--epochs`     | No       | `100`   | Number of training epochs                                    |
| `--lr`         | No       | `0.001` | Learning rate                                                |
| `--batch_size` | No       | `256`   | Mini-batch size                                              |
| `--split_prop` | No       | `0.1`   | Fraction of data held out for validation                     |
| `--dropout`    | No       | `0.5`   | Dropout rate on fully connected layers                       |
| `--seed`       | No       | `1`     | Random seed                                                  |

---

## Inference

Runs a trained model on an audio file and outputs a playable `.tja` chart.

### Usage

```bash
python model/inference.py \
  --audio path/to/song.mp3 \
  --bpm 140 \
  --model models/my_model.pth \
  --out path/to/output.tja
```

### Arguments

| Argument      | Required | Default      | Description                                                                                                          |
| ------------- | -------- | ------------ | -------------------------------------------------------------------------------------------------------------------- |
| `--audio`     | Yes      | —            | Path to input audio file                                                                                             |
| `--bpm`       | Yes      | —            | BPM of the song. Songs with BPM changes mid-way will produce inaccurate charts.                                      |
| `--model`     | Yes      | —            | Path to trained model `.pth` file                                                                                    |
| `--out`       | Yes      | —            | Path to write output `.tja` file                                                                                     |
| `--title`     | No       | `"Untitled"` | Song title written into the TJA header                                                                               |
| `--offset`    | No       | `0.0`        | Seconds of silence before the music starts in the audio file                                                         |
| `--threshold` | No       | `0.5`        | Minimum model confidence to count as a note (0–1). Increase to reduce false positives, decrease to catch more notes. |

```

```
