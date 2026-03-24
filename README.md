# taiko

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

#### 1. Install dependencies

```bash
cd data
npm i
pip install librosa
pip install torch
```

Make sure you also have Node.js installed.

#### 2. Drag your songs into the `tracks/` directory

- Various track sets can be found at [TJA Portal](https://tjaportal.neocities.org/).
- Track folders can be nested, but just make sure that any folder that contains a `.tja` file also has an audio file.
- Most audio types should be supported. See `data/src/spectrogram_utils.py` for supported audio types.

#### 3. Run the dataset builder script

Supported flags:

| Flag | Type | Description |
|------|------|-------------|
| `-d` | Required | Course difficulty. See [supported difficulties](https://jozsefsallai.github.io/tja-js/classes/Difficulty.html) |
| `-f` | Required | Output directory name under `<data>/preprocessed/exports/` |
| `-n` | Required | Note types (comma-separated, e.g. `don,ka`. See `data/src/spectrogram_utils.py` for supported note types.) |
| `-b` | Optional | Batch size; songs per dataset file (default: `50`) |
| `-c` | Optional | Clears labels directory for the specified difficulty. |

Example:
```bash
./data/src/build_dataset.sh -d easy -f my_dataset -n don,ka -b 50
```

#### 4. Import .npz file for each batch

Example:
```python
import numpy as np

data = np.load(file="../preprocessed/exports/my_dataset/batch_1.npz")
X, y = data["X"], data["y"]

print(X.shape)
print(y.shape)
```
Note that there are multiple batch files per dataset. Load them in individually while training.