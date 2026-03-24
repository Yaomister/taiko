# taiko

## Usage

### 1. Install dependencies

```bash
cd data
npm i
pip install librosa
pip install torch
```

Make sure you also have Node.js installed.

### 2. Drag your songs into the `tracks/` directory

- Various track sets can be found at [TJA Portal](https://tjaportal.neocities.org/).
- Track folders can be nested, but just make sure that any folder that contains a `.tja` file also has an audio file.
- Most audio types should be supported. See `data/src/spectrogram_utils.py` for supported audio types.

### 3. Run the dataset builder script

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
