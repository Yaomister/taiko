# osu-beatmap-gen

A CNN-based OSU standard (mode 0) beatmap generator, forked from [Yaomister/taiko](https://github.com/Yaomister/taiko).

The taiko project converted MP3 files into spectrograms and used a CNN to detect drum hits. This project extends that approach to OSU standard mode, which requires predicting hit timing, object type (circle/slider/spinner), and x/y position on the playfield.

---

## Housekeeping (do this first)

- Delete `trained_model/two.pth` and `trained_model/three.pth` — these are taiko classifiers and are useless for OSU standard.
- Delete the `jobs/` scripts — all taiko-specific, will need new ones later.
- The entire `data/src/*.ts` stack (TypeScript TJA parser) is being replaced with a Python `.osu` parser. You can leave the files for reference but they won't be used.

---

## Step 1 — Write the .osu parser (Python)

**Goal:** given a `.osu` file, return the difficulty name and a list of hit objects with `time_ms`, `type` (circle/slider/spinner), `x`, and `y`.

The `.osu` format is plain text. You only need `[General]`, `[Metadata]`, and `[HitObjects]`.

### Difficulty in .osu files

A single `.osz` pack contains multiple `.osu` files — one per difficulty, all sharing the same audio file. Each difficulty has a `Version:` field in `[Metadata]` which is its freeform name (e.g. `"Easy"`, `"Insane"`, `"Sotarks' Extra"`).

The parser should read and return this so the dataset builder can filter on it. There is no standardized difficulty enum — mappers name them freely. In practice you'll want to group them yourself (e.g. filter for files whose `Version` contains `"Insane"` or `"Extra"` to get hard maps).

### .osu file format references
- [Official .osu format spec](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_%28file_format%29) — authoritative. Read the `[General]`, `[Metadata]`, and `[HitObjects]` sections.
- Hit object type is encoded in a bitmask on column 4: bit 0 = circle, bit 1 = slider, bit 3 = spinner.
- x and y are columns 1 and 2 (0-indexed). Playfield is 512 × 384 px — normalize to [0, 1] by dividing x/512, y/384.
- Filter `Mode: 0` only — skip taiko (1), catch (2), mania (3).

### What to write
A single `data/src/parse_osu.py` with a function like:

```python
def parse_osu(path: str) -> dict:
    # returns {
    #   "version": str,           # difficulty name from Version: field
    #   "hit_objects": [
    #     {"time_ms": int, "type": str, "x": float, "y": float},
    #     ...
    #   ]
    # }
```

Test it manually on a few `.osu` files before moving on. OSU beatmap packs (`.osz` files) are just zip archives — rename to `.zip` and extract to get the `.osu` files inside.

### Where to get .osu files
- [osu! beatmap listing](https://osu.ppy.sh/beatmapsets) — download `.osz` packs (free with account). Filter by mode: osu! standard.
- Aim for a mix of difficulties and song types.

---

## Step 2 — Rebuild the dataset builder

**Goal:** extend `data/src/spectrogram.py` so each training sample carries `(x, y, type)` labels in addition to the existing spectrogram slice and hit/no-hit label.

### Difficulty filtering mirrors the taiko approach

The taiko codebase already threads `--diff` through the dataset builder and into metadata. Do the same here: pass a `--diff` string (e.g. `"Insane"`) to the dataset builder, which filters `.osu` files by whether their `Version:` field contains that string (case-insensitive substring match is fine).

Train a separate model per difficulty tier. Mixing difficulties is a bad idea — the same spectrogram frame maps to completely different x/y targets depending on difficulty, which gives the model a contradictory training signal.

### What changes
- The JSON label files currently store `{ time_ms, type }` (taiko note type). Replace with `{ version: str, hit_objects: [{ time_ms, type: "circle"|"slider"|"spinner", x: float, y: float }] }` from your new parser. The dataset builder uses `version` to filter, then discards it.
- In `spectrogram.py`, the `y` array currently holds a scalar class ID per frame. Extend it to hold a struct or split into separate arrays:
  - `y_hit`: 0 or 1 (does a hit object start at this frame?)
  - `y_type`: 0/1/2 (circle/slider/spinner) — only meaningful when `y_hit=1`
  - `y_pos`: (x, y) normalized floats — only meaningful when `y_hit=1`
- Save these alongside `X` in the `.npz` batch files.

### Resources
- [numpy.savez_compressed](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html) — you're already using this; just add new keys.
- The existing `process_song` function in `spectrogram_utils.py` is where the per-frame label assignment happens — that's where the new labels get attached.

---

## Step 3 — Add output heads to the model

**Goal:** the CNN currently outputs one head (hit class logits). Add two more: one for object type, one for (x, y) position.

### What changes in `model/cnn.py`
After `fc2` (the existing hit detection head), add:
- `fc_type`: `Linear(256, 3)` — outputs logits over circle/slider/spinner
- `fc_pos`: `Linear(256, 2)` — outputs (x, y) in [0, 1]

The `forward()` method should return all three heads as a tuple. Only the type and position heads need to be active (loss applied) on frames where a hit is predicted.

### Resources
- [PyTorch nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- [Multi-task learning with PyTorch](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html) — the loss combination section is relevant.

---

## Step 4 — Update the training loop

**Goal:** replace the single `CrossEntropyLoss` with a combined loss over all three heads.

### Loss design
- **Hit detection**: `BCEWithLogitsLoss` on `y_hit` (binary, all frames)
- **Object type**: `CrossEntropyLoss` on `y_type` (only on hit frames)
- **Position**: `MSELoss` on `y_pos` (only on hit frames)
- Total loss: `L_hit + λ_type * L_type + λ_pos * L_pos` — start with all λ = 1.0 and tune later.

Masking non-hit frames out of the type and position losses is important — don't backprop position on background frames.

### Resources
- [BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
- [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)
- Masking pattern: `loss_type = ce(logits_type[hit_mask], y_type[hit_mask])`

---

## What stays the same

- Spectrogram generation (mel, windowing, slicing) — `spectrogram_utils.py` is untouched
- CNN backbone (all three conv blocks + `fc1`)
- Train/val loop structure, optimizer, early stopping, batch loading
- `.npz` batch file format (just gains new keys)

---

## Deferred

Slider `curvePoints` reconstruction is deferred. Parse sliders as hit objects (time + start position) for now, and ignore curve geometry. Come back to this once hit detection and position are working on circles.

When you do tackle it:
- [Slider curve types](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_%28file_format%29#sliders) — bezier, catmull, linear, perfect circle
- Slider length is in `osu!pixels` and needs to be converted using the `SliderMultiplier` from `[Difficulty]`
