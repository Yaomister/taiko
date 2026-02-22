# taiko

## Usage

### 1. Install dependencies

```bash
cd data
npm i
```

Make sure you also have Node.js installed.

### 2. Drag your songs into the `tracks/` directory

Various song sources can be found at [TJA Portal](https://tjaportal.neocities.org/).

### 3. Run the parser

Assuming you have your input `.tja` files in the `tracks/` directory:

```bash
cd data
tsc
node dst/parser.js
```

- The parsed chart data will be saved as JSON files in the `track_data/` directory.
- Right now, the data for each song looks like this:

```
[
  {
    "timeMs": 783, // when the note occurs
    "type": "don" // note type
  },
  ...
]
```

- Each `.tja` file should be in its own subdirectory under `tracks/`.
- The script looks for exactly one `.tja` file per subdirectory.

### Notes

- If the output JSON for a chart already exists, you'll need to delete it before re-running the parser.
- Only charts with "Easy" or "Normal" difficulty are processed by default. This can be changed later
