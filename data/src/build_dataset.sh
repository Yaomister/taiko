#!/bin/bash

# Orchestrates the dataset building process. Creates labels by running the labeller script, then
# creates the spectrograms by running the spectrogram script.
# 
# If the script sees existing labels/.npz files, it will skip over them instead of overwriting them.
# Use the clear flag to delete existing data.
#
# Directories must fit expected layout:
# root dir/
#   data/
#     src/              Should contain this script
#     dst/              Compiled JS (labels.js, validateArgs.js); built with: tsc -p data
#     tracks/           Track folders can be nested, but must each contain exactly one .tja and one
#                       audio file. The script searches this folder recursively.
#     preprocessed/     Created if missing; labels under labels/<difficulty>/; .npz under
#                       exports/ (basename required via -f).
#
# Flags: 
#   Required:
#     -d: the course difficulty. Supported difficulties documented here: https://jozsefsallai.github.io/tja-js/classes/Difficulty.html
#     -f: name of the directory the dataset will be exported to under <data>/preprocessed/exports/<name>.
#     -n: note types (comma-separated, e.g. don,ka).
#
#   Optional:
#     -b: batch size; number of songs per dataset file. Default: 50
#     -c: clears labels/<difficulty>/ under preprocessed (or -o labels tree).
#     -r: negative-to-positive sample ratio. Use -1 for all negatives. Default: 1.0
#     -H: hard negative radius in frames. Negatives sampled within this many frames of a note event.
#         Set to -1 to disable. Default: 60
#     -W: onset weight radius in frames. Background frames within this radius get linearly reduced
#         loss weight (weight = dist / radius). Set to 0 to disable. Default: 4

# Constants
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$DATA_DIR")" # The parent of the data directory should be the repo root
DATA_DST="${DATA_DIR}/dst"
DATA_SRC="${DATA_DIR}/src"
LABELLER_PATH="${DATA_DST}/labels.js"
ARGS_VALIDATOR_PATH="${DATA_DST}/validateArgs.js"
PREPROCESSED_PATH="${DATA_DIR}/preprocessed"
LABELS_DIR="$PREPROCESSED_PATH/labels"
TRACKS_DIR="${DATA_DIR}/tracks"

# Flags
clear_flag=''
diff_flag=''
labels_dir_flag=''
dataset_dir_flag=''
note_types=''
batch_size=''
negative_ratio=''
hard_negative_radius=''
onset_weight_radius=''

# Args handling
print_usage() {
    printf "Usage: %s/build_dataset.sh -d <difficulty> -f <export_dir> -n <note_types> [-b <batch_size>] [-r <negative_ratio>] [-H <hard_negative_radius>] [-W <onset_weight_radius>] [-c]\n" "$DATA_SRC"
}

while getopts ":d:f:n:b:r:H:W:c" flag; do
  case "$flag" in
    c) clear_flag='true' ;;
    d) diff_flag="$OPTARG" ;;
    f) dataset_dir_flag="$OPTARG" ;;
    n) note_types="$OPTARG" ;;
    b) batch_size="$OPTARG" ;;
    r) negative_ratio="$OPTARG" ;;
    H) hard_negative_radius="$OPTARG" ;;
    W) onset_weight_radius="$OPTARG" ;;
    :)
      echo "Option -$OPTARG requires an argument"
      exit 1 ;;
    ?)
      print_usage
      exit 1 ;;
  esac
done

# Required flags
if [[ -z "$diff_flag" || -z "$dataset_dir_flag" || -z "$note_types" ]]; then
    echo "Missing required flag(s)."
    print_usage
    exit 1
fi


export_dir="${PREPROCESSED_PATH}/exports/${dataset_dir_flag}"
if [[ -d "$export_dir" ]]; then
  echo "Directory already exists: $export_dir"
  exit 1
fi

if ! [[ -d "$DATA_DST" && -d "$DATA_SRC" ]]; then
    echo "Couldn't find $DATA_DST or $DATA_SRC."
    exit 1
fi

# Clear existing labels if requested
diff_folder="$(printf '%s' "$diff_flag" | tr '[:upper:]' '[:lower:]')"
existing_labels="${LABELS_DIR}/${diff_folder}"
if [[ $clear_flag != '' && -d "$existing_labels" ]]; then
    rm -rf "$existing_labels"
    echo "Cleared existing labels."
fi

# Compile scripts, validate arguments
echo "Compiling labeller scripts..."
tsc -p "$DATA_DIR"

node "$ARGS_VALIDATOR_PATH" "$diff_flag" "$LABELS_DIR" "$TRACKS_DIR"
if [ $? -ne 0 ]; then
    echo "Invalid arguments. See errors above."
    exit 1
fi 

# Create labels
echo "Creating labels..." 

node "$LABELLER_PATH" "$diff_flag" "$LABELS_DIR" "$TRACKS_DIR" 
if [ $? -ne 0 ]; then
    echo "Label creation failed. See errors above."
    exit 1
fi 

echo "Label creation complete. Creating dataset..." 

# Run from repo root
cmd=(
  python -m data.src.spectrogram
  --audio_dir "$TRACKS_DIR"
  --json_dir "${LABELS_DIR}/${diff_folder}"
  --out_path "$export_dir"
  --note_types "$note_types"
  --diff "$diff_flag"
)

if [[ -n "$batch_size" ]]; then
  cmd+=(--batch_size "$batch_size")
fi

if [[ -n "$negative_ratio" ]]; then
  cmd+=(--negative_ratio "$negative_ratio")
fi

if [[ -n "$hard_negative_radius" ]]; then
  cmd+=(--hard_negative_radius "$hard_negative_radius")
fi

if [[ -n "$onset_weight_radius" ]]; then
  cmd+=(--onset_weight_radius "$onset_weight_radius")
fi

PYTHONPATH="$REPO_ROOT" "${cmd[@]}"

if [ $? -ne 0 ]; then
    echo "Dataset export failed. See errors above."
    exit 1
fi

echo "Exported dataset to $export_dir."
exit 0