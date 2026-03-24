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
#   -d: the course difficulty. Supported difficulties documented here: https://jozsefsallai.github.io/tja-js/classes/Difficulty.html
#   -f: name of the exported dataset file under <data>/preprocessed/exports/.
#   -n: beat types (comma-separated, e.g. don,ka). Labels JSON still lists every type.
#   -c (optional): clears labels/<difficulty>/ under preprocessed (or -o labels tree).
#   -o (optional): parent directory for label JSON; a subdirectory named after the difficulty (lowercased) is created inside it. Default: <data>/preprocessed/labels

# Constants
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$DATA_DIR")" # The parent of the data directory should be the repo root
DATA_DST="${DATA_DIR}/dst"
DATA_SRC="${DATA_DIR}/src"
LABELLER_PATH="${DATA_DST}/labels.js"
ARGS_VALIDATOR_PATH="${DATA_DST}/validateArgs.js"
PREPROCESSED_PATH="${DATA_DIR}/preprocessed"
TRACKS_DIR="${DATA_DIR}/tracks"
LOGS_DIR="${PREPROCESSED_PATH}/logs" # Where script logs go

# Flags
clear_flag=''
diff_flag=''
output_dir_flag=''
export_file_flag=''
note_types=''

# Args handling
print_usage() {
    printf "Usage: %s/build_dataset.sh -d <difficulty> -f <export_basename.npz> -n <note_types> [-o <labels_parent_dir>] [-c]\n" "$DATA_SRC"
}

while getopts ":d:o:f:n:c" flag; do
  case "$flag" in
    c) clear_flag='true' ;;
    d) diff_flag="$OPTARG" ;;
    o) output_dir_flag="$OPTARG" ;;
    f) export_file_flag="$OPTARG" ;;
    n) note_types="$OPTARG" ;;
    :)
      echo "Option -$OPTARG requires an argument"
      exit 1 ;;
    ?)
      print_usage
      exit 1 ;;
  esac
done

if [[ $diff_flag == '' ]]; then
    echo "Missing difficulty flag."
    print_usage
    exit 1
fi

if [[ -z "$export_file_flag" ]]; then
    echo "Missing export filename flag (-f)."
    print_usage
    exit 1
fi

if [[ -z "$note_types" ]]; then
    echo "Missing note types flag (-n). Example: -n don,ka"
    print_usage
    exit 1
fi

export_basename="$export_file_flag"
if [[ "$export_basename" == */* ]]; then
    echo "Export filename (-f) must be a basename only (no /)."
    exit 1
fi

if [[ -z "$output_dir_flag" ]]; then
    labels_output_parent="$PREPROCESSED_PATH/labels"
else
    labels_output_parent="$output_dir_flag"
fi
if [[ "$labels_output_parent" != /* ]]; then
    labels_output_parent="$DATA_DIR/$labels_output_parent"
fi

diff_folder="$(printf '%s' "$diff_flag" | tr '[:upper:]' '[:lower:]')"

export_npz_path="${PREPROCESSED_PATH}/exports/${export_basename}.npz"
if [[ -f "$export_npz_path" ]]; then
  echo "File already exists: $export_npz_path"
  exit 1
fi

if ! [[ -d "$DATA_DST" || -d "$DATA_SRC" ]]; then
    echo "Couldn't find $DATA_DST or $DATA_SRC."
    exit 1
fi

# Clear existing labels if requested
existing_labels="${labels_output_parent}/${diff_folder}"
if [[ $clear_flag != '' && -d "$existing_labels" ]]; then
    rm -rf "$existing_labels"
    echo "Cleared existing labels."
fi

# Create logs directory if it doesn't exist
mkdir -p "$LOGS_DIR"
logs_file="$LOGS_DIR/${export_basename}_logs.txt"

# Compile scripts, validate arguments
if [[ ! -f "$LABELLER_PATH" || ! -f "$ARGS_VALIDATOR_PATH" ]]; then
    echo "Labeller scripts not found, compiling new scripts."
    tsc -p "$DATA_DIR"
fi

node "$ARGS_VALIDATOR_PATH" "$diff_flag" "$labels_output_parent" "$TRACKS_DIR"
if [ $? -ne 0 ]; then
    echo "Invalid arguments. See errors above."
    exit 1
fi 

# Create labels
echo "Creating labels..." 

node "$LABELLER_PATH" "$diff_flag" "$labels_output_parent" "$TRACKS_DIR" 
if [ $? -ne 0 ]; then
    echo "Label creation failed. See errors above."
    exit 1
fi 

echo "Label creation complete. Creating dataset..." 

# Run from repo root
PYTHONPATH="$REPO_ROOT" python -m data.src.spectrogram \
    --audio_dir "$TRACKS_DIR" \
    --json_dir "${labels_output_parent}/${diff_folder}" \
    --out_path "$export_npz_path" \
    --note_types "$note_types" 

if [ $? -ne 0 ]; then
    echo "Dataset export failed. See errors above."
    exit 1
fi

# TODO: add thread pool in spectrogram.py
# TODO: isntead of printing every single name, just print a progress number and current track

echo "Exported dataset to $export_npz_path."
exit 0