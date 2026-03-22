# !/bin/bash

# Orchestrates the dataset building process. Creates labels by running the labeller script, then
# creates the spectrograms by running the spectrogram script.
# 
# If the script sees existing labels/.npz files, it will skip over them instead of overwriting them.
# Use the clear flag to delete existing data.
#
# Run this script from the root folder.
#
# Flags: 
#   -d: the course difficulty. Supported difficulties documented here: https://jozsefsallai.github.io/tja-js/classes/Difficulty.html
#   -c (optional): clears all existing data for the given course difficulty in the labels folder, and the .npz file.

# Constants
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DST="${DATA_DIR}/dst"
DATA_SRC="${DATA_DIR}/src"
LABELLER_PATH="${DATA_DST}/labels.js"
ARGS_VALIDATOR_PATH="${DATA_DST}/validateArgs.js"
PREPROCESSED_PATH="${DATA_DIR}/preprocessed"
NPZ_EXT=".npz"

# Args
clear_flag=''
diff_flag=''

# Args handling
print_usage() {
    printf "Usage: %s/build_dataset.sh -d <difficulty> [-c]\n" "$DATA_SRC"
}

while getopts ":d:c" flag; do
  case "$flag" in
    c) clear_flag='true' ;;
    d) diff_flag="$OPTARG" ;;
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

# Check and create necessary folders
if ! [[ -d "$DATA_DST" || -d "$DATA_SRC" ]]; then
    echo "Couldn't find $DATA_DST or $DATA_SRC. Make sure you're running the script from the root directory."
    current_dir=$(pwd)
    echo "Current dir is: $current_dir"
    exit 1
fi

# Compile scripts, validate arguments
if [[ ! -f "$LABELLER_PATH" || ! -f "$ARGS_VALIDATOR_PATH" ]]; then
    echo "Labeller scripts not found, compiling..."
    tsc -p "$DATA_DIR"
fi

node "$ARGS_VALIDATOR_PATH" "$diff_flag"
if [ $? -ne 0 ]; then
    echo "Invalid difficulty flag. See errors above."
    exit 1
fi 

# If the preprocessed data folder doesn't exist, create it
if ! [[ -d "$PREPROCESSED_PATH" ]]; then
    echo "$PREPROCESSED_PATH" not found. Creating it...
    mkdir -p $PREPROCESSED_PATH
else 
    # Deal with existing data
    if [[ $clear_flag != '' ]]; then
        existing_data="$PREPROCESSED_PATH/labels/$diff_flag/*"
        echo "Clearing existing data from $existing_data."
        rm -rf $existing_data
    fi
    
    # Check if there's an existing .npz file
    files=("$PREPROCESSED_PATH"/*"$NPZ_EXT")
    if [ -e "${files[0]}" ]; then
        echo "Found existing $NPZ_EXT file(s): ${files[*]}"
        echo "Delete them using the -clear flag before trying again."
        exit 1
    fi
fi

# Create labels
node "$LABELLER_PATH" "$diff_flag"
if [ $? -ne 0 ]; then
    echo "Label creation failed. See errors above."
    exit 1
fi 

echo "Finished creating labels. Creating spectrograms..."

# Create spectrograms


exit 0
# TODO: npz files should be placed in a separate folder in preprocessed