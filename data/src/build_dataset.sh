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
#   -diff: the course difficulty. Supported difficulties documented here: https://jozsefsallai.github.io/tja-js/classes/Difficulty.html
#   -clear (optional): clears all existing data for the given course difficulty in the labels folder, and the .npz file.

# Constants
DATA_FOLDER=./data
DATA_DST="${DATA_FOLDER}/dst"
DATA_SRC="${DATA_FOLDER}/src"
LABELLER_PATH="${DATA_DST}/labels.js"
PREPROCESSED_PATH="${DATA_FOLDER}/preprocessed"
NPZ_EXT=".npz"

# Args
# TODO: add args handling
COURSE_DIFF=$1

if ! [[ -d "$DATA_DST" || -d "$DATA_SRC" ]]; then
    current_dir=$(pwd)
    echo "Couldn't find $DATA_DST or $DATA_SRC. Make sure you're running the script from the root directory."
    echo "Current dir is: $current_dir"
    exit 1
fi

# If the preprocessed data folder doesn't exist, create it
if ! [[ -d "$PREPROCESSED_PATH" ]]; then
    echo "$PREPROCESSED_PATH" not found. Creating it...
    mkdir -p $PREPROCESSED_PATH
else
    # Check if there's an existing .npz file
    files=("$PREPROCESSED_PATH"/*"$NPZ_EXT")
    if [ -e "${files[0]}" ]; then
        echo "Found existing $NPZ_EXT file(s): ${files[*]}"
        echo "Delete them using the -clear flag before trying again."
        exit 1
    fi
fi

# Create labels
if [ -f "$LABELLER_PATH" ]; then
    echo "Found existing compiled labeller script at $LABELLER_PATH. Running..."
else
    echo "Labeller script not found at $LABELLER_PATH, compiling..."
    tsc
    echo "Creating labels..."
fi

node $LABELLER_PATH $COURSE_DIFF
if [ $? -ne 0 ]; then
    echo "Label creation failed. See errors above."
    exit 1
fi 

echo "Finished creating labels. Creating spectrograms..."

# Create spectrograms


exit 0
# TODO: Aadd flag for clearing existing data

# TODO: finish this, fix spectrogram.py, add courseDiff arg for labels.ts?