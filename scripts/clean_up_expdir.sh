#!/bin/bash

# Base directory containing the sub-directories with checkpoint files
BASE_DIR="/data/sls/scratch/clai24/word-seg/codebook-learning/exp/debug_collection_v2.1"

# Loop through all sub-directories
find "$BASE_DIR" -type f \( -name "model_epoch_*_loss_*.pth" \) | while read -r file; do
    # Extract the filename from the path
    filename=$(basename -- "$file")

    # Check if the file matches any of the preservation criteria
    case "$filename" in
    "best_loss_model.pth"|"best_acc_model.pth"|"top5_avg_loss_model.pth")
        # If the file is to be preserved, skip the deletion
        echo "Preserving $file"
        ;;
    *)
        # If the file does not match the preservation list, delete it
        echo "Deleting $file"
        rm "$file"
        ;;
    esac
done

