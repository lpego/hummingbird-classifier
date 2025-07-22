#!/bin/bash

# Script to run precision/recall computation for all FH videos
# Usage: ./run_precision_recall_all_videos.sh [results_folder] [output_folder] [buffer]

set -e  # Exit on any error

# Default values
RESULTS_FOLDER="${1:-./results_diffs/humbs/}"
OUTPUT_FOLDER="${2:-./precision_recall_results/}"
BUFFER="${3:-1}"

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# List of FH videos (extracted from your data folder)
VIDEOS=(
    "FH102_02"
    "FH103_01"
    "FH303_01"
    "FH402_01"
    "FH509_01"
    "FH706_01"
    "FH803_01"
)

echo "Starting precision/recall computation for ${#VIDEOS[@]} FH videos..."
echo "Results folder: $RESULTS_FOLDER"
echo "Output folder: $OUTPUT_FOLDER"
echo "Buffer: $BUFFER"
echo "================================"

# Initialize counters
SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_VIDEOS=()

# Loop through each video
for video in "${VIDEOS[@]}"; do
    echo ""
    echo "Processing video: $video"
    echo "--------------------------------"
    
    # Run the precision/recall computation
    if python compute_precision_recall.py "$video" \
        --method both \
        --buffer "$BUFFER" \
        --output "$OUTPUT_FOLDER" \
        --plot \
        --results-folder "$RESULTS_FOLDER" \
        --gt-folder "./data"; then
        
        echo "✅ Successfully processed $video"
        ((SUCCESS_COUNT++))
    else
        echo "❌ Failed to process $video"
        ((FAILED_COUNT++))
        FAILED_VIDEOS+=("$video")
    fi
done

echo ""
echo "================================"
echo "SUMMARY:"
echo "Total videos: ${#VIDEOS[@]}"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAILED_COUNT"

if [ $FAILED_COUNT -gt 0 ]; then
    echo ""
    echo "Failed videos:"
    for failed_video in "${FAILED_VIDEOS[@]}"; do
        echo "  - $failed_video"
    done
fi

echo ""
echo "Results saved to: $OUTPUT_FOLDER"
echo "Done!"