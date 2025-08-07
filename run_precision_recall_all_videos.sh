#!/bin/bash

# Script to run precision/recall computation for all FH videos
# Usage: ./run_precision_recall_all_videos.sh [results_folder] [output_folder] [buffer]

set -e  # Exit on any error

# Default values
RESULTS_FOLDER="${1:-./results_test_sanity/hummingbird/}"
OUTPUT_FOLDER="${2:-./results_test_sanity/hummingbird/precision_recall_results/}"
BUFFER="${3:-1}"

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# List of FH videos (extracted from your data folder)
# Automatically detect all FH*.avi videos in the data folder
VIDEOS=()
while IFS= read -r -d '' video_path; do
    # Extract just the basename without .avi extension
    video_name=$(basename "$video_path" .avi)
    VIDEOS+=("$video_name")
done < <(find ./data -name "FH*.avi" -type f -print0 | sort -z)

echo "Found ${#VIDEOS[@]} FH videos:"
for video in "${VIDEOS[@]}"; do
    echo "  - $video"
done
echo ""

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
    
    # Run the precision/recall computation # combined dl_only 
    if python compute_precision_recall.py "$video" \
        --method colorhist triplet running_mean  \
        --buffer "$BUFFER" \
        --output "$OUTPUT_FOLDER" \
        --plot \
        --results-folder "$RESULTS_FOLDER" \
        --gt-folder "./data/cleaned_ground_truth.csv"; then
        
        echo "++ Successfully processed $video"
        ((SUCCESS_COUNT++))
    else
        echo "-- Failed to process $video"
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