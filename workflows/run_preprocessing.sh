#!/bin/bash

# Configuration
ROOT_DIR="/work/hummingbird-classifier"
INPUT_VIDEO="$ROOT_DIR/data/***"
OUTPUT_FRAMES="$ROOT_DIR/data/***"

echo "Starting video preprocessing..."
echo "Input video: $INPUT_VIDEO"
echo "Output frames: $OUTPUT_FRAMES"

# Extract frames from videos
echo "Extracting frames..."
python scripts/preprocess/preprocess_extract_frames.py \
    --input_loc "$INPUT_VIDEO" \
    --output_loc "$OUTPUT_FRAMES"

echo "Frame extraction completed."

# Uncomment to convert videos to .mp4 format
# echo "Converting videos to MP4..."
# python scripts/preprocess/preprocess_convert_videos.py \
#     --input-video-path="/data/shared/raw-video-import/data/AnnotatedVideos/" \
#     --output-video-path="/data/shared/raw-video-import/data/RECODED_AnnotatedVideos/"

# Uncomment to create learning sets from videos
# echo "Creating learning sets..."
# python scripts/preprocess/preprocess_prepare_learning_sets_frames.py \
#     --learning-set-folder="data/lset_test/" \
#     --config="configs/configuration_hummingbirds.yml"