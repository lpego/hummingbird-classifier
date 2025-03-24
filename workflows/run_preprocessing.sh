#!/bin/bash

ROOT_DIR="/work/hummingbird-classifier"
INPUT_VIDEO="/work/hummingbird-classifier/data/try/2024_0906_185436_001.MP4"
OUTPUT_FRAMES="/work/hummingbird-classifier/data/try/2024_0906_185436_001"

# Extract frames from videos (any format)
python scripts/preprocess/scripts/preprocess/preprocess_extract_frames.py \
    --input_loc "$INPUT_VIDEO" \
    --output_loc "$OUTPUT_FRAMES"

# Convert or recode videos to .mp4 format
# python scripts/preprocess/preprocess_convert_videos.py \
#   --input-video-path=/data/shared/raw-video-import/data/AnnotatedVideos/ \
#   --output-video-path=/data/shared/raw-video-import/data/RECODED_AnnotatedVideos/

# Create learning sets from videos
# python scripts/preprocess/preprocess_prepare_learning_sets_frames.py \
#     --learning-set-folder=data/lset_test/ \
#     -c configs/configuration_hummingbirds.yml