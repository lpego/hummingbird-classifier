#!/bin/bash

ROOT_DIR="/work/hummingbird-classifier"
INPUT_CSV="/work/hummingbird-classifier/outputs/mobilenet-v0/Alaspungo.csv"
THRESHOLD=.95
OUT_FOLDER="/work/hummingbird-classifier/outputs/filtered_images2"
INPUT_DIR="/work/hummingbird-classifier/data/demo/testdir_subdirs/Alaspungo"

# Copy positive images to out_folder
python scripts/postprocess/postprocess_move_positive_frames.py \
    --input_csv "$INPUT_CSV" \
    --threshold "$THRESHOLD" \
    --out_folder "$OUT_FOLDER" \
    --verbose True

# # Delete images from input folder
# python scripts/postprocess/postprocess_cleanup.py \
#     --input_dir "$INPUT_DIR" \
#     --input_csv "$INPUT_CSV" \
#     --threshold "$THRESHOLD" \
#     --out_folder "$OUT_FOLDER"