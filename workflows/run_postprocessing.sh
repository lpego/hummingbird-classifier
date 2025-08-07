#!/bin/bash

# Configuration
readonly ROOT_DIR="."
readonly INPUT_CSV="${ROOT_DIR}/outputs/mobilenet-v0/scores.csv"
readonly THRESHOLD=0.95
readonly OUT_FOLDER="${ROOT_DIR}/outputs/postprocessing"
readonly INPUT_DIR="${ROOT_DIR}/outputs/mobilenet-v0/testdir_subdirs"

echo "Starting postprocessing workflow..."

# Copy positive images to output folder
echo "Copying positive images (threshold: ${THRESHOLD})..."
python scripts/postprocess/postprocess_move_positive_frames.py \
    --input_csv "${INPUT_CSV}" \
    --threshold "${THRESHOLD}" \
    --out_folder "${OUT_FOLDER}" \
    --verbose

echo "Postprocessing complete. Images saved to: ${OUT_FOLDER}"

# Optional: Delete images from input folder
# echo "Cleaning up input directory..."
# python scripts/postprocess/postprocess_cleanup.py \
#     --input_dir "${INPUT_DIR}" \
#     --input_csv "${INPUT_CSV}" \
#     --threshold "${THRESHOLD}" \
#     --out_folder "${OUT_FOLDER}"