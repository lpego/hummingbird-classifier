#!/bin/bash

# Configuration
ROOT_DIR="/work/hummingbird-classifier"
MODEL="mobilenet-v0"
LSET_FOLD="${ROOT_DIR}/data/lset_test/"
VIDEO_PATH="${ROOT_DIR}/data/demo"
ANNOTATIONS="${ROOT_DIR}/data/Weinstein2018MEE_ground_truth.csv"
CONFIG_FILE="${ROOT_DIR}/configs/configuration_hummingbirds.yaml"

# Script paths
FINETUNE_SCRIPT="${ROOT_DIR}/scripts/main_classification_finetune.py"
INFERENCE_SCRIPT="${ROOT_DIR}/scripts/inference/main_score_inference.py"
ASSESSMENT_SCRIPT="${ROOT_DIR}/scripts/inference/main_assessment.py"
PLOTTING_SCRIPT="${ROOT_DIR}/scripts/evaluation/main_plotting.py"

# Model and output paths
MODEL_PATH="${ROOT_DIR}/models/${MODEL}"
OUTPUT_PATH="${ROOT_DIR}/outputs/video_scores/${MODEL}/"

echo "Starting hummingbird classifier pipeline..."

# FINETUNE MODEL
# Fine-tune the classification model based on new aggregated learning sets
# Uncomment to enable:
# echo "Fine-tuning model..."
# python "${FINETUNE_SCRIPT}" \
#     --input_dir="${LSET_FOLD}" \
#     --save_model="${MODEL_PATH}" \
#     --config_file="${CONFIG_FILE}"

# RUN INFERENCE
echo "Running inference..."
python "${INFERENCE_SCRIPT}" \
    --videos_root_folder="${VIDEO_PATH}" \
    --model_path="${MODEL_PATH}" \
    --annotation_file="${ANNOTATIONS}" \
    --output_file_folder="${OUTPUT_PATH}" \
    --config_file="${CONFIG_FILE}" \
    --update

# RUN EVALUATION
# Uncomment to enable:
# echo "Running evaluation..."
# python "${ASSESSMENT_SCRIPT}" \
#     --results_path="${OUTPUT_PATH}" \
#     --config_file="${CONFIG_FILE}" \
#     --update \
#     --aggregate \
#     --make_plots

# PLOT RESULTS
# Generate detailed visualizations
# Uncomment to enable:
# echo "Generating plots..."
# python "${PLOTTING_SCRIPT}" \
#     --results_path="${OUTPUT_PATH}" \
#     --config_file="${CONFIG_FILE}" \
#     --update \
#     --aggregate

echo "Pipeline completed successfully!"
