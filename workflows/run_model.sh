#!/bin/bash


ROOT_DIR="D:/hummingbird-classifier-vm"
MODEL="mobilenet-v0"
# LSET_FOLD="${ROOT_DIR}/data/mzb_example_data/aggregated_learning_set"
LSET_FOLD="${ROOT_DIR}/data/lset_test/"
VIDEO_PATH="/data/shared/frame-diff-anomaly/data/annotated_videos"
ANNOTATIONS="${ROOT_DIR}/data/Weinstein2018MEE_ground_truth.csv"

## FINETUNE MDOEL
## -------------------------------------------------------------------------------- 
## This is run to finetune the classification model based on the new aggreagted learning sets; it will return a new model.
# python ${ROOT_DIR}/scripts/main_classification_finetune.py \
#     --input_dir=${LSET_FOLD} \
#     --save_model=${ROOT_DIR}/models/${MODEL} \
#     --config_file=${ROOT_DIR}/configs/configuration_hummingbirds.yaml \

# ## RUN INFERENCE
# ## -------------------------------------------------------------------------------- 
# python ${ROOT_DIR}/scripts/inference/main_score_inference.py \
#     --videos_root_folder=${VIDEO_PATH} \
#     --model_path=${ROOT_DIR}/models/${MODEL} \
#     --annotation_file=${ANNOTATIONS} \
#     --output_file_folder=${ROOT_DIR}/outputs/video_scores/${MODEL}/ \
#     --config_file=${ROOT_DIR}/configs/configuration_hummingbirds.yaml \
#     --update

## RUN EVALUATION
## -------------------------------------------------------------------------------- 
python ${ROOT_DIR}/scripts/inference/main_assessment.py \
    --results_path=${ROOT_DIR}/outputs/video_scores/${MODEL} \
    --config_file=${ROOT_DIR}/configs/configuration_hummingbirds.yaml \
    --update \
    --aggregate \
    --make_plots # this only makes an overview plot to summarise results, it's not the nice one


# ## PLOT RESULTS
# ## -------------------------------------------------------------------------------- 
# python ${ROOT_DIR}/scripts/evaluation/main_plotting.py \
#     --results_path=${ROOT_DIR}/outputs/video_scores/${MODEL} \
#     --config_file=${ROOT_DIR}/configs/configuration_hummingbirds.yaml \
#     --update \
#     --aggregate

