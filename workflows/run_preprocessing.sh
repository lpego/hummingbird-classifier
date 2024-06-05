#!/bin/bash

## Step 1: Convert videos to mp4
# python scripts/preprocess/preprocess_convert_videos.py 
#   --input-video-path=/data/shared/raw-video-import/data/AnnotatedVideos/ 
#   --output-video-path=/data/shared/raw-video-import/data/RECODED_AnnotatedVideos/

## Step 2: Extract frames from videos and create learning sets
# python scripts/preprocess/preprocess_prepare_learning_sets_frames.py 
#     --learning-set-folder=data/lset_test/ 
#     -c configs/configuration_hummingbirds.yml