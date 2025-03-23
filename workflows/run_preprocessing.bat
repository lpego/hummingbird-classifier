@echo off

SET ROOT_DIR=D:\hummingbird-classifier
SET INPUT_VIDEO=D:\hummingbird-classifier\data\boris\2024_0906_185436_001.MP4
SET OUTPUT_FRAMES=D:\hummingbird-classifier\data\boris\2024_0906_185436_001

@REM ## Extract frames from videos (any format)
python scripts\preprocess\scripts\preprocess\preprocess_extract_frames.py^
    --input_loc %INPUT_VIDEO%^
    --output_loc %OUTPUT_FRAMES%

@REM @REM ## Convert or recode videos to .mp4 format
@REM python scripts\preprocess\preprocess_convert_videos.py^
@REM   --input-video-path=\data\shared\raw-video-import\data\AnnotatedVideos\^
@REM   --output-video-path=\data\shared\raw-video-import\data\RECODED_AnnotatedVideos\

@REM @REM ## Create learning sets from videos
@REM python scripts\preprocess\preprocess_prepare_learning_sets_frames.py^
@REM     --learning-set-folder=data\lset_test\^
@REM     -c configs\configuration_hummingbirds.yml