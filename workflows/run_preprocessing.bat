@echo off

@REM @REM ## Step 1: Convert videos to mp4
@REM python scripts\preprocess\preprocess_convert_videos.py^
@REM   --input-video-path=\data\shared\raw-video-import\data\AnnotatedVideos\^
@REM   --output-video-path=\data\shared\raw-video-import\data\RECODED_AnnotatedVideos\

@REM @REM ## Step 2: Extract frames from videos and create learning sets
@REM python scripts\preprocess\preprocess_prepare_learning_sets_frames.py^
@REM     --learning-set-folder=data\lset_test\^
@REM     -c configs\configuration_hummingbirds.yml