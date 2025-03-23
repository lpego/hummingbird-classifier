@echo off

SET ROOT_DIR=D:\hummingbird-classifier
SET INPUT_CSV=D:\hummingbird-classifier\outputs\mobilenet-v0\Alaspungo.csv
SET THRESHOLD=.95
SET OUT_FOLDER=D:\hummingbird-classifier\outputs\filtered_images2
SET INPUT_DIR=D:\hummingbird-classifier\data\demo\testdir_subdirs\Alaspungo

@REM ## Copy positive images to out_folder
python scripts\postprocess\postprocess_move_positive_frames.py^
    --input_csv %INPUT_CSV%^
    --threshold %THRESHOLD%^
    --out_folder %OUT_FOLDER%^
    --verbose True

@REM ## Delete images from input folder
python scripts\postprocess\postprocess_cleanup.py^
    --input_dir %INPUT_DIR%^
    --input_csv %INPUT_CSV%^
    --threshold %THRESHOLD%^
    --out_folder %OUT_FOLDER%