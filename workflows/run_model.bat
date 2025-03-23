@echo off

@REM ## ------------------------------------------------------------------------------------ ##
@REM ## Definition of running parameters. 
@REM ## The path specified in ROOT_DIR is for virtual sessions on Renkulab, yours may differ! 
SET ROOT_DIR=D:\hummingbird-classifier
SET MODEL=mobilenet-v0
@REM SET LSET_FOLD="%ROOT_DIR%\data\mzb_example_data\aggregated_learning_set"
SET LSET_FOLD=%ROOT_DIR%\data\lset_test\
SET VIDEO_PATH=D:\hummingbird-classifier\data\demo
SET ANNOTATIONS=%ROOT_DIR%\data\Weinstein2018MEE_ground_truth.csv

@REM @REM ## FINETUNE MDOEL
@REM @REM ## -------------------------------------------------------------------------------- 
@REM @REM ## This is run to finetune the classification model based on the new aggreagted learning sets; it will return a new model.
@REM python %ROOT_DIR%\scripts\main_classification_finetune.py^
@REM     --input_dir %LSET_FOLD%^
@REM     --save_model %ROOT_DIR%\models\%MODEL%^
@REM     --config_file %ROOT_DIR%\configs\configuration_hummingbirds.yaml

@REM ## RUN INFERENCE
@REM ## -------------------------------------------------------------------------------- 
python %ROOT_DIR%\scripts\inference\main_score_inference.py^
    --videos_root_folder %VIDEO_PATH%^
    --model_path D:\hummingbird-classifier-vm\models\%MODEL%^
    --annotation_file %ANNOTATIONS%^
    --output_file_folder %ROOT_DIR%\outputs\demo^
    --config_file %ROOT_DIR%\configs\configuration_hummingbirds.yaml^
    --update

@REM @REM ## RUN EVALUATION
@REM @REM ## -------------------------------------------------------------------------------- 
@REM python %ROOT_DIR%\scripts\inference\main_assessment.py^
@REM     --results_path %ROOT_DIR%\outputs\video_scores\%MODEL%^
@REM     --config_file %ROOT_DIR%\configs\configuration_hummingbirds.yaml^
@REM     --update^
@REM     --aggregate^
@REM     --make_plots

@REM @REM ## PLOT RESULTS
@REM @REM ## -------------------------------------------------------------------------------- 
@REM python %ROOT_DIR%\scripts\evaluation\main_plotting.py^
@REM     --results_path %ROOT_DIR%\outputs\video_scores\%MODEL%^
@REM     --config_file %ROOT_DIR%\configs\configuration_hummingbirds.yaml^
@REM     --update^
@REM     --aggregate