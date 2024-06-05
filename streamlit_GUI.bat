@echo off

@REM ## ------------------------------------------------------------------------------------ ##
@REM ## Running Streamlit Graphical User Interface for Hummingbirds Detector. 
@REM SET ROOT = D:\hummingbird-classifier
@REM chdir %ROOT%
@REM # Activate conda environment; can use conda instead of mamba too
call mamba activate humb
@REM mamba activate base

@REM # Run Streamlit app
call streamlit run src\streamlit\start.py

@REM Keep terminal open while streamlit executes
@pause