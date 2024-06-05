#!/bin/bash 

# ## ------------------------------------------------------------------------------------ ##
## Definition of running parameters, yours may differ! 
ROOT_DIR="D:/hummingbird-classifier"
cd ${ROOT_DIR} 

## Activate mamba anvironment; can also use conda instead
mamba activate humb
# mamba activate base

## Run the streamlit app
streamlit run src/streamlit/start.py