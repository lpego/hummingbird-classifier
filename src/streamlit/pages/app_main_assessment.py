import os
import time, datetime
import tkinter as tk
from tkinter import filedialog
import streamlit as st
import pandas as pd
import yaml

st.title ('BioDetect')
# run the file: streamlit run app.py
# file uploader by streamlit run through folder selected per loop
# give path to folder
# select from txt array
# write to csv/json (folder path, threshold value, model name)

### Custom functions
def select_folder(): 
   root = tk.Tk()
   folder_path = filedialog.askdirectory(master=root)
   root.destroy()
   return folder_path

def clear():
    st.session_state.folder_path=None
    st.session_state.model_path=None
    st.session_state.config_file=None
    st.session_state.folder_path3=None
    st.session_state.results_path=None
    st.session_state.config_file=None
    st.session_state.threshold=0
    st.session_state.model_select=""
    
def checkEmpty():   
    results_var=''
    config_var=''
    # update_var=''
    # aggregate_var=''
    # plots_var= ''
    try:
        results_var = st.session_state.results_path
    except:
        st.write('**:red[Please select folder where results / scores are stored]**')
    try:
        config_var = st.session_state.config_file
    except:
        st.write('**:red[Please select path to the configuration file]**')
    # try:
    #     update_var = st.session_state.update
    # except:
    #     st.write('**:red[Please select whther to update metrics for all videos]**')
    # try:
    #     aggregate_var = st.session_state.aggregate
    # except:
    #     st.write('**:red[Please select whether to aggregate metrics per folder]**')
    # try:
    #     plots_var = st.session_state.plots
    # except:
    #     st.write('**:red[Please select whether to plot metrics]**')
    if (
        results_var != ''
        and config_var != ''
        # and update_var != ''
        # and aggregate_var != ''
        # and plots_var != ''
    ):
        return True
    else:
        return False

debug = True # set to False for regular operation
streamlit_log = {"app_main_assessment": {"start_time": datetime.datetime.now()}}

### Actual variables to grab
# # ### Template
# selected_folder_path = st.session_state.get("folder_path", None)
# folder_select_button = st.button("Select Folder")
# if folder_select_button:
#    selected_folder_path = select_folder()
#    st.session_state.folder_path = selected_folder_path
# if selected_folder_path:
#    st.write("Selected folder path:", selected_folder_path)

results_path = st.session_state.get("results_path", None)
folder_select_button4 = st.button("Select folder where results / score CSV are stored")
if folder_select_button4:
    results_path = select_folder()
    st.session_state.results_path = results_path
if results_path:
    st.write("Selected folder path:", results_path)
    streamlit_log["app_main_assessment"]["results_path"] = results_path

config_file = st.session_state.get("config_file", None)
folder_select_button5 = st.button("Select path to the config file")
if folder_select_button5:
    config_file = select_folder()
    st.session_state.config_file = config_file
if config_file:
    config_file_text = st.write("Selected folder path:", config_file)
    streamlit_log["app_main_assessment"]["config_file"] = config_file

### Choose model
option = st.selectbox(
    'Choose Model',
    ('','Model Alpha', 'Model Bravo', 'Model Charlie'), 
    key='model_select')
# st.write('You selected:', option)
if option:
    streamlit_log["app_main_assessment"]["option"] = option

###
update = st.checkbox("Recompute all metrics for videos in the folder", 
                     value=False, 
                     key="update"
                     )
if update: 
    streamlit_log["app_main_assessment"]["update"] = update

aggregate = st.checkbox("Aggregate metrics for all videos in the folder", 
                        value=False,
                        key="aggregate"
                        )
if aggregate: 
    streamlit_log["app_main_assessment"]["aggregate"] = aggregate
    
plots = st.checkbox("Plot metrics", 
                    value=False,
                    key="plots"
                    )
if plots: 
    streamlit_log["app_main_assessment"]["plots"] = plots

### Choose threshold
threshold = st.slider('Enter threshold in %', 
                      0,100, 
                      key='threshold') / 100
if threshold:
    streamlit_log["app_main_assessment"]["threshold"] = threshold

### Testing printouts
if debug: 
    st.write("DEBUGGING INFO:", streamlit_log)

### Run button
loadingButton = st.button('Run')

if loadingButton and checkEmpty():
    ### Progress bar
    progress_text = 'Loading...'
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(2)
    my_bar.empty()

    # ### Write the YAML
    streamlit_log["app_main_assessment"]["end_time"] = datetime.datetime.now()
    with open('streamlit_log.yaml', 'a') as outfile:
        yaml.dump(streamlit_log, outfile, sort_keys=False)
    del(streamlit_log)
    st.write('Success, appending run parameters to "streamlit_log.yaml"')

elif loadingButton:
    st.write('**:red[Please fill out all fields!]**')
    
### Clear Button
clearbtn = st.button('Clear', on_click=clear)

### 🐛 KNOWN BUG: try...except blocks don't catch exceptions after clear button is pressed! 