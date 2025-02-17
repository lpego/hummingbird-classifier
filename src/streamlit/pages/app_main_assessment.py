import os, sys
import time, datetime
import tkinter as tk
from tkinter import filedialog
import streamlit as st
import yaml
from streamlit_utils import select_folder, file_selector

sys.path.append(r'..\..\..')
from scripts.inference.main_assessment import per_video_assessment

st.title ('Assessment - main_assessment.py')

### Custom functions
def clear():
    st.session_state.results_path=None
    st.session_state.config_folder=False
    st.session_state.config_file=None
    st.session_state.update=False
    st.session_state.aggregate=False
    st.session_state.plots=False
    
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
st.write("Path to the video_scores sub-folder, specific to a model; this folder should contain CSV files with the raw pipeline scores for each video")
results_path_button = st.button("Select folder")
if results_path_button:
    results_path = select_folder()
    st.session_state.results_path = results_path
if results_path:
    st.write("Selected results path `%s`" %  results_path)
    streamlit_log["app_main_assessment"]["results_path"] = results_path

### Config file nested folder-file select
st.write("Select configuration file")
col1, col2 = st.columns([1, 4])
### Set up nested buttons states
if not "config_folder" in st.session_state:
    st.session_state["config_folder"] = False
### Select config folder
with col1: 
    config_folder = st.session_state.get("config_folder", None)
    # st.write("Select folder where config files are stored")
    config_folder_button = st.button("Select config folder")
    if config_folder_button:
        config_folder = select_folder()
        st.session_state.config_folder = config_folder
### Select file in config_folder
with col2:
    if st.session_state.config_folder is not False:
        config_file = file_selector(config_folder)
        st.session_state.config_file = config_folder
        st.write('You selected `%s`' % config_file)
        streamlit_log["app_main_assessment"]["config_file"] = config_file

### Checkboxes
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

### Testing printouts
if debug: 
    st.write("DEBUGGING INFO:", streamlit_log)

### Run button
loadingButton = st.button('Run')

if loadingButton and checkEmpty():
    # ### Progress bar
    # progress_text = 'Loading...'
    # my_bar = st.progress(0, text=progress_text)
    # for percent_complete in range(100):
    #     time.sleep(0.01)
    #     my_bar.progress(percent_complete + 1, text=progress_text)
    # time.sleep(2)
    # my_bar.empty()
    
    per_video_assessment(results_path, 3, config_file)
    my_bar = st.progress(0, text='Running...')

    ### Write the YAML
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