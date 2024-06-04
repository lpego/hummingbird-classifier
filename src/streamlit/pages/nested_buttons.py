import os
import time, datetime
import tkinter as tk
from tkinter import filedialog
import streamlit as st
import pandas as pd
import yaml
from streamlit_utils import select_folder, file_selector

# ### Custom functions
# def select_folder(): 
#    root = tk.Tk()
#    folder_path = filedialog.askdirectory(master=root)
#    root.destroy()
#    return folder_path

# def file_selector(folder_path):
#     filenames = os.listdir(folder_path)
#     selected_filename = st.selectbox('Select a file', filenames)
#     return os.path.join(folder_path, selected_filename)

### Set up logging
debug = True # set to False for regular operation
streamlit_log = {"nested_buttons": {"start_time": datetime.datetime.now()}}

### Threshold slider to test
threshold0 = st.slider('Enter threshold in %',
                      0,100, 
                      key='threshold0') / 100
if threshold0: 
    streamlit_log["nested_buttons"]["threshold0"] = threshold0

### Set up columns (to fake indentation)
st.write("Select model")
col1, col2 = st.columns([1, 4])

### Choose model
### Set up nested buttons truth source
if not "model_folder" in st.session_state: 
    st.session_state["model_folder"] = False
### Select model folder
with col1: 
    model_folder = st.session_state.get("model_folder", None)
    model_folder_button = st.button("Select folder where models are stored")
    if model_folder_button:
        model_folder = select_folder()
        st.session_state.model_folder = model_folder
### Select file in model_folder
with col2: 
    if st.session_state.model_folder is not False:
        model_path = file_selector(model_folder)
        st.write('You selected `%s`' % model_path)
        streamlit_log["nested_buttons"]["model_path"] = model_path

### Choose threshold
threshold = st.slider('Enter threshold in %',
                      0,100,
                      key='threshold') / 100
if threshold:
    streamlit_log["nested_buttons"]["threshold"] = threshold

option = st.selectbox(
    'Choose Model',
    ('','Model Alpha', 'Model Bravo', 'Model Charlie'),
    key='model_select')
# st.write('You selected:', option)
if option:
    streamlit_log["nested_buttons"]["option"] = option

if debug: 
    st.write("DEBUGGING INFO:", streamlit_log)

### Run button
loadingButton = st.button('Run')

if loadingButton:
    ### Progress bar
    progress_text = 'Loading...'
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(2)
    my_bar.empty()

    ### Write the YAML
    streamlit_log["nested_buttons"]["end_time"] = datetime.datetime.now()
    with open('streamlit_log.yaml', 'a') as outfile:
        yaml.dump(streamlit_log, outfile, sort_keys=False)
    del(streamlit_log)
    st.write('Success, appending run parameters to "streamlit_log.yaml"')

elif loadingButton:
    st.write('**:red[Please fill out all fields!]**')
