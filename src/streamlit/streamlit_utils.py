import os
import time, datetime
import tkinter as tk
from tkinter import filedialog
import streamlit as st
import yaml

###
def select_folder(): 
   root = tk.Tk()
   folder_path = filedialog.askdirectory(master=root)
   root.destroy()
   return folder_path

def file_selector(folder_path):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

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
    # threshold_var=''
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
    # try: 
    #     threshold != 0 | threshold != None
    # except: 
    #     st.write('**:red[Please select a threshold]**')
    if (
        results_var != ''
        and config_var != ''
        # and update_var != ''
        # and aggregate_var != ''
        # and plots_var != ''
        # and threshold != None
    ):
        return True
    else:
        return False