import os
import time
import tkinter as tk
from tkinter import filedialog
import streamlit as st

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
    st.session_state.folder_path1=None
    st.session_state.folder_path2=None
    st.session_state.folder_path3=None
    st.session_state.folder_path4=None
    st.session_state.folder_path5=None
    st.session_state.threshold=0
    st.session_state.model_select=""
    # st.session_state.reset=False
    
def checkEmpty():
    results_var=''
    config_main_var=''
    update_main_var=''
    aggregate_var=''
    plots_var=''
    try:
        results_var = st.session_state.folder_path1
    except:
        st.write('**:red[Please select path to the model checkpoint]**')
    try:
        config_main_var = st.session_state.folder_path2
    except:
        st.write('**:red[Please select path to video subfolder]**')
    try:
        update_main_var = st.session_state.folder_path3
    except:
        st.write('**:red[Please select folder of the video frames annotation file]**')
    try:
        aggregate_var = st.session_state.folder_path4
    except:
        st.write('**:red[Please select folder where results / score CSV are stored]**')
    try:
        plots_var = st.session_state.folder_path5
    except:
        st.write('**:red[Please select path to the config file]**')
    if (
        results_var != ''
        and config_main_var != ''
        and update_main_var != ''
        and aggregate_var != ''
        and plots_var != ''
    ):
        return True
    else:
        return False
    
### Actual variables to grab
# # ### Template
# selected_folder_path = st.session_state.get("folder_path", None)
# folder_select_button = st.button("Select Folder")
# if folder_select_button:
#    selected_folder_path = select_folder()
#    st.session_state.folder_path = selected_folder_path

# if selected_folder_path:
#    st.write("Selected folder path:", selected_folder_path)

results_path = st.session_state.get("folder_path1")
folder_select_button1 = st.button("Select path to the model checkpoint")
if folder_select_button1:
    results_path = select_folder()
    st.session_state.folder_path1 = results_path
if results_path and st.session_state.folder_path1:
    st.write("Selected folder path:", results_path)

config_file = st.session_state.get("folder_path2", None)
folder_select_button2 = st.button("Select path to video subfolder")
if folder_select_button2:
    config_file = select_folder()
    st.session_state.folder_path2 = config_file
if config_file:
    st.write("Selected folder path:", config_file)

update = st.session_state.get("folder_path3", None)
folder_select_button3 = st.button("Select folder of the video frames annotation file")
if folder_select_button3:
    update = select_folder()
    st.session_state.folder_path3 = update
if update:
    st.write("Selected folder path:", update)

aggregate = st.session_state.get("folder_path4", None)
folder_select_button4 = st.button("Select folder where results / score CSV are stored")
if folder_select_button4:
    aggregate = select_folder()
    st.session_state.folder_path4 = aggregate
if aggregate:
    st.write("Selected folder path:", aggregate)

make_plots = st.session_state.get("folder_path5", None)
folder_select_button5 = st.button("Select path to the config file")
if folder_select_button5:
    make_plots = select_folder()
    st.session_state.folder_path5 = make_plots
if make_plots:
    make_plots_text = st.write("Selected folder path:", make_plots)

### Choose model
option = st.selectbox(
    'Choose Model',
    ('','Model Alpha', 'Model Bravo', 'Model Charlie'), 
    key='model_select')
# st.write('You selected:', option)

### Choose threshold
threshold = st.slider('Enter threshold in %', 
                      0,100, 
                      key='threshold') / 100

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

    ### Write the CSV
    with open('streamlit_log.csv', 'a') as file:
        # file.write(st.session_state.dirname + ', ')
        # file.write(selected_folder_path + ', ')
        file.write(option + ', ')
        file.write(str(threshold) + ', ' + '\n')   
    st.write('Success, appending run parameters to "streamlit_log.csv"')

elif loadingButton:
    st.write('**:red[Please fill out all fields!]**')
    
### Clear Button
clearbtn = st.button('Clear', on_click=clear)
