import streamlit as st
import tkinter as tk
from tkinter import filedialog
import os
import time

# run the file: streamlit run app.py

st.title ('BioDetect')
# file uploader by streamlit run through folder selected per loop
# give path to folder
# select from txt array
# write to csv/json (folder path, treshold value, model name)

def select_folder12():
   root = tk.Tk()
   root.withdraw()
   folder_path12 = filedialog.askdirectory(master=root)
   root.destroy()
   return folder_path12

def select_folder13():
   root = tk.Tk()
   root.withdraw()
   folder_path13 = filedialog.askdirectory(master=root)
   root.destroy()
   return folder_path13


# ### Template
# selected_folder_path = st.session_state.get("folder_path", None)
# folder_select_button = st.button("Select Folder")
# if folder_select_button:
#   selected_folder_path = select_folder()
#   st.session_state.folder_path = selected_folder_path

# if selected_folder_path:
#    st.write("Selected folder path:", selected_folder_path)

# Clear Button (not working)
#clearButton = st.button('Clear')
#
#if clearButton:
#    if 'folder_path12' and 'folder_path13' not in st.session_state:
#        st.session_state.folder_path12 = ''
#        st.session_state.folder_path13 = ''
#        st.session_state.option = ''
#        st.session_state.threshold = 0
   
### Actual variables to grab
learning_set_folder = st.session_state.get("folder_path12", None)
folder_select_button12 = st.button("Select path to learning sets (data subfolder)")
if folder_select_button12:
  learning_set_folder = select_folder12()
  st.session_state.folder_path12 = learning_set_folder

if learning_set_folder:
   st.write("Selected folder path:", learning_set_folder)

config = st.session_state.get("folder_path13", None)
folder_select_button13 = st.button("Select path to config file")
if folder_select_button13:
  config = select_folder13()
  st.session_state.folder_path13 = config

if config:
   st.write("Selected folder path:", config)

# Choose model
option = st.selectbox(
    'Choose Model',
    ('','Model Alpha', 'Model Bravo', 'Model Charlie'))
# st.write('You selected:', option)

# Choose threshold
threshold = st.slider('Enter threshold in %', 0,100)/100

# Finish button
loadingButton = st.button('Finish')

def checkEmpty():
    config_var=''
    learning_set_var=''

    try:
        learning_set_var = st.session_state.folder_path12
    except:
       st.write('**:red[Please select path to learning sets]**')

    try:
        config_var = st.session_state.folder_path13
    except:
       st.write('**:red[Please select path to config file]**')


    if  config_var != '' and  learning_set_var != '' and option != '':

        return True
    else:

        return False

if loadingButton and checkEmpty():

    # Progress bar
    progress_text = 'Loading...'
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(3)
    my_bar.empty()

    # Write the CSV 
    file = open('predictions.csv', 'a')    
    # file.write(st.session_state.dirname + ', ')
   # file.write(selected_folder_path + ', ')
    file.write(option + ', ')
    file.write(str(threshold) + ', ' + '\n')
    
    ### yaml import does not work for me
 #   import yaml
 #   from pathlib import Path
#
 #   ROOT_DIR = Path("D:\hummingbird-classifier") #Path("/home/jovyan/work/hummingbird-classifier")
#
 #   arguments = {
 #       "results_path": results_path, 
 #       "config_file": config_file, 
 #   #    [...]]
 #       }
 #       
 #   with open(str(arguments["config_file"]), "r") as f:
 #       cfg = yaml.load(f, Loader=yaml.FullLoader)
#
 #   from src.utils import cfg_to_arguments
#
 #   args = cfg_to_arguments(arguments)
 #   cfg = cfg_to_arguments(cfg)
#
 #   from scripts.inference.main_assessment import main as hb_inference
#
 #   hb_inference(args, cfg)
    
    st.write('Success')
elif loadingButton:
    st.write('**:red[Please fill out all fields]**')
