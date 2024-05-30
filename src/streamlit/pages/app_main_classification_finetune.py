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

def select_folder7():
   root = tk.Tk()
   root.withdraw()
   folder_path7 = filedialog.askdirectory(master=root)
   root.destroy()
   return folder_path7

def select_folder8():
   root = tk.Tk()
   root.withdraw()
   folder_path8 = filedialog.askdirectory(master=root)
   root.destroy()
   return folder_path8

def select_folder9():
   root = tk.Tk()
   root.withdraw()
   folder_path9 = filedialog.askdirectory(master=root)
   root.destroy()
   return folder_path9


# ### Template
# selected_folder_path = st.session_state.get("folder_path", None)
# folder_select_button = st.button("Select Folder")
# if folder_select_button:
#   selected_folder_path = select_folder()
#   st.session_state.folder_path = selected_folder_path

# if selected_folder_path:
#    st.write("Selected folder path:", selected_folder_path)
   
### Actual variables to grab


config_file = st.session_state.get("folder_path7", None)
folder_select_button7 = st.button("Select path to config file with per-script args")
if folder_select_button7:
  config_file = select_folder7()
  st.session_state.folder_path7 = config_file

if config_file:
   st.write("Selected folder path:", config_file)

input_dir = st.session_state.get("folder_path8", None)
folder_select_button8 = st.button("Select path with images for training")
if folder_select_button8:
  input_dir = select_folder8()
  st.session_state.folder_path8 = input_dir

if input_dir:
   st.write("Selected folder path:", input_dir)

save_model = st.session_state.get("folder_path9", None)
folder_select_button9 = st.button("Select path to where to save model checkpoints")
if folder_select_button9:
  save_model = select_folder9()
  st.session_state.folder_path9 = save_model

if save_model:
   st.write("Selected folder path:", save_model)


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
   config_main_var=''
   input_main_var=''
   save_var=''

   try:
      config_main_var = st.session_state.folder_path7
   except:
      st.write('**:red[Please path to config file with per-script args]**')

   try:
      input_main_var = st.session_state.folder_path8
   except:
      st.write('**:red[Please select path with images for training]**')

   try:
      save_var = st.session_state.folder_path9
   except:
      st.write('**:red[Please select path to where to save model checkpoints]**')

   if  config_main_var != '' and input_main_var != '' and save_var != '' and option != '':

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
    time.sleep(9)
    my_bar.empty()

    # Write the CSV
    file = open('predictions.csv', 'a')    
    # file.write(st.session_state.dirname + ', ')
  #  file.write(selected_folder_path + ', ')
    file.write(option + ', ')
    file.write(str(threshold) + ', ' + '\n')
    
    ###
   # import yaml
   # from pathlib import Path
#
   # ROOT_DIR = Path("D:\hummingbird-classifier") #Path("/home/jovyan/work/hummingbird-classifier")
#
   # arguments = {
   #     "results_path": results_path, 
   #     "config_file": config_file, 
   # #    [...]]
   #     }
   #     
   # with open(str(arguments["config_file"]), "r") as f:
   #     cfg = yaml.load(f, Loader=yaml.FullLoader)
#
   # from src.utils import cfg_to_arguments
#
   # args = cfg_to_arguments(arguments)
   # cfg = cfg_to_arguments(cfg)
#
   # from scripts.inference.main_assessment import main as hb_inference

    hb_inference(args, cfg)
    
    st.write('Success')
elif loadingButton:
    st.write('**:red[Please fill out all fields]**')
