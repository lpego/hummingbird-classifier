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

def select_folder10():
   root = tk.Tk()
   root.withdraw()
   folder_path10 = filedialog.askdirectory(master=root)
   root.destroy()
   return folder_path10

def select_folder11():
   root = tk.Tk()
   root.withdraw()
   folder_path11 = filedialog.askdirectory(master=root)
   root.destroy()
   return folder_path11


# ### Template
# selected_folder_path = st.session_state.get("folder_path", None)
# folder_select_button = st.button("Select Folder")
# if folder_select_button:
#   selected_folder_path = select_folder()
#   st.session_state.folder_path = selected_folder_path

# if selected_folder_path:
#    st.write("Selected folder path:", selected_folder_path)
   
### Actual variables to grab
input_video_path = st.session_state.get("folder_path10", None)
folder_select_button10 = st.button("Select path to input video folder")
if folder_select_button10:
  input_video_path = select_folder10()
  st.session_state.folder_path10 = input_video_path

if input_video_path:
   st.write("Selected folder path:", input_video_path)

output_video_path = st.session_state.get("folder_path11", None)
folder_select_button11 = st.button("Select path to output video folder")
if folder_select_button11:
  output_video_path = select_folder11()
  st.session_state.folder_path11 = output_video_path

if output_video_path:
   st.write("Selected folder path:", output_video_path)

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
   video_input_var=''
   video_output_var=''
  
   try:
      video_input_var = st.session_state.folder_path10
   except:
      st.write('**:red[Please select path to input video folder]**')

   try:
      video_output_var = st.session_state.folder_path11
   except:
      st.write('**:red[Please select path to output video folder]**')


   if  video_input_var != '' and video_output_var != '' and option != '':

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
 #   file.write(selected_folder_path + ', ')
    file.write(option + ', ')
    file.write(str(threshold) + ', ' + '\n')
    
    ###
 # import yaml
 # from pathlib import Path

 # ROOT_DIR = Path("D:\hummingbird-classifier") #Path("/home/jovyan/work/hummingbird-classifier")

 # arguments = {
 #     "results_path": results_path, 
 #     "config_file": config_file, 
 # #    [...]]
 #     }
 #     
 # with open(str(arguments["config_file"]), "r") as f:
 #     cfg = yaml.load(f, Loader=yaml.FullLoader)

 # from src.utils import cfg_to_arguments

 # args = cfg_to_arguments(arguments)
 # cfg = cfg_to_arguments(cfg)

 # from scripts.inference.main_assessment import main as hb_inference

 # hb_inference(args, cfg)
    
    st.write('Success')
elif loadingButton:
    st.write('**:red[Please fill out all fields]**')
