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

def select_folder14():
   root = tk.Tk()
   root.withdraw()
   folder_path14 = filedialog.askdirectory(master=root)
   root.destroy()
   return folder_path14

def select_folder15():
   root = tk.Tk()
   root.withdraw()
   folder_path15 = filedialog.askdirectory(master=root)
   root.destroy()
   return folder_path15

def select_folder16():
   root = tk.Tk()
   root.withdraw()
   folder_path16 = filedialog.askdirectory(master=root)
   root.destroy()
   return folder_path16

def select_folder17():
   root = tk.Tk()
   root.withdraw()
   folder_path17 = filedialog.askdirectory(master=root)
   root.destroy()
   return folder_path17

def select_folder18():
   root = tk.Tk()
   root.withdraw()
   folder_path18 = filedialog.askdirectory(master=root)
   root.destroy()
   return folder_path18

def select_folder19():
   root = tk.Tk()
   root.withdraw()
   folder_path19 = filedialog.askdirectory(master=root)
   root.destroy()
   return folder_path19


# ### Template
# selected_folder_path = st.session_state.get("folder_path", None)
# folder_select_button = st.button("Select Folder")
# if folder_select_button:
#   selected_folder_path = select_folder()
#   st.session_state.folder_path = selected_folder_path

# if selected_folder_path:
#    st.write("Selected folder path:", selected_folder_path)
   
### Actual variables to grab
model_path = st.session_state.get("folder_path14", None)
folder_select_button14 = st.button("Select path to the model checkpoint")
if folder_select_button14:
  model_path = select_folder14()
  st.session_state.folder_path14 = model_path

if model_path:
   st.write("Selected folder path:", model_path)

videos_root_folder = st.session_state.get("folder_path15", None)
folder_select_button15 = st.button("Select path to video subfolder")
if folder_select_button15:
  videos_root_folder = select_folder15()
  st.session_state.folder_path15 = videos_root_folder

if videos_root_folder:
   st.write("Selected folder path:", videos_root_folder)

annotation_file = st.session_state.get("folder_path16", None)
folder_select_button16 = st.button("Select folder of the video frames annotation file")
if folder_select_button16:
  annotation_file = select_folder16()
  st.session_state.folder_path16 = annotation_file

if annotation_file:
   st.write("Selected folder path:", annotation_file)

output_file_folder = st.session_state.get("folder_path17", None)
folder_select_button17 = st.button("Select folder where results / score CSV are stored")
if folder_select_button17:
  output_file_folder = select_folder17()
  st.session_state.folder_path17 = output_file_folder

if output_file_folder:
   st.write("Selected folder path:", output_file_folder)

update = st.session_state.get("folder_path18", None)
folder_select_button18 = st.button("Select flag to force recomputing the scores")
if folder_select_button18:
  update = select_folder18()
  st.session_state.folder_path18 = update

if update:
  st.write("Selected folder path:", update)

config_file = st.session_state.get("folder_path19", None)
folder_select_button19 = st.button("Select path to the config file")
if folder_select_button19:
  config_file = select_folder19()
  st.session_state.folder_path19 = config_file

if config_file:
   st.write("Selected folder path:", config_file)

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
   model_var=''
   videos_var=''
   annotation_var=''
   output_var=''
   update_inf_var=''
   config_inf_var=''

   try:
      model_var = st.session_state.folder_path14
   except:
      st.write('**:red[Please select path to learning sets]**')

   try:
      videos_var = st.session_state.folder_path15
   except:
      st.write('**:red[Please select path to video subfolder]**')

   try:
      annotation_var = st.session_state.folder_path16
   except:
      st.write('**:red[Please select folder of the video frames annotation file]**')

   try:
      output_var = st.session_state.folder_path17
   except:
      st.write('**:red[Please select folder where results / score CSV are stored]**')

   try:
      update_inf_var = st.session_state.folder_path18
   except:
      st.write('**:red[Please select flag to force recomputing the scores]**')

   try:
      config_inf_var = st.session_state.folder_path19
   except:
      st.write('**:red[Please select path to config file]**')


   if  config_inf_var != '' and update_inf_var != '' and output_var != '' and annotation_var != '' and videos_var != '' and model_var != '' and option != '':

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
    time.sleep(16)
    my_bar.empty()

    # Write the CSV
    file = open('predictions.csv', 'a')    
    # file.write(st.session_state.dirname + ', ')
#    file.write(selected_folder_path + ', ')
    file.write(option + ', ')
    file.write(str(threshold) + ', ' + '\n')
    
    ###
#    import yaml
#    from pathlib import Path
#
#    ROOT_DIR = Path("D:\hummingbird-classifier") #Path("/home/jovyan/work/hummingbird-classifier")
#
#    arguments = {
#        "results_path": results_path, 
#        "config_file": config_file, 
#    #    [...]]
#        }
#        
#    with open(str(arguments["config_file"]), "r") as f:
#        cfg = yaml.load(f, Loader=yaml.FullLoader)
#
#    from src.utils import cfg_to_arguments
#
#    args = cfg_to_arguments(arguments)
#    cfg = cfg_to_arguments(cfg)
#
#    from scripts.inference.main_assessment import main as hb_inference
#
#    hb_inference(args, cfg)
#    
    st.write('Success')
elif loadingButton:
    st.write('**:red[Please fill out all fields]**')
