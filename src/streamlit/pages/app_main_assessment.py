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

if 'field1' not in st.session_state:
   st.session_state.field1 = False
if 'reset' not in st.session_state:
   st.session_state.reset = False

def clear():
   st.session_state.field1=False
   st.session_state.reset=False


def select_folder():
   root = tk.Tk()
   folder_path = filedialog.askdirectory(master=root)
   root.destroy()
   return folder_path

def select_folder2():
   root = tk.Tk()
   folder_path2 = filedialog.askdirectory(master=root)
   root.destroy()
   return folder_path2

def select_folder3():
   root = tk.Tk()
   folder_path3 = filedialog.askdirectory(master=root)
   root.destroy()
   return folder_path3

def select_folder4():
   root = tk.Tk()
   folder_path4 = filedialog.askdirectory(master=root)
   root.destroy()
   return folder_path4

def select_folder5():
   root = tk.Tk()
   folder_path5 = filedialog.askdirectory(master=root)
   root.destroy()
   return folder_path5


# ### Template
# selected_folder_path = st.session_state.get("folder_path", None)
# folder_select_button = st.button("Select Folder")
# if folder_select_button:
#   selected_folder_path = select_folder()
#   st.session_state.folder_path = selected_folder_path

# if selected_folder_path:
#    st.write("Selected folder path:", selected_folder_path)
   
### Actual variables to grab
results_path = st.session_state.get("folder_path", None)
folder_select_button1 = st.button("Select path to the model checkpoint")
if folder_select_button1:
  results_path = select_folder()
  st.session_state.folder_path = results_path
  st.session_state.field1 = True

if results_path and st.session_state.field1:
   st.write("Selected folder path:", results_path)

if folder_select_button1:
   st.session_state.field1 = True

config_file = st.session_state.get("folder_path2", None)
folder_select_button2 = st.button("Select path to video subfolder")
if folder_select_button2:
  config_file = select_folder2()
  st.session_state.folder_path2 = config_file

if config_file:
   st.write("Selected folder path:", config_file)


update = st.session_state.get("folder_path3", None)
folder_select_button3 = st.button("Select folder of the video frames annotation file")
if folder_select_button3:
  update = select_folder3()
  st.session_state.folder_path3 = update

if update:
   st.write("Selected folder path:", update)

aggregate = st.session_state.get("folder_path4", None)
folder_select_button4 = st.button("Select folder where results / score CSV are stored")
if folder_select_button4:
  aggregate = select_folder4()
  st.session_state.folder_path4 = aggregate

if aggregate:
   st.write("Selected folder path:", aggregate)


make_plots = st.session_state.get("folder_path5", None)
folder_select_button5 = st.button("Select path to the config file")
if folder_select_button5:
  make_plots = select_folder5()
  st.session_state.folder_path5 = make_plots

if make_plots:
  make_plots_text = st.write("Selected folder path:", make_plots)




# Choose model
option = st.selectbox(
    'Choose Model',
    ('','Model Alpha', 'Model Bravo', 'Model Charlie'))
# st.write('You selected:', option)

# Choose threshold
threshold = st.slider('Enter threshold in %', 0,100)/100

# Finish button
loadingButton = st.button('Finish')

# Clear Button
clearbtn = st.button('Clear', on_click=clear)

if clearbtn:
   st.session_state.folder_path = ''

def checkEmpty():
   results_var=''
   config_main_var=''
   update_main_var=''
   aggregate_var=''
   plots_var=''

   try:
      results_var = st.session_state.folder_path
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


   if  plots_var != '' and plots_var != '' and aggregate_var != '' and update_main_var != '' and config_main_var != '' and results_var != '' and option != '':

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
    time.sleep(2)
    my_bar.empty()

    # Write the CSV
    file = open('predictions.csv', 'a')    
    # file.write(st.session_state.dirname + ', ')
 #   file.write(selected_folder_path + ', ')
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
    
    st.write('Success')
elif loadingButton:
    st.write('**:red[Please fill out all fields]**')
