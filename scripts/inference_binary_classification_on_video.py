# %%
%load_ext autoreload
%autoreload 2

# standard ecosystem
import os, sys, time, copy
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import ffmpeg
import cv2

prefix = "../"
sys.path.append(f"{prefix}src")

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import random_split as RandomSplit, DataLoader, BatchSampler

# torchvision
from torchvision import models, transforms

from HummingBirdLoader import HummingBirdLoader, Denormalize
from learning_loops import train_model, visualize_model, infer_model


from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

hub_dir = Path(f"/data/shared/hummingbird-classifier/models/").resolve()
torch.hub.set_dir(hub_dir)

print(f"current torch hub directory: {torch.hub.get_dir()}")
# %%
# ATTEMPT PERFRAME VIDEO INFERENCE
# 1) load video
# 2) get frame, convert it to PIL
# 3) apply preprocessing transorms
# 4) loop through whole video and record per frame: 
# 	- probs
# 	- label
# 	- frame number 
# 	- ideally timestamp  

# %% 0 - prepare model

device =  "cuda" if torch.cuda.is_available() else "cpu"

model_folder = Path(f"{hub_dir}/vgg_newlearningsets/")
save_pos_frames = model_folder / "extracted_video_frames"
save_pos_frames.mkdir(exist_ok=True, parents=True)


model_pars = torch.load(model_folder / "model_pars_best.pt", map_location="cpu",)
model_state = torch.load(model_folder / "model_state_best.pt", map_location="cpu",)

model = model_pars
model.load_state_dict(model_state)
model.to(device);


# %% 1 - prepare video 
# 
video = Path("/data/shared/raw-video-import/data/RECODED_HummingbirdVideo/FH101_01.avi")

probe = ffmpeg.probe(video)
n_frames = int(probe["streams"][0]["nb_frames"])

augment = transforms.Compose(
    [
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224, 224), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

#%%

model.eval()
frame_list = np.arange(n_frames)

cap = cv2.VideoCapture(str(video))

df = pd.DataFrame([], index=frame_list, columns=["predicted_class", "prob_0", "prob_1"])

for ff in tqdm(frame_list):
    # print(ff)
    cap.set(1, ff)
    _, frame = cap.read()
    pframe = Image.fromarray(frame.astype('uint8'), 'RGB')
    frame = augment(pframe).to(device)

    outputs = model(frame[None, ...])
    proba = nn.Softmax(dim=1)(outputs).detach().squeeze()
    _, preds = torch.max(outputs, 1)

    # print(proba, preds)

    df.iloc[ff,:].loc["predicted_class"] = preds.cpu().numpy().squeeze()
    df.iloc[ff,:].loc["prob_0"] = proba[0].cpu().numpy()
    df.iloc[ff,:].loc["prob_1"] = proba[1].cpu().numpy()

    if df.iloc[ff,:].loc["prob_1"] > 0.8:
        pframe.save(save_pos_frames / (str(video.name)[:-4] + ".jpg"))
        # plt.figure()
        # plt.imshow(pframe)


# %%
# df
# %%
