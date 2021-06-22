# %%
# %load_ext autoreload
# %autoreload 2

# standard ecosystem
import os, sys, time, copy
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import datetime

import ffmpeg
import cv2

prefix = ""
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

device = "cuda" if torch.cuda.is_available() else "cpu"

model_folder = Path(f"{hub_dir}/vgg_newlearningsets/")
save_pos_frames = model_folder / "extracted_video_frames"
save_pos_frames.mkdir(exist_ok=True, parents=True)


model_pars = torch.load(model_folder / "model_pars_best.pt", map_location="cpu",)
model_state = torch.load(model_folder / "model_state_best.pt", map_location="cpu",)

model = model_pars
model.load_state_dict(model_state)
model.to(device)
model.eval()

augment = transforms.Compose(
    [
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224, 224), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

# %% 1 - prepare video

videos = list(
    Path("/data/shared/raw-video-import/data/RECODED_HummingbirdVideo/").glob("*.avi")
)
videos.sort()

SAVE_POSITIVES = False
for iv, video in enumerate(videos[:2]):
    # tqdm.write(f"{video.name}, {iv+1}/{len(videos)}")
    # print()
    save_frames = save_pos_frames / (str(video.name)[:-4])
    save_frames.mkdir(exist_ok=True, parents=True)

    probe = ffmpeg.probe(video)
    n_frames = int(probe["streams"][0]["nb_frames"])
    framerate = float(
        eval(probe["streams"][0]["avg_frame_rate"])
    )  # * float(eval(probe["streams"][0]["time_base"]))
    duration_s = str(
        datetime.timedelta(seconds=float(probe["streams"][0]["duration"]))
    )[:-4]
    # print(n_frames, length, framerate, duration_s)

    # frame_list = np.arange(15000, 16000, 1)#n_frames)
    frame_list = np.arange(0, n_frames, 2)

    cap = cv2.VideoCapture(str(video))

    df = pd.DataFrame(
        [],
        index=frame_list,
        columns=[
            "frame_number",
            "timestamp_video",
            "predicted_class",
            "prob_0",
            "prob_1",
        ],
    )

    for i, ff in enumerate(
        tqdm(frame_list, desc=f"{video.name}, {iv+1}/{len(videos)}")
    ):
        # print(ff)
        cap.set(1, ff)
        _, frame = cap.read()
        pframe = Image.fromarray(frame.astype("uint8"), "RGB")
        frame = augment(pframe).to(device)

        outputs = model(frame[None, ...])
        proba = nn.Softmax(dim=1)(outputs).detach().squeeze()
        _, preds = torch.max(outputs, 1)

        # print(proba, preds)

        time = str(datetime.timedelta(seconds=ff * 1 / framerate))
        df.iloc[i, :].loc["timestamp_video"] = time[:-5] + "/" + duration_s
        df.iloc[i, :].loc["frame_number"] = ff
        df.iloc[i, :].loc["predicted_class"] = preds.cpu().numpy().squeeze()
        df.iloc[i, :].loc["prob_0"] = proba[0].cpu().numpy()
        df.iloc[i, :].loc["prob_1"] = proba[1].cpu().numpy()

        if SAVE_POSITIVES:
            if df.iloc[i, :].loc["prob_1"] > 0.5:
                pframe.save(save_frames / (str(ff) + ".jpg"))
                # plt.figure()
                # plt.imshow(pframe)

    #  save DF
    df.to_csv(save_frames / "summary.csv")
# %%


# frame_list = (-df[df.predicted_class == 1].prob_1).sort_values().index.astype(int)

# cap = cv2.VideoCapture(str(video))

# for ff in frame_list[:20]:
#     # print(ff)
#     cap.set(1, ff)
#     _, frame = cap.read()
#     pframe = Image.fromarray(frame.astype('uint8'), 'RGB')
#     plt.figure()
#     plt.title(f"{ff}, {df.loc[ff,'prob_1']:.2f}")
#     plt.imshow(pframe)

# # %%
# plt.figure(figsize=(15,7))
# plt.plot(df.iloc[15000:17000].prob_1)
# # %%

# im_1 = Image.open("/data/shared/hummingbird-classifier/models/vgg_newlearningsets/extracted_video_frames/FH101_01/15000.jpg").convert("RGB")
# im_2 = Image.open("/data/shared/hummingbird-classifier/models/vgg_newlearningsets/extracted_video_frames/FH101_01/15066.jpg").convert("RGB")

# plt.imshow(np.array(im_2) - np.array(im_1) + 125)
# %%
