# %%

import os, sys
import cv2

import shutil
import pandas as pd
import numpy as np

from pathlib import Path
from PIL import Image

import torch

pref = ""

# %%
# print(f"is cuda: {torch.cuda.is_available()}, N GPU = {torch.cuda.device_count()}")


def count_frames(video_fold):
    n_frames = 0
    cap = cv2.VideoCapture(str(video))
    nofail, _ = cap.read()

    while nofail:
        nofail, _ = cap.read()
        n_frames += 1

    cap.release()
    return n_frames


# %%

vid_path = Path(f"{pref}../../../shared/raw-video-import/data/HummingbirdVideo/")

print(f"videos from {vid_path}")

videos = list(vid_path.glob("*.AVI"))
save_fold = Path(f"{pref}data/negative_frames/")
save_fold.mkdir(exist_ok=True, parents=True)

freq = 250
#  %%
for video in videos[::]:
    cap = cv2.VideoCapture(str(video))
    nofail, _ = cap.read()
    cc = 0

    vname = str(video).split("/")[-1][:-4]

    # n_frames = count_frames(video)
    n_frames = 0

    print(f"{vname}:: number of frames = {n_frames}")
    # %
    while nofail:
        nofail, frame = cap.read()
        if (cc % freq) == 0:
            cv2.imwrite(f"{save_fold}/{vname}_neg{cc}.jpg", frame)
            # if n_frames == 0:
            #     print(cc)
            # else:
            #     print(f"{cc * 100 / n_frames:.2}%")
        cc += 1

    cap.release()

# %%

