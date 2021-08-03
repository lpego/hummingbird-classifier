# %%
# %load_ext autoreload
# %autoreload 2

import os, sys, time, copy
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

# from tqdm import tqdm
import datetime
import time
from matplotlib import pyplot as plt

from joblib import Parallel, delayed

import ffmpeg
import cv2

prefix = "../"
sys.path.append(f"{prefix}src")


# %%
video_name = "FH102_02"
# %%
df = pd.read_csv("../data/Weinstein2018MEE_ground_truth.csv")
pos_vid = df[df.Video == video_name].copy()
pos_vid.Truth = pos_vid.Truth.replace({"Positive": 1, "Negative": 0})

# %%
def norm_frame(frin):
    frame = frin - np.mean(frin, axis=(0, 1))
    # frame = frame / np.std(frin, axis=(0, 1))

    frame = (frame - np.min(frame, axis=(0, 1))) / (
        np.max(frame, axis=(0, 1)) - np.min(frame, axis=(0, 1))
    )
    return frame


video = Path(
    f"/data/shared/raw-video-import/data/RECODED_AnnotatedVideos/{video_name}.avi"
)
videos = [video]

for vi, video in enumerate(videos[:1]):
    probe = ffmpeg.probe(video)
    n_frames = int(probe["streams"][0]["nb_frames"])
    w, h = int(probe["streams"][0]["width"]), int(probe["streams"][0]["height"])

    framerate = float(eval(probe["streams"][0]["avg_frame_rate"]))
    duration_s = str(
        datetime.timedelta(seconds=float(probe["streams"][0]["duration"]))
    )[:-4]

    frame_list = np.arange(970, 990, 1)

    # print(n_frames, length , framerate, duration_s)
    fr_rgb = np.empty((w * h, 3, len(frame_list)), dtype=np.uint8)

    cap = cv2.VideoCapture(str(video))

    _, ft0 = cap.read()
    hsv = np.zeros_like(ft0)

    gft0 = cv2.cvtColor(ft0, cv2.COLOR_BGR2GRAY)
    hsv[..., 1] = 255

    for fi, ff in enumerate(frame_list):

        cap.set(1, ff)
        ret, ft1 = cap.read()
        gft1 = cv2.cvtColor(ft1, cv2.COLOR_BGR2GRAY)

        # flow = cv2.calcOpticalFlowFarneback(gft0, gft1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = cv2.calcOpticalFlowFarneback(gft0, gft1, None, 0.75, 5, 50, 3, 10, 20, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        optf = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        diff = 255 * np.abs(norm_frame(ft1) - norm_frame(ft0))
        # diff = 255 * (0.5 * (norm_frame(ft1) - norm_frame(ft0) + 1))
        diff2 = np.linalg.norm(diff, axis=2).astype(np.uint8)
        diff = diff.astype(np.uint8)

        if ff in pos_vid.Frame:
            lab = pos_vid[pos_vid.Frame == ff].Truth.values.astype(str)
        else:
            lab = "nolab"

        print(fi, ff, lab)

        f, a = plt.subplots(1, 5, figsize=(15, 45))
        a[0].imshow(optf)
        a[1].imshow(diff)
        a[2].imshow(diff2)
        a[3].imshow(ft0)
        a[4].imshow(ft1)

        plt.show()
        # time.sleep(5)

        ft0 = ft1

    cap.release()

# %%
