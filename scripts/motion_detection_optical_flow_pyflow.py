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

prefix = ""
sys.path.append(f"{prefix}src")

import pyflow

# %%
video_name = "FH303_01"
CODE = "pyflow"
# %%
df = pd.read_csv(f"{prefix}data/Weinstein2018MEE_ground_truth.csv")
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


# %%
# Flow Options:
alpha = 0.01  # 0.012
ratio = 0.7
minWidth = 50
nOuterFPIterations = 5
nInnerFPIterations = 2
nSORIterations = 10
colType = float(0)  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

video = Path(
    f"/data/shared/raw-video-import/data/RECODED_AnnotatedVideos/{video_name}.avi"
)
videos = [video]
PRINT_IMAGE = False

for vi, video in enumerate(videos[:1]):
    print(vi, video)
    probe = ffmpeg.probe(video)
    n_frames = int(probe["streams"][0]["nb_frames"])
    w, h = int(probe["streams"][0]["width"]), int(probe["streams"][0]["height"])

    out_flow_videos = Path(
        f"/data/shared/hummingbird-classifier/outputs/diff_videos/{video.name[:-4]}_optical_flow.avi"
    )

    framerate = float(eval(probe["streams"][0]["avg_frame_rate"]))
    duration_s = str(
        datetime.timedelta(seconds=float(probe["streams"][0]["duration"]))
    )[:-4]

    frame_list = np.arange(0, n_frames, 1)  # n_frames)

    # print(n_frames, length , framerate, duration_s)
    fr_rgb = np.empty((w * h, 3, len(frame_list)), dtype=np.uint8)

    cap = cv2.VideoCapture(str(video))
    of_out = cv2.VideoWriter(
        str(out_flow_videos), cv2.VideoWriter_fourcc("M", "J", "P", "G"), 5, (w, h),
    )

    _, ft0 = cap.read()
    ft0 = ft0[:, :, [2, 1, 0]].astype(float) / 255.0
    ft0 = ft0.copy(order="C")

    hsv = np.zeros_like(ft0)

    # gft0 = cv2.cvtColor(ft0, cv2.COLOR_BGR2GRAY)

    for fi, ff in enumerate(frame_list):
        print(f"\r{1+fi} / {len(frame_list)}", end="")

        cap.set(1, ff)
        ret, ft1 = cap.read()
        ft1 = ft1[:, :, [2, 1, 0]].astype(float) / 255.0
        ft1 = ft1.copy(order="C")

        u, v, im2W = pyflow.coarse2fine_flow(
            ft0,
            ft1,
            alpha,
            ratio,
            minWidth,
            nOuterFPIterations,
            nInnerFPIterations,
            nSORIterations,
            colType,
        )

        flow = np.concatenate((u[..., None], v[..., None]), axis=2)

        hsv = np.zeros(ft1.shape, dtype=np.uint8)
        hsv[:, :, 0] = 255
        hsv[:, :, 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        optf = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if PRINT_IMAGE:

            diff = 255 * np.abs(norm_frame(ft1) - norm_frame(ft0))
            # diff = 255 * (0.5 * (norm_frame(ft1) - norm_frame(ft0) + 1))
            # diff2 = np.linalg.norm(diff, axis=2).astype(np.uint8)
            diff = diff.astype(np.uint8)

            # if ff in pos_vid.Frame:
            #     lab = pos_vid[pos_vid.Frame == ff].Truth.values.astype(str)
            # else:
            #     lab = "nolab"

            # print(fi, ff, lab)
            f, a = plt.subplots(1, 4, figsize=(20, 30))
            a[0].imshow(optf)
            a[0].axis("off")
            a[1].imshow(diff)
            a[1].axis("off")
            # a[2].imshow(diff2)
            a[2].imshow(ft0)
            a[2].axis("off")
            a[3].imshow(ft1)
            a[3].axis("off")
            plt.show()

        conv_o = optf[:, :, [2, 1, 0]].astype(np.uint8)
        of_out.write(conv_o)

        # time.sleep(5)

        ft0 = ft1

    cap.release()
    of_out.release()

# %%
