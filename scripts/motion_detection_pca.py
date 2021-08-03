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

from torch import nn
from sklearn.decomposition import PCA

prefix = "../"
sys.path.append(f"{prefix}src")

# %%
def generate_frame_blocks(n_frames, block_size, frequency=1):
    n_blocks = n_frames // block_size
    remainder = n_frames % block_size
    blocks = []
    for bb in range(n_blocks):
        blocks.append(
            np.arange(bb * block_size, bb * block_size + block_size, frequency)
        )
    bb += 1
    blocks.append(np.arange(bb * block_size, bb * block_size + remainder, frequency))
    return blocks


def norm_frame(frin):
    frame = frin - np.mean(frin, axis=(0, 1))
    # frame = frame / np.std(frin, axis=(0, 1))

    frame = (frame - np.min(frame, axis=(0, 1))) / (
        np.max(frame, axis=(0, 1)) - np.min(frame, axis=(0, 1))
    )
    return frame


# test it
# frame_blocks = generate_frame_blocks(105, 10, 1)
# print(frame_blocks)
# print(len(frame_blocks))

# %%
video_name = "FH112_01"
# %%

df = pd.read_csv("../data/Weinstein2018MEE_ground_truth.csv")
pos_vid = df[df.Video == video_name].copy()
pos_vid.Truth = pos_vid.Truth.replace({"Positive": 1, "Negative": 0})

# %%

video = Path(
    f"/data/shared/raw-video-import/data/RECODED_HummingbirdVideo/{video_name}.avi"
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
    # print(n_frames, length, framerate, duration_s)

    blocks = generate_frame_blocks(n_frames, block_size=300, frequency=1)

    cap = cv2.VideoCapture(str(video))
    # out = cv2.VideoWriter(
    #     f"{prefix}outputs/diff_videos/diff_{video.name}",
    #     cv2.VideoWriter_fourcc("M", "J", "P", "G"),
    #     10,
    #     (w, h),
    # )

    norm_eigs = np.zeros((n_frames,))
    pcam = PCA(n_components=50)

    for bi, frame_block in enumerate(blocks[:]):

        # fr_px = np.empty((w * h, 3, len(frame_block) - 1), dtype=np.uint8)
        fr_d = np.empty((w * h, len(frame_block) - 1), dtype=np.uint8)

        for fi, ff in enumerate(frame_block):
            print(
                f"\r{video.name}, frame {fi+1}/{len(frame_block)} block {bi+1}/{len(blocks)}, video {vi+1} of {len(videos)}",
                end="",
            )

            if fi > 0:
                cap.set(1, frame_block[fi - 1])
                _, ft0 = cap.read()
                ft0 = norm_frame(ft0)

                cap.set(1, ff)
                _, ft1 = cap.read()
                ft1 = norm_frame(ft1)

                # ft0 = cv2.blur(ft0, (10, 10))
                # ft1 = cv2.blur(ft1, (10, 10))

                diff = ft1 - ft0
                # diff = np.uint8(255 * (diff + 1) / 2)

                # fr_px[:, :, i] = ft0.reshape(-1,3)
                fr_d[:, fi - 1] = np.linalg.norm(diff, axis=2).ravel()

        pcam.fit(fr_d)
        norm_eigs[frame_block[1:]] = np.linalg.norm(pcam.components_, axis=0)

    cap.release()
    # out.release()

    # cap.set(1, ff)
    # _, frame = cap.read()
    # frame = frame[:, :, [2, 1, 0]]
    # pframe = Image.fromarray(frame.astype("uint8"), "RGB")
    # frame = augment(pframe).to(device)
# %%
plt.figure(figsize=(13, 5))
plt.plot(norm_eigs[:])

# %%
# Visualize detected frames vor the video

probe = ffmpeg.probe(video)
n_frames = int(probe["streams"][0]["nb_frames"])
cap = cv2.VideoCapture(str(video))

# fr_px = np.empty((w * h, 3, len(frame_block) - 1), dtype=np.uint8)
# fr_d = np.empty((w * h, len(frame_block) - 1), dtype=np.uint8)
frame_list = np.where(norm_eigs > 0.99)[0][30:50]
print(len(frame_list))

for fi, ff in enumerate(frame_list):
    cap.set(1, ff)
    _, frame = cap.read()
    # frame = norm_frame(frame)

    plt.figure()
    plt.imshow(frame)
    plt.show()

    # if fi == 5:
    #     break

# %%
# plt.figure(figsize=(13, 5))
# plt.imshow(fr_px[:, 0, :], aspect="auto")
# plt.figure(figsize=(13, 5))
# plt.imshow(fr_d, aspect="auto")

# pcm = PCA(n_components=50)
# pc = pcm.fit_transform(fr_d)

# %%

plt.figure(figsize=(13, 5))
plt.imshow(pcm.components_, aspect="auto")

plt.figure(figsize=(13, 5))
plt.plot(np.linalg.norm(pcm.components_, axis=0))


# %%

comm = np.linalg.norm(pcm.components_, axis=0)
fr = np.where(comm > 0.7)[0]
print(len(fr))
plt.figure()
for i in fr:
    plt.imshow(fr_px[:, :, i].reshape(h, w, 3))
    plt.show()
# time.sleep(1)
