# %%

import os, sys
import cv2

import shutil
import pandas as pd
import numpy as np

from pathlib import Path
from PIL import Image
import ffmpeg

# %% define function to count frames
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
root_f = Path("/data/users/michele/hummingbird-classifier")
vid_path = Path(f"/data/shared/raw-video-import/data/RECODED_HummingbirdVideo/")
videos = list(vid_path.glob("**/*.avi"))

trs, vas, tss = int(0.6 * len(videos)), int(0.2 * len(videos)), int(0.2 * len(videos))

np.random.seed(42)
np.random.shuffle(videos)
trv = videos[:trs]
vav = videos[trs : (trs + vas)]
tsv = videos[(trs + vas) :]

vids_learn_set = {
    "trn": {"vids": trv, "folder": Path(f"{root_f}/data/training_set/class_0/"),},
    "val": {"vids": vav, "folder": Path(f"{root_f}/data/validation_set/class_0/"),},
    "tst": {"vids": tsv, "folder": Path(f"{root_f}/data/test_set/class_0/")},
}

print(f"{len(videos)} videos from {vid_path}")
# %%

# split videos in training, validation and test.
## TODO: use for testing the videos with annotations.
FREQ = 75
learning_sets = ["trn", "val", "tst"]

for l_set in learning_sets:
    save_fold = vids_learn_set[l_set]["folder"]
    save_fold.mkdir(exist_ok=True, parents=True)

    videos = vids_learn_set[l_set]["vids"]

    for i, video in enumerate(videos[:]):

        probe = ffmpeg.probe(video)
        n_frames = int(probe["streams"][0]["nb_frames"])

        vname = str(video).split("/")[-1][:-4]

        frame_list = np.arange(n_frames)
        frame_list = frame_list[::FREQ]

        print(
            f"{vname} -> {l_set} :: ({i+1}/{len(videos)}):: number of frames = {n_frames}, frame_list {len(frame_list)}"
        )

        cap = cv2.VideoCapture(str(video))
        for ff in frame_list:
            # print(ff)
            cap.set(1, ff)
            _, frame = cap.read()
            cv2.imwrite(f"{save_fold}/{vname}_neg_{ff}.jpg", frame)

        cap.release()

