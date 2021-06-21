# %%

import os, sys
import cv2

import shutil
import pandas as pd
import numpy as np
from shutil import copyfile

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


def transform_path_to_name(fpath):
    fpath = Path(fpath)
    fname = "_".join([a for a in str(fpath).split("/") if a != "foundframes"])
    return fname


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

# %% Prepare positive frames
# Split frames according to their geo location.
# i) read Interactions_corrected.csv and filter by "file_exists == True"
# ii) group data by "waypoint", but could also be "site"
# iii) ensure no image from the same grouping variable is included in more than one learning set. The initial waypoint splitting is mixed at random (as videos in the sections above)

root_destination = Path("/data/users/michele/hummingbird-classifier")
root_origin = Path("/data/shared/raw-data-import/data/raw-hierarchy/")
annotation_file = Path(
    "/data/shared/raw-data-import/data/annotations/Interactions_corrected.csv"
)
annotations = pd.read_csv(annotation_file)
annotations = annotations.loc[annotations.file_exists, :]

sites, counts = np.unique(annotations.waypoint.astype(str), return_counts=True)

np.random.seed(42)
rand_ind_list = np.random.permutation(np.arange(len(sites)))
sites = sites[rand_ind_list]
counts = counts[rand_ind_list]

perc_counts = np.cumsum(counts) / np.sum(counts)

trv = sites[perc_counts < 0.6]
vav = sites[(perc_counts >= 0.6) & ((perc_counts < 0.8))]
tsv = sites[perc_counts > 0.8]

stills_learn_set = {
    "trn": {"sites": trv, "folder": Path(f"{root_f}/data/training_set/class_1/"),},
    "val": {"sites": vav, "folder": Path(f"{root_f}/data/validation_set/class_1/"),},
    "tst": {"sites": tsv, "folder": Path(f"{root_f}/data/test_set/class_1/")},
}

print(f"{len(sites)} sites from {annotation_file}")
# %%

# split still frames in the data dump into training, validation and test.
learning_sets = ["trn", "val", "tst"]

for l_set in learning_sets:
    save_fold = stills_learn_set[l_set]["folder"]
    save_fold.mkdir(exist_ok=True, parents=True)

    stills = stills_learn_set[l_set]["sites"]

    for i, site in enumerate(stills[:]):

        data_bag = annotations.fullpath_pre[annotations.waypoint == site]

        for image in data_bag:
            fname = transform_path_to_name(image)
            # copy-paste image to destination
            # print(i, site, image, fname)
            copyfile(
                root_origin / image,
                root_destination / stills_learn_set[l_set]["folder"] / fname,
            )

# %% verify sizes
root = Path("/data/users/michele/hummingbird-classifier/data")
paths = ["training_set", "validation_set", "test_set"]

for fold in paths:
    fdir = root / fold
    for class_dir in fdir.iterdir():
        n_files = len(list(class_dir.glob("*.jpg")))
        print(f"{fold}, {class_dir.name}, {n_files}")


# %%
