# %%
# Remember to convert videos before running this!

import os, sys
import cv2

import shutil
import pandas as pd
import numpy as np
from shutil import copyfile

from pathlib import Path
from PIL import Image
import ffmpeg

from joblib import Parallel, delayed

# %%
## BALANCED
# FREQ = 33 # for unique videos
# FREQ = 75  # for all
## MORE NEGATIVES (2x)
FREQ = 38  # for all
PARALLEL = True  # make video frame extraction in parallel on CPU
data_subfolder = "negatives_from_annotated"

# %%
def transform_path_to_name(fpath):
    """
        transforms the PosixPath path of a frame in the dropbox dump in a comprehensive filename.  
    """
    fpath = Path(fpath)
    fname = "_".join([a for a in str(fpath).split("/") if a != "foundframes"])
    return fname


def extract_frames_from_video(save_fold, video, FREQ):
    """
        function to extract frames with frequency `FREQ` (type: int) from the video at path `video` (type: PosixPath), and save frames as jpg at path `save_fold` (type: PosixPath). 
    """
    probe = ffmpeg.probe(video)
    n_frames = int(probe["streams"][0]["nb_frames"])

    vname = str(video).split("/")[-1][:-4]

    frame_list = np.arange(n_frames)
    frame_list = frame_list[::FREQ]

    cap = cv2.VideoCapture(str(video))
    for ff in frame_list:
        # print(ff)
        cap.set(1, ff)
        _, frame = cap.read()
        cv2.imwrite(f"{save_fold}/{vname}_neg_{ff}.jpg", frame)

    cap.release()


# %%
root_f = Path("/data/shared/hummingbird-classifier")
vid_path = Path(f"/data/shared/raw-video-import/data/RECODED_AnnotatedVideos/")

# 1) get all the videos that end in _01, as same name corresponds same location / video
videos = list(vid_path.glob("**/*_01.avi"))
videos.sort()

trs, vas, tss = int(0.6 * len(videos)), int(0.2 * len(videos)), int(0.2 * len(videos))

np.random.seed(42)
np.random.shuffle(videos)
trv = videos[:trs]
vav = videos[trs : (trs + vas)]
tsv = videos[(trs + vas) :]

# 2) for each set, loop through videos and append those that have same name but different ending
tem_ = trv.copy()
trvf = []
for vv in tem_:
    vi = list(vid_path.glob(f"**/{vv.name[:-6]}*"))
    for v in vi:
        trvf.append(v)

tem_ = vav.copy()
vavf = []
for vv in tem_:
    vi = list(vid_path.glob(f"**/{vv.name[:-6]}*"))
    for v in vi:
        vavf.append(v)

tem_ = tsv.copy()
tsvf = []
for vv in tem_:
    vi = list(vid_path.glob(f"**/{vv.name[:-6]}*"))
    for v in vi:
        tsvf.append(v)


vids_learn_set = {
    "trn": {
        "vids": trvf,
        "folder": Path(f"{root_f}/data/{data_subfolder}/training_set/class_0/"),
    },
    "val": {
        "vids": vavf,
        "folder": Path(f"{root_f}/data/{data_subfolder}/validation_set/class_0/"),
    },
    "tst": {
        "vids": tsvf,
        "folder": Path(f"{root_f}/data/{data_subfolder}/test_set/class_0/"),
    },
}
print(f"trn: unique {len(trv)}, total: {len(trvf)} videos from {vid_path}")
print(f"val: unique {len(vav)}, total: {len(vavf)} videos from {vid_path}")
print(f"tst: unique {len(tsv)}, total: {len(tsvf)} videos from {vid_path}")
print(
    f"total: unique {len(tsv) + len(vav) + len(trv)}, total: {len(tsvf) + len(vavf) + len(trvf)} videos from {vid_path}"
)

# %% NOW THE LOOPS ARE PARALLEL but have to check wether they work as supposed

# split videos in training, validation and test.
## TODO: use for testing the videos with annotations.


learning_sets = ["trn", "val", "tst"]
if PARALLEL:
    pool = Parallel(n_jobs=8, verbose=1, backend="threading")

for l_set in learning_sets:
    save_fold = vids_learn_set[l_set]["folder"]
    save_fold.mkdir(exist_ok=True, parents=True)

    videos = vids_learn_set[l_set]["vids"]

    # print(f"{l_set} :: {len(videos)}")
    if PARALLEL:
        pool(delayed(extract_frames_from_video)(save_fold, vid, FREQ) for vid in videos)
    else:
        for i, video in enumerate(videos[:]):
            extract_frames_from_video(save_fold, video, FREQ)

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
    "trn": {
        "sites": trv,
        "folder": Path(f"{root_f}/data/{data_subfolder}/training_set/class_1/"),
    },
    "val": {
        "sites": vav,
        "folder": Path(f"{root_f}/data/{data_subfolder}/validation_set/class_1/"),
    },
    "tst": {
        "sites": tsv,
        "folder": Path(f"{root_f}/data/{data_subfolder}/test_set/class_1/"),
    },
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
