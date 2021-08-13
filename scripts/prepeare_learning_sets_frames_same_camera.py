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
PARALLEL = True  # make video frame extraction in parallel on CPU
data_subfolder = "same_camera"  # annotated_videos

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
vid_path = Path(f"/data/shared/raw-video-import/data/RECODED_HummingbirdVideo/")
# vid_path = Path(f"/data/shared/raw-video-import/data/RECODED_AnnotatedVideos/")

# 1) get all the videos that end in _01 and _02 or _03
trv = list(vid_path.glob("**/*_01.avi"))
trv.sort()
vav = list(vid_path.glob("**/*_02.avi"))
vav.sort()
tsv = list(vid_path.glob("**/*_03.avi"))
tsv.sort()
# vav.extend(tsv)

# 2) cross check that filenames are present in both sets and remove those whithout correspondance

tr_n = [a.name[:-7] for a in trv]
va_n = [a.name[:-7] for a in vav]
ts_n = [a.name[:-7] for a in tsv]

tr_va_intersection = list(set(va_n).intersection(set(tr_n)))
ts_intersection = list(set(tr_va_intersection).intersection(set(ts_n)))

trv = [a for a in trv if str(a.name[:-7]) in tr_va_intersection]
vav = [a for a in vav if str(a.name[:-7]) in tr_va_intersection]
tsv = [a for a in tsv if str(a.name[:-7]) in ts_intersection]

# expand test set with video names not there already. start from the bottom for no reason
cc = 0
for vicand in vav[::-1]:
    vna = vicand.name[:-7]
    cond = [vna not in str(a) for a in tsv]
    if np.all(cond):
        tsv.append(vicand)
        vav.remove(vicand)
        cc += 1
    if cc == 19:  # What is this
        break

vids_learn_set = {
    "trn": {
        "vids": trv,
        "folder": Path(f"{root_f}/data/{data_subfolder}/training_set/class_0/"),
        "freq": 50,  # HummingbirdVideos 50: Positives for the training: 19k. This gives 20k negatives
        # AnnotatedVideos: 35
    },
    "val": {
        "vids": vav,
        "folder": Path(f"{root_f}/data/{data_subfolder}/validation_set/class_0/"),
        "freq": 80,  # HummingbirdVideos 80: Positives for the validation: 6.5k. This gives 6.5k negatives
        # AnnotatedVideos: 100
    },
    "tst": {
        "vids": tsv,
        "folder": Path(f"{root_f}/data/{data_subfolder}/test_set/class_0/"),
        "freq": 75,  # HummingbirdVideos 75: Positives for the test: 6.5k. This gives 6.3k negatives
        # AnnotatedVideos: 100
    },
}
print(f"trn: total: {len(trv)} videos from {vid_path}")
print(f"val: total: {len(vav)} videos from {vid_path}")
print(f"tst: total: {len(tsv)} videos from {vid_path}")
print(f"total: total repeated videos {len(tsv) + len(vav) + len(trv)}")

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
    print(f"{l_set}: {vids_learn_set[l_set]['freq']}")
    # print(f"{l_set} :: {len(videos)}")
    if PARALLEL:
        pool(
            delayed(extract_frames_from_video)(
                save_fold, vid, vids_learn_set[l_set]["freq"]
            )
            for vid in videos
        )
    else:
        for i, video in enumerate(videos[:]):
            extract_frames_from_video(save_fold, video, vids_learn_set[l_set]["freq"])

# %% Prepare positive frames
# Split frames according to their geo location.
# i) read Interactions_corrected.csv and filter by "file_exists == True"
# ii) group data by "waypoint", but could also be "site"
# iii) ensure no image from the same grouping variable is included in more than one learning set. The initial waypoint splitting is mixed at random (as videos in the sections above)
if 0:
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
root = Path(f"{root_f}/data/{data_subfolder}")
paths = ["training_set", "validation_set", "test_set"]

for fold in paths:
    fdir = root / fold
    for class_dir in fdir.iterdir():
        n_files = len(list(class_dir.glob("*.jpg")))
        print(f"{fold}, {class_dir.name}, {n_files}")

# %%
