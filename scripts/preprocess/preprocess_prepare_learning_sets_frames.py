# %%
# Remember to convert videos before running this!

# import os, sys
import argparse
import cv2
import pandas as pd
import numpy as np
import ffmpeg
import yaml

from shutil import copyfile
from pathlib import Path
from joblib import Parallel, delayed

import sys

sys.path.append(".")
from src.utils import cfg_to_arguments


# %%
def transform_path_to_name(fpath):
    """
    transforms the PosixPath path of a frame in the dropbox dump in a comprehensive filename.
    """
    fpath = Path(fpath)
    fname = "_".join([a for a in str(fpath).split("/") if a != "foundframes"])
    return fname


def extract_frames_from_video(save_fold, video, ntot):
    """
    function to extract frames with frequency `FREQ` (type: int) from the video at path `video` (type: PosixPath), and save frames as jpg at path `save_fold` (type: PosixPath).
    """
    probe = ffmpeg.probe(video)
    # print(probe["format"]["filename"])
    n_frames = int(probe["streams"][0]["nb_frames"])
    FREQ = int(np.ceil(n_frames / ntot))

    vname = str(video).split("/")[-1][:-4]

    frame_list = np.arange(n_frames)
    frame_list = frame_list[::FREQ]
    # print(f"{vname} has {n_frames} --> {len(frame_list)}, n: {ntot}, freq: {FREQ}")
    # return len(frame_list)

    cap = cv2.VideoCapture(str(video))
    for ff in frame_list:
        # print(ff)
        cap.set(1, ff)
        _, frame = cap.read()
        cv2.imwrite(f"{save_fold}/{vname}_neg_{ff}.jpg", frame)

    cap.release()


def prepare_sets(vid_parsing_pars, videos, config):
    """
    function to extract frames with frequency `FREQ` (type: int) from the video at path `video` (type: PosixPath), and save frames as jpg at path `save_fold` (type: PosixPath).
    """
    # %% Prepare positive frames
    # Split frames according to their geo location.
    # i) read Interactions_corrected.csv and filter by "file_exists == True"
    # ii) group data by "waypoint", but could also be "site"
    # iii) ensure no image from the same grouping variable is included in more than one learning set. The initial waypoint splitting is mixed at random (as videos in the sections above)

    annotations = pd.read_csv(config.annotations_file)
    annotations = annotations.loc[annotations.file_exists, :]

    sites, counts = np.unique(annotations.waypoint.astype(str), return_counts=True)

    np.random.seed(config.rng_seed)
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
            "folder": Path(
                f"{vid_parsing_pars['positive_data_subfolder']}/trn_set/class_1/"
            ),
        },
        "val": {
            "sites": vav,
            "folder": Path(
                f"{vid_parsing_pars['positive_data_subfolder']}/val_set/class_1/"
            ),
        },
        "tst": {
            "sites": tsv,
            "folder": Path(
                f"{vid_parsing_pars['positive_data_subfolder']}/tst_set/class_1/"
            ),
        },
    }

    print(f"{len(sites)} sites from {config.annotations_file}")

    # split still frames in the data dump into training, validation and test.

    COPY_POSITIVES = True
    if (
        vid_parsing_pars["positive_data_subfolder"]
        == vid_parsing_pars["negative_data_subfolder"]
    ):
        learning_sets = ["trn", "val", "tst"]
        for l_set in learning_sets:
            save_fold = stills_learn_set[l_set]["folder"]
            save_fold.mkdir(exist_ok=True, parents=True)

            print(f"Copying {l_set} set to {save_fold}")
            stills = stills_learn_set[l_set]["sites"]

            for i, site in enumerate(stills[:]):
                data_bag = annotations.fullpath_pre[annotations.waypoint == site]
                for image in data_bag:
                    fname = transform_path_to_name(image)
                    copyfile(
                        Path(config.still_frames_location) / image,
                        stills_learn_set[l_set]["folder"] / fname,
                    )
    # %%
    # Get positive class size and compute negative fractions

    root = vid_parsing_pars["positive_data_subfolder"]
    paths = ["trn_set", "val_set", "tst_set"]

    n_frames = {}
    for fold in paths:
        fdir = root / fold
        for class_dir in fdir.iterdir():
            n_files = len(list(class_dir.glob("*.jpg")))
            n_frames[fold.split("_")[0]] = {}
            n_frames[fold.split("_")[0]]["n_positives"] = n_files
            # print(f"{root}, {fold}, {class_dir.name}, {n_files}")

    # Prepare Negatives as random frames from videos
    trs, vas, tss = (
        int(0.6 * len(videos)),
        int(0.2 * len(videos)),
        int(0.2 * len(videos)),
    )

    np.random.seed(config.rng_seed)
    np.random.shuffle(videos)
    trv = videos[:trs]
    vav = videos[trs : (trs + vas)]
    tsv = videos[(trs + vas) :]

    # 2) for each set, loop through videos and append those that have same name but different ending
    tem_ = trv.copy()
    trvf = []
    for vv in tem_:
        vi = list(vv.parents[0].glob(f"{vv.name[:-6]}*"))
        for v in vi:
            trvf.append(v)

    tem_ = vav.copy()
    vavf = []
    for vv in tem_:
        vi = list(vv.parents[0].glob(f"{vv.name[:-6]}*"))
        for v in vi:
            vavf.append(v)

    tem_ = tsv.copy()
    tsvf = []
    for vv in tem_:
        vi = list(vv.parents[0].glob(f"{vv.name[:-6]}*"))
        for v in vi:
            tsvf.append(v)

    vids_learn_set = {
        "trn": {
            "vids": trvf,
            "folder": Path(
                f"{vid_parsing_pars['negative_data_subfolder']}/trn_set/class_0/"
            ),
            # "freq": vid_pars["FREQ"]["trn"],
        },
        "val": {
            "vids": vavf,
            "folder": Path(
                f"{vid_parsing_pars['negative_data_subfolder']}/val_set/class_0/"
            ),
            # "freq": vid_pars["FREQ"]["val"],
        },
        "tst": {
            "vids": tsvf,
            "folder": Path(
                f"{vid_parsing_pars['negative_data_subfolder']}/tst_set/class_0/"
            ),
            # "freq": vid_pars["FREQ"]["tst"],
        },
    }
    print(
        f"class_0 trn: unique {len(trv)}, total: {len(trvf)} videos from {vids_learn_set['trn']['vids'][0].parents[0]}"
    )
    print(
        f"class_0 val: unique {len(vav)}, total: {len(vavf)} videos from {vids_learn_set['val']['vids'][0].parents[0]}"
    )
    print(
        f"class_0 tst: unique {len(tsv)}, total: {len(tsvf)} videos from {vids_learn_set['tst']['vids'][0].parents[0]}"
    )
    print(
        f"total: unique {len(tsv) + len(vav) + len(trv)}, total: {len(tsvf) + len(vavf) + len(trvf)} videos from"
    )

    # compute per video frame sampling rate to be balanced wrt positive class
    bias = config.sampling_rate_negatives
    for i, key in enumerate(vids_learn_set.keys()):
        vids_learn_set[key]["n_per_neg_vid"] = bias[i] + 2 * np.ceil(
            n_frames[key]["n_positives"] / len(vids_learn_set[key]["vids"])
        )

    # THE LOOPS ARE PARALLEL but have to check wether they work as supposed
    # split videos in training, validation and test.

    learning_sets = ["trn", "val", "tst"]
    if vid_parsing_pars["parallell_process"]:
        pool = Parallel(
            n_jobs=config.cores_parallel_jobs, verbose=1, backend="threading"
        )

    for l_set in learning_sets:
        save_fold = vids_learn_set[l_set]["folder"]
        save_fold.mkdir(exist_ok=True, parents=True)

        videos = vids_learn_set[l_set]["vids"]

        # print(f"{l_set} :: {len(videos)}")
        if vid_parsing_pars["parallell_process"]:
            pool(
                delayed(extract_frames_from_video)(
                    save_fold, vid, vids_learn_set[l_set]["n_per_neg_vid"]
                )
                for vid in videos
            )
        else:
            for i, video in enumerate(videos[:]):
                extract_frames_from_video(
                    save_fold, video, vids_learn_set[l_set]["n_per_neg_vid"]
                )

    root = Path("/data/shared/hummingbird-classifier/data")
    paths = ["trn_set", "val_set", "tst_set"]

    for subd in ["positive_data_subfolder", "negative_data_subfolder"]:
        for fold in paths:
            fdir = root / (vid_parsing_pars[subd] + "/" + fold)
            for class_dir in fdir.iterdir():
                n_files = len(list(class_dir.glob("*.jpg")))
                print(
                    f"{root}, {vid_parsing_pars[subd] + '/' + fold}, {class_dir.name}, {n_files}"
                )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--learning-set-folder",
        type=str,
        help="Path to learning sets (`data` subfolder)",
    )
    args.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/configuration_hummingbirds.yml",
        help="Path to config file",
    )
    args = args.parse_args()

    # parse config file
    with open(args.config, "r") as f:
        args.config = yaml.safe_load(f)
    config = cfg_to_arguments(args.config)

    config.still_frames_location = Path(config.still_frames_location)

    ## BALANCED
    # FREQ = 33 # for unique videos
    # FREQ = 75  # for all
    ## MORE NEGATIVES (2x)

    # change to absolute folders, e.g.:
    # positive and negative folders can be different for positives and negatives, but does not look like a good idea
    # Will need to be changed accordingly in the HummingbirdModel class
    vid_parsing_pars = {
        "parallell_process": True,  # make video frame extraction in parallel on CPU
        "positive_data_subfolder": Path(
            f"{config.root_folder}/{args.learning_set_folder}"
        ),
        "negative_data_subfolder": Path(
            f"{config.root_folder}/{args.learning_set_folder}"
        ),
    }
    current_vid_root = Path(config.current_video_root)
    video_list = sorted(list(current_vid_root.glob("RECODED_HummingbirdVideo*/*.avi")))

    prepare_sets(vid_parsing_pars, video_list, config)
