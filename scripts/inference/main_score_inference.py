# %%
import os, sys

os.environ["MKL_THREADING_LAYER"] = "GNU"

import torch

torch.hub.set_dir("/data/shared/hummingbird-classifier/models/")

import argparse
import yaml

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from pathlib import Path
from skimage import exposure

sys.path.append(".")

from src.utils import (
    find_checkpoints,
    cfg_to_arguments,
)
from src.ChangeDetectionUtils import main_triplet_difference
from src.HummingbirdModel import HummingbirdModel


def per_video_frame_inference(video_folder, args, config):
    """
    Predict scores for frame of a given video

    Parameters
    ----------
    video_folder : str
        Path to the video folder, where each frame is stored independently as a jpg file
    args : argparse.Namespace
        Arguments passed to the script
    config : dict
        Configuration dictionary

    Returns
    -------
    None
        Saves the scores in a csv file at a designated location
    """

    # Load trained model
    dirs = find_checkpoints(args.model_path, type=config.infe_load_model)
    mod_path = dirs[
        -1
    ]  # pick last if several best (in config can be set how many topK models to keep)

    # THIS WORKS: sfrfhnc3, 24zruk7z DENSENET161: 38tn45xv, 2col29g3, tba 130ch647
    # || GOOD ixpfqgvo very very long 3pau0qtg / very long 32tka2n9 / long 22m0pigr / mid bqoy698f / short 23rgsozp
    # THIS WORKS LESS WELL: 1zh8fqdf
    # dirs = find_checkpoints(Path(f"{prefix}hummingbirds-pil"), version="24zruk7z", log="last")#.glob("**/*.ckpt"))

    model = HummingbirdModel()
    model = model.load_from_checkpoint(
        checkpoint_path=mod_path,
        # hparams_file= str(mod_path.parents[1] / 'hparams.yaml') #same params as args
    )

    # Not used at inference, blank out
    model.pos_data_dir = Path("")
    model.neg_data_dir = Path("")
    model.eval()

    # Load video into dataloader
    video_name = video_folder.stem

    #
    # This is file specific, different CSV might have different colnames etc
    annot = pd.read_csv(args.annotation_file).drop_duplicates()
    annot = annot[annot.Video == video_name].sort_values("Frame")
    annot = annot.replace({"Positive": 1.0, "Negative": 0.0})
    # annot.Truth = annot.Truth.apply(lambda x: x == "Positive")
    annot.Frame -= 1  # count from 0
    annot = annot.drop_duplicates()
    annot = annot.set_index("Frame")
    # annot = annot.drop_duplicates()

    args.output_file_dataframe = args.output_file_folder / f"{video_name}.csv"

    # now check if all scores are in the prediction csv summary, if not (or if flag "update == True") compute
    # 1 - ouput frame probabilities from trained model
    # try to read the results csv, and get columns. if the file does not exist, raise a flag
    # if the file exists, check if the right column is there, if not raise a flag to compute it

    file_missing = False if args.output_file_dataframe.exists() else True
    if not file_missing:
        video_scores = pd.read_csv(args.output_file_dataframe)
        update_score = True if "score_class" not in video_scores.columns else False
        update_diff = True if video_scores["diff_score"].isna().any() else False
        update_change = (
            True
            if video_scores["change_score"].isna().any() not in video_scores.columns
            else False
        )
        update_gt = (
            True
            if video_scores["ground_truth"].isna().any() not in video_scores.columns
            else False
        )
    else:
        video_scores = pd.DataFrame(
            [], columns=["score_class", "diff_score", "change_score", "ground_truth"]
        )

    if args.update or file_missing or update_score:
        dataloader = model.tst_external_dataloader(path=video_folder, batch_size=64)

        pbar_cb = pl.callbacks.progress.TQDMProgressBar(refresh_rate=5)
        trainer = pl.Trainer(
            max_epochs=1,
            devices=1 if torch.cuda.is_available() else 1,
            callbacks=[pbar_cb],
            enable_checkpointing=False,
            logger=False,
        )

        outputs = trainer.predict(
            model=model, dataloaders=[dataloader], return_predictions=True
        )

        probabilities = []
        for o in outputs:
            probabilities.append(
                np.asarray(o[1])
            )  # o[0] is the label, o[2] is the ground truth
        probabilities = np.concatenate(probabilities, axis=0)
        video_scores["score_class"] = probabilities[:, 1]
        video_scores.to_csv(args.output_file_dataframe, index=False)

    if args.update or file_missing or update_diff:
        diff_s = video_scores["score_class"].diff()
        diff_s.iloc[0] = 0
        video_scores["diff_score"] = np.abs(diff_s)
        video_scores.to_csv(args.output_file_dataframe, index=False)

    if args.update or file_missing or update_change:
        change_det_file = (
            args.output_file_dataframe.parents[1]
            / f"{video_name}_change_diff_scores.csv"
        )
        if change_det_file.exists():
            change_d = pd.read_csv(change_det_file)
        else:
            change_d = main_triplet_difference(
                video_folder,
                save_csv=change_det_file,
            )
        video_scores["change_score"] = change_d["mag_std"]
        video_scores.to_csv(args.output_file_dataframe, index=False)

    if args.update or file_missing or update_gt:
        video_scores["ground_truth"] = -1 * np.ones((len(video_scores.ground_truth),))
        video_scores.loc[annot.index, "ground_truth"] = annot.Truth
        video_scores.to_csv(args.output_file_dataframe, index=False)

    return None


# %%

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Hummingbird inference script")
    args.add_argument(
        "--model_path",
        type=Path,
        help="Path to the model checkpoint to use for inference",
    )
    args.add_argument(
        "--videos_root_folder",
        type=Path,
        help="Path to the video(s) subfolders, where each will be parsed. Each subfolder at this level should contain the frames of a given video, as jpg files",
    )
    args.add_argument(
        "--annotation_file",
        type=Path,
        help="Path to the video frames annotation file",
    )
    args.add_argument(
        "--output_file_folder",
        type=Path,
        help="Path to the folder where results / score CSV are stored",
    )
    args.add_argument(
        "--update",
        "-u",
        action="store_true",
        help="Flag to force recomputing (all) the scores",
    )
    args.add_argument(
        "--config_file",
        "-c",
        type=Path,
        help="Path to the config file",
    )
    args = args.parse_args()

with open(args.config_file, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

config = cfg_to_arguments(config)

# create output folder if it does not exist
if not args.output_file_folder.exists():
    args.output_file_folder.mkdir(parents=True)

video_list = sorted(list(args.videos_root_folder.glob("*")))
print(f"Found {len(video_list)} videos, running inference on those.")
# for video in video_list:
#     print(video)

for video in video_list[:]:
    print(f"Running inference on {video}")
    per_video_frame_inference(video, args, config)


# %%
# args = {}
# args["model_path"] = Path("/data/shared/hummingbird-classifier/models/convnext_v0")
# args["video_name"] = Path("/data/shared/frame-diff-anomaly/data/FH102_02")
# args["annotation_file"] = Path(
#     "/data/shared/hummingbird-classifier/data/Weinstein2018MEE_ground_truth.csv"
# )
# args["output_file_dataframe"] = Path(
#     "/data/shared/hummingbird-classifier/outputs/video_scores/"
# )
# args["update"] = False
# args = cfg_to_arguments(args)
