# %%
import os, sys

os.environ["MKL_THREADING_LAYER"] = "GNU"

import torch

torch.hub.set_dir("/data/shared/hummingbird-classifier/models/")

import argparse

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from pathlib import Path
from PIL import Image


from matplotlib import pyplot as plt
from skimage import exposure


sys.path.append("../../")

from src.utils import (
    read_pretrained_model,
    find_checkpoints,
    Denormalize,
    cfg_to_arguments,
)
from src.ChangeDetectionUtils import main_triplet_difference

# from src.HummingbirdLoader import HummingbirdLoader
from src.HummingbirdModel import HummingbirdModel

# %%
args = {}
args["model_path"] = Path("/data/shared/hummingbird-classifier/models/convnext_v0")
args["video_name"] = Path("/data/shared/frame-diff-anomaly/data/FH107_01")
args["annotation_file"] = Path(
    "/data/shared/hummingbird-classifier/data/Weinstein2018MEE_ground_truth.csv"
)
args["output_file_dataframe"] = Path(
    "/data/shared/hummingbird-classifier/outputs/video_scores/"
)
args = cfg_to_arguments(args)
# %%
# def predict_frames_per_video(args, config):
"""
Predict scores for frame of a given video
"""

# Load trained model
dirs = find_checkpoints(args.model_path, type="best")
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
video_name = args.video_name

# This is file specific, different CSV might have different colnames etc
annot = pd.read_csv(args.annotation_file).drop_duplicates()
annot = annot[annot.Video == video_name.stem].sort_values("Frame")
annot.Truth = annot.Truth.apply(lambda x: x == "Positive")
annot.Frame -= 1  # count from 0
# %%

args.output_file_dataframe = args.output_file_dataframe / f"{video_name.stem}.csv"

# %%
# now check if all scores are in the prediction csv summary, if not (or if flag "update == True") compute
# 1 - ouput frame probabilities from trained model
# try to read the results csv, and get columns. if the file does not exist, raise a flag
# if the file exists, check if the right column is there, if not raise a flag to compute it

file_missing = False if args.output_file_dataframe.exists() else True
if not file_missing:
    video_scores = pd.read_csv(args.output_file_dataframe)
    update_score = True if "score_class" not in video_scores.columns else False
    update_diff = True if "diff_score" not in video_scores.columns else False
    update_change = True if "change_score" not in video_scores.columns else False
    update_gt = True if "ground_truth" not in video_scores.columns else False
else:
    video_scores = pd.DataFrame(
        [], columns=["score_class", "diff_score", "change_score", "ground_truth"]
    )

if args.update or file_missing or update_score:
    dataloader = model.tst_external_dataloader(path=video_name)

    pbar_cb = pl.callbacks.progress.TQDMProgressBar(refresh_rate=5)
    trainer = pl.Trainer(
        max_epochs=1,
        gpus=1,  # [0,1],
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

# %%

# return None


# if __name__ == "__main__":
#     args = argparse.ArgumentParser(description="Hummingbird inference script")
#     args.add_argument("--model", type=str, help="Path to model")
#     args.add_argument("--input_dir", type=str, help="Path to input directory")
#     args.add_argument("--output_dir", type=str, help="Path to output directory")
#     args.add_argument("--batch_size", type=int, default=1, help="Batch size")
#     args.add_argument("--num_workers", type=int, default=1, help="Number of workers")
