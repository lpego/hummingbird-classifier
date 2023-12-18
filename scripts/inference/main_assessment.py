# %%
import os, sys

import argparse
import yaml
import json

import numpy as np
import pandas as pd

from pathlib import Path

from matplotlib import pyplot as plt

sys.path.append("../../")

from src.utils import (
    cfg_to_arguments,
)

# %%


# %%
def precision_at_k(predictions, K):
    """
    Computes precision at K for a given list of predictions

    Parameters
    ----------
    predictions : list
        List of predictions, 1 if positive, 0 if negative
    K : int
        Number of top predictions to consider

    Returns
    -------
    float
        Precision at K

    """
    return sum(predictions[:K]) / K


def recall_at_k(predictions, K):
    """
    Computes recall at K for a given list of predictions

    Parameters
    ----------
    predictions : list
        List of predictions, 1 if positive, 0 if negative
    K : int
        Number of top predictions to consider

    Returns
    -------
    float
        Recall at K

    """
    return sum(predictions[:K]) / sum(predictions)


def aggregate_assessment():
    """

    Parameters
    ----------

    Returns
    -------

    """

    return None


# %%
config_file = Path("../../configs/configuration_hummingbirds.yaml")
with open(config_file, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

config = cfg_to_arguments(config)
# %%

args = {}
args["top_k"] = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]

args = cfg_to_arguments(args)

# %% def per_video_assessment(video_result, args, config):
"""
This function runs the assessment of the hummingbird detection pipeline for a given video.
- Sorts scores by descending order, from most likely to contain a hummingbird to least likely
- Selects a subset of top K frames, independently of the score, set as an int or percentage [0, 1]. This is a list of values
- Computes retrieval metrics:
    - Precision at K
    - Recall at K
    - F1 at K
    - Mean Average Precision

- dumps a report in each video folder:
    - a `metrics.csv` file with the metrics for each K

Parameters
----------
video_result : str
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
# %%
# Read score file
video_result = "/data/shared/hummingbird-classifier/outputs/video_scores/densenet161-v0/FH102_02.csv"
score_file = Path(video_result)
scores = pd.read_csv(score_file)
# %%
# Calculate score as linear combination of the three scores:
# -classification, classificaiton dynamic, class-agnostic change detection
# score_class, diff_score, change_score
scores["aggregated_score"] = (
    config.infe_weight_classification_score * scores["score_class"]
    + config.infe_weight_classification_dynamic * scores["diff_score"]
    + config.infe_weight_triplet_change_detection * scores["change_score"]
)

# # Sort scores by descending order
# scores = scores.sort_values(by="aggregated_score", ascending=False)

# Select top K frames from list of top K threhsolds
if args.top_k[-1] <= 1:
    args.top_k_frames = [int(a * len(scores)) for a in args.top_k]
else:
    args.top_k_frames = args.top_k

# %%
# Compute metrics for 3 approaches: 2 baselines (pc and change detection) and the aggregated score

scores_to_test = ["score_class", "change_score", "aggregated_score"]
metrics = {}
metrics["top_k_frames"] = args.top_k_frames
metrics["top_k"] = args.top_k
metrics["video_name"] = score_file.stem
metrics["source_folder"] = score_file.parent.stem
for score_name in scores_to_test:
    pr_na = score_name.replace("_", " ").capitalize()
    score = scores[[score_name, "ground_truth"]].sort_values(
        by=score_name, ascending=False
    )

    metrics[score_name] = {}
    metrics[score_name]["precision"] = []
    metrics[score_name]["recall"] = []
    metrics[score_name]["f1"] = []
    for K in args.top_k_frames:
        # Assume first K retrieved frames are positives
        metrics[score_name]["precision"].append(
            precision_at_k(score.loc[score["ground_truth"] >= 0, score_name], K)
        )
        metrics[score_name]["recall"].append(
            recall_at_k(score.loc[scores["ground_truth"] >= 0, score_name], K)
        )
        metrics[score_name]["f1"].append(
            2
            * metrics[score_name]["precision"][-1]
            * metrics[score_name]["recall"][-1]
            / (metrics[score_name]["precision"][-1] + metrics[score_name]["recall"][-1])
        )

# write json file
metrics_file = Path(video_result).parent / f"{score_file.stem}_metrics.json"
with open(metrics_file, "w") as f:
    json.dump(metrics, f, indent=4)

# %%

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Hummingbird inference script")
    args.add_argument(
        "--results_path",
        type=Path,
        help="Path to the model checkpoint to use for inference",
    )
    args = args.parse_args()

    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = cfg_to_arguments(config)

    # compute per video assessment
    # video_results = Path(args.results_path) / "video_scores"

    # aggregate_assessment
