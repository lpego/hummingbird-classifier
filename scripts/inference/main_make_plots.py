# %%
import os, sys

import argparse
import yaml

import numpy as np
import pandas as pd

from pathlib import Path

from matplotlib import pyplot as plt

sys.path.append("../../")

from src.utils import (
cfg_to_arguments,
)

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
video_result = "/data/shared/hummingbird-classifier/outputs/video_scores/densenet161-v0/FH107_01.csv"
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

f, a = plt.subplots(4,1,figsize=(15,9))
a[0].scatter(np.where(scores.ground_truth == 0)[0], 1*(np.ones(np.sum(scores.ground_truth == 0))), marker="v", color="k")
a[0].scatter(np.where(scores.ground_truth == 1)[0], 0.9*(np.ones(np.sum(scores.ground_truth == 1))), marker="v", color="r")
a[0].plot(scores["aggregated_score"], label="Full anomaly score")
a[0].grid(True)
a[0].set_ylabel("Full anomaly score")
a[1].plot(scores["score_class"], label="Hummingbird probability")
a[1].grid(True)
a[1].set_ylabel("Hummingbird probability")
a[2].plot(scores["change_score"], label="Change detection")
a[2].grid(True)
a[2].set_ylabel("Change detection")
a[3].plot(scores["diff_score"], label="P differential")
a[3].grid(True)
a[3].set_ylabel("P differential")
a[3].set_xlabel("Frame number");

# %% 
# Compute metrics for 3 approaches: 2 baselines (pc and change detection) and the aggregated score
    
scores_to_test = ["score_class", "change_score", "aggregated_score"] 
markers = ["o", "x", "s"]
cols = ["r", "b", "g"]
ltype = ["-", "--", "-."]

i = 0

metrics = {}
plt.figure()

for score_name in scores_to_test:
    pr_na = score_name.replace("_", " ").capitalize()
    score = scores[[score_name, "ground_truth"]].sort_values(by=score_name, ascending=False)

    metrics[score_name] = {}
    metrics[score_name]["precision"] = []
    metrics[score_name]["recall"] = []
    metrics[score_name]["f1"] = []
    for K in args.top_k_frames:
        # Assume first K retrieved frames are positives
        metrics[score_name]["precision"].append(precision_at_k(score.loc[score["ground_truth"] >= 0, score_name], K))
        metrics[score_name]["recall"].append(recall_at_k(score.loc[scores["ground_truth"] >= 0, score_name], K))
        metrics[score_name]["f1"].append(
            2
            * metrics[score_name]["precision"][-1]
            * metrics[score_name]["recall"][-1]
            / (metrics[score_name]["precision"][-1] + metrics[score_name]["recall"][-1])
        )

    plt.plot(metrics[score_name]["precision"], label = f"{pr_na} p", marker=markers[i], c=cols[i], ls=ltype[0])
    plt.plot(metrics[score_name]["recall"], label = f"{pr_na} r", marker=markers[i], c=cols[i], ls=ltype[1])
    plt.plot(metrics[score_name]["f1"], label = f"{pr_na} f1", marker=markers[i], c=cols[i], ls=ltype[2])
    plt.legend(handlelength=3)
    plt.grid()
    plt.xticks(range(len(args.top_k)), zip(args.top_k, args.top_k_frames), rotation=45)
    plt.xlabel("top K")
    plt.ylabel("metric")
    plt.title(f"Metrics for video {score_file.stem}")
    i += 1

# %%
from PIL import Image
# plot some detections 
score_to_sort = "aggregated_score"
score_sorted = scores.sort_values(by=score_to_sort, ascending=False)
list_frames = score_sorted.index[:10].tolist()
print(list_frames)

frame_path = Path("../../../frame-diff-anomaly/data/annotated_videos/FH102_02") 
plt.figure(figsize=(15,5))
for i, frame in enumerate(list_frames): 
    # if number of frame digits are less than 5, add zeros in front
    frame = str(frame).zfill(5)
    plt.subplot(2,5,i+1)
    plt.imshow(Image.open(frame_path / f"frame_{frame}.jpg"))
    plt.axis("off")
    plt.title(f"frame {frame}")
#%% 
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
