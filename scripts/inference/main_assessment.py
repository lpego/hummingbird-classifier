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
def per_video_assessment(score_file, top_k, config):
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
        - a `<video_name>_metrics.json` file with the metrics for each K

    Parameters
    ----------
    score_file : Path
        Path to the file containing the detection scores, contained  in video_scores/<model> subfolder, where each CSV
    config : dict
        Configuration dictionary

    Returns
    -------
    None
        Saves the scores in a json file at a designated location

    """

    scores = pd.read_csv(score_file)
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
    if top_k[-1] <= 1:
        top_k_frames = [int(a * len(scores)) for a in top_k]
    else:
        top_k_frames = top_k

    # Compute metrics for 3 approaches: 2 baselines (pc and change detection) and the aggregated score
    scores_to_test = ["score_class", "change_score", "aggregated_score"]
    metrics = {}
    metrics["top_k_frames"] = top_k_frames
    metrics["top_k"] = top_k
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
        for K in top_k_frames:
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
                / (
                    metrics[score_name]["precision"][-1]
                    + metrics[score_name]["recall"][-1]
                )
            )

    # write json file
    metrics_file = Path(score_file).parent / f"{score_file.stem}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)


# %%

# if __name__ == "__main__":
# args = argparse.ArgumentParser(description="Hummingbird inference script")
# args.add_argument(
#     "--results_path",
#     type=Path,
#     help="Path to the video_scores sub-folder, specific to a model. This folder contains CSV files with the raw pipeline scores for each video",
# )
# args.add_argument(
#     "--config_file",
#     type=Path,
#     help="Path to the configuration file",
# args.add_argument(
#     "--update",
#     action="store_true",
#     help="If set, will recompute the metrics for all videos in the folder",
# )
# args.add_argument(
#     "--aggregate",
#     action="store_true",
#     help="If set, will aggregate the metrics for all videos in the folder",
# )
# args = args.parse_args()
args = {}
args["results_path"] = Path(
    "/data/shared/hummingbird-classifier/outputs/video_scores/densenet161-v1"
)
args["config_file"] = Path(
    "/data/shared/hummingbird-classifier/configs/configuration_hummingbirds.yaml"
)
args["update"] = True
args["aggregate"] = True
args = cfg_to_arguments(args)

with open(args.config_file, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

config = cfg_to_arguments(config)


# compute per video assessment
video_scores_list = sorted(list(Path(args.results_path).glob("*.csv")))
if args.update:
    for video_score in video_scores_list[:]:
        try:
            print(f"Processing {video_score}")
            per_video_assessment(video_score, top_k=config.infe_top_k, config=config)
        except:
            print(f"Error processing {video_score}")
            continue
else:
    print(f"Skipping video assessment, update set to {args.update}")

# aggregate results at same top_k rate
if args.update and args.aggregate:
    video_metrics_list = sorted(list(Path(args.results_path).glob("*.json")))
    # remove _aggregated_metrics.json if present
    video_metrics_list = [
        x for x in video_metrics_list if "_aggregated_metrics.json" not in str(x)
    ]

    print(video_metrics_list)
# if args.update:
# aggregate_assessments(video_metrics, config=config)

# aggregate_assessment
# Read score file

# def aggregate_assessments(video_metrics, config):
#     """
#     This function aggregates the results of the per-video assessment into a single json file
#     - Reads the metrics json files for each video
#     - Aggregates the results for each K
#     - Saves the results in a single json file

#     Parameters
#     ----------
#     video_metrics : list[Path]
#         List of paths to the video metrics json files
#     config : dict
#         Configuration dictionary

#     Returns
#     -------
#     None
#         Saves the aggregated metrics in a json file at a designated location

#     """

# Current metrics dictionary contains:
# - score_class
#   - precision
#   - recall
#   - f1
# - change_score
#   - precision
#   - recall
#   - f1
# - aggregated_score
#   - precision
#   - recall
#   - f1

# 1 - create empty dictionary, with same structure as metrics
# also add a field as counter with a list of all filenames included in this aggregation

# scores to aggregate:
scoagg = ["score_class", "change_score", "aggregated_score"]

# init empty dictionaries
agg_metrics = {}
agg_metrics["video_list"] = []
agg_metrics["top_k_frames"] = []
for i, single_video_metrics in enumerate(video_metrics_list):
    print(
        f"Processing of {single_video_metrics}, {i+1} out of {len(video_metrics_list)}"
    )
    # read json file
    with open(single_video_metrics, "r") as f:
        single_video_metrics = json.load(f)

    # check if first iteration, if so, init empty arrays
    if i == 0:
        agg_metrics["top_k"] = single_video_metrics["top_k"]
        agg_metrics["source_folder"] = single_video_metrics["source_folder"]
        temp_scores = {}
        for score_name in scoagg:  # init empty arrays for storing scores
            temp_scores[score_name] = {}
            temp_scores[score_name]["precision"] = np.zeros(
                (len(video_metrics_list), len(agg_metrics["top_k"]))
            )
            temp_scores[score_name]["recall"] = np.zeros(
                (len(video_metrics_list), len(agg_metrics["top_k"]))
            )
            temp_scores[score_name]["f1"] = np.zeros(
                (len(video_metrics_list), len(agg_metrics["top_k"]))
            )
    else:
        # check for consistency if results do come from same source (roughly speaking...)
        assert len(agg_metrics["top_k"]) == len(
            single_video_metrics["top_k"]
        ), f"top_k_frames are not consistent across videos: {agg_metrics['top_k']} vs {single_video_metrics['top_k']}"
        assert (
            agg_metrics["source_folder"] == single_video_metrics["source_folder"]
        ), "source_folder is not consistent across videos"

    # add video name to list
    agg_metrics["top_k_frames"].append(single_video_metrics["top_k_frames"])
    agg_metrics["video_list"].append(
        single_video_metrics["video_name"]
    )  # to be used as counter implicitly

    # loop over scores to aggregate
    for score_name in scoagg:
        # update means
        temp_scores[score_name]["precision"][i, :] = np.asarray(
            single_video_metrics[score_name]["precision"]
        )[np.newaxis, :]
        temp_scores[score_name]["recall"][i, :] = np.asarray(
            single_video_metrics[score_name]["recall"]
        )[np.newaxis, :]
        temp_scores[score_name]["f1"][i, :] = np.asarray(
            single_video_metrics[score_name]["f1"]
        )[np.newaxis, :]

# check if all has been processed as len(agg_metrics["video_list"]) == len(video_metrics)
assert len(agg_metrics["video_list"]) == i + 1, "Not all videos have been processed"

# compute means and stds into final dictionary
for score_name in scoagg:
    agg_metrics[score_name] = {}
    agg_metrics[score_name]["precision"] = {}
    agg_metrics[score_name]["precision"]["mean"] = np.mean(
        temp_scores[score_name]["precision"], axis=0
    ).tolist()
    agg_metrics[score_name]["precision"]["std"] = np.std(
        temp_scores[score_name]["precision"], axis=0
    ).tolist()
    agg_metrics[score_name]["recall"] = {}
    agg_metrics[score_name]["recall"]["mean"] = np.mean(
        temp_scores[score_name]["recall"], axis=0
    ).tolist()
    agg_metrics[score_name]["recall"]["std"] = np.std(
        temp_scores[score_name]["recall"], axis=0
    ).tolist()
    agg_metrics[score_name]["f1"] = {}
    agg_metrics[score_name]["f1"]["mean"] = np.mean(
        temp_scores[score_name]["f1"], axis=0
    ).tolist()
    agg_metrics[score_name]["f1"]["std"] = np.std(
        temp_scores[score_name]["f1"], axis=0
    ).tolist()

save_path = Path(args.results_path) / "_aggregated_metrics.json"
with open(save_path, "w") as f:
    json.dump(agg_metrics, f, indent=4)

# %%
