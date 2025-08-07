# %%
import os, sys

import argparse
import yaml
import json

import numpy as np
import pandas as pd

from pathlib import Path

from matplotlib import pyplot as plt

sys.path.append(".")

from src.utils import (
    cfg_to_arguments,
)


# %%
def precision(predictions: np.ndarray, gt: np.ndarray) -> float:
    """
    Computes precision given a list of predictions

    Parameters
    ----------
    predictions : list
        List of predictions, 1 if positive, 0 if negative
    gt : list
        List of ground truth, 1 if positive, 0 if negative

    Returns
    -------
    float
        Precision

    """
    return sum(predictions == gt) / sum(predictions)


def recall(predictions: np.ndarray, gt: np.ndarray) -> float:
    """
    Computes recall given a list of predictions

    Parameters
    ----------
    predictions : list
        List of predictions, 1 if positive, 0 if negative
    gt : list
        List of ground truth, 1 if positive, 0 if negative

    Returns
    -------
    float
        Recall

    """
    return sum(predictions == gt) / sum(gt[gt > -1]) if sum(gt[gt > -1]) > 0 else 0


# %%
def per_video_assessment(score_file: Path, top_k: list, config: dict) -> None:
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
    print(score_file)
    scores = pd.read_csv(score_file)
    print(scores.head())
    # Calculate score as linear combination of the three scores:
    # -classification, classificaiton dynamic, class-agnostic change detection
    # score_class, diff_score, change_score

    # sigmoid for change score
    # sigmoid = lambda x: 1 / (1 + np.exp(-x))
    # scores["change_score"] = scores["change_score"].apply(sigmoid)

    # min-max normalisation
    scores["score_class"] = (scores["score_class"] - scores["score_class"].min()) / (
        1e-6 + scores["score_class"].max() - scores["score_class"].min()
    )

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
            # Assume first K retrieved frames are positives after sorting:
            preds = np.ones((K,))
            # print(len(preds), len(score["ground_truth"].iloc[:K]))
            P_at_K = precision(preds, score["ground_truth"].iloc[:K].values)
            R_at_K = recall(preds, score["ground_truth"].iloc[:K].values)

            # print(preds, score["ground_truth"].iloc[:K])
            # print(len(preds), len(score["ground_truth"].iloc[:K]))
            # print(f"{pr_na} Pr@{K} = {P_at_K:.3f}")
            # print(f"{pr_na} Re@{K} = {R_at_K:.3f}")
            # exit()

            metrics[score_name]["precision"].append(P_at_K)
            metrics[score_name]["recall"].append(R_at_K)
            metrics[score_name]["f1"].append(
                2
                * metrics[score_name]["precision"][-1]
                * metrics[score_name]["recall"][-1]
                / (
                    1e-6
                    + metrics[score_name]["precision"][-1]
                    + metrics[score_name]["recall"][-1]
                )
            )

    # write json file
    metrics_file = Path(score_file).parent / f"{score_file.stem}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)


# %%
def aggregate_assessments(video_metrics_list: list[Path], config: dict) -> None:
    """
    This function aggregates the results of the per-video assessment into a single json file
    - Reads the metrics json files for each video
    - Aggregates the results for each K
    - Saves the results in a single json file

    Parameters
    ----------
    video_metrics_list : list[Path]
        List of paths to the video metrics json files
    config : dict
        Configuration dictionary

    Returns
    -------
    None
        Saves the aggregated metrics in a json file at a designated location

    """

    # scores to aggregate:
    scoagg = ["score_class", "change_score", "aggregated_score"]

    # init empty dictionaries
    agg_metrics = {}
    agg_metrics["video_list"] = []
    agg_metrics["top_k_frames"] = []
    for i, single_video_metrics in enumerate(video_metrics_list):
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
        agg_metrics[score_name]["precision"]["all"] = temp_scores[score_name][
            "precision"
        ].tolist()
        agg_metrics[score_name]["recall"] = {}
        agg_metrics[score_name]["recall"]["mean"] = np.mean(
            temp_scores[score_name]["recall"], axis=0
        ).tolist()
        agg_metrics[score_name]["recall"]["std"] = np.std(
            temp_scores[score_name]["recall"], axis=0
        ).tolist()
        agg_metrics[score_name]["recall"]["all"] = temp_scores[score_name][
            "recall"
        ].tolist()
        agg_metrics[score_name]["f1"] = {}
        agg_metrics[score_name]["f1"]["mean"] = np.mean(
            temp_scores[score_name]["f1"], axis=0
        ).tolist()
        agg_metrics[score_name]["f1"]["std"] = np.std(
            temp_scores[score_name]["f1"], axis=0
        ).tolist()
        agg_metrics[score_name]["f1"]["all"] = temp_scores[score_name]["f1"].tolist()
    save_path = Path(config.results_path) / "_aggregated_metrics.json"

    with open(save_path, "w") as f:
        json.dump(agg_metrics, f, indent=4)


# %%
def plot_aggregated_metrics(results_path: Path) -> None:
    """
    This function plots the aggregated metrics for a given model
    - Reads the aggregated metrics json file
    - Plots the results for each K

    Parameters
    ----------
    results_path : Path
        Path to the aggregated metrics json file

    Returns
    -------
    None
        Saves the aggregated metrics in a json file at a designated location

    """

    scores_to_test = ["score_class", "change_score", "aggregated_score"]
    markers = ["o", "x", "s"]
    cols = ["r", "b", "g"]
    ltype = ["-", "--", "-."]

    # read json file
    with open(Path(results_path) / "_aggregated_metrics.json", "r") as f:
        metrics = json.load(f)

    i = 0
    plt.figure()

    for score_name in scores_to_test:
        pr_na = score_name.replace("_", " ").capitalize()

        plt.plot(
            metrics[score_name]["precision"]["mean"],
            label=f"{pr_na} Pr",
            marker=markers[i],
            c=cols[i],
            ls=ltype[0],
        )
        plt.plot(
            metrics[score_name]["recall"]["mean"],
            label=f"{pr_na} Re",
            marker=markers[i],
            c=cols[i],
            ls=ltype[1],
        )
        plt.plot(
            metrics[score_name]["f1"]["mean"],
            label=f"{pr_na} F1",
            marker=markers[i],
            c=cols[i],
            ls=ltype[2],
        )
        plt.legend(handlelength=3)
        plt.grid()
        plt.xticks(range(len(metrics["top_k"])), metrics["top_k"], rotation=45)
        plt.xlabel("top K")
        plt.ylabel("metric")
        plt.title(f"Summary metrics")
        i += 1
    plt.tight_layout()
    plt.savefig(Path(results_path) / "_aggregated_metrics_plot.pdf")


# %%
if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Hummingbird inference script")
    args.add_argument(
        "--results_path",
        type=Path,
        help="Path to the video_scores sub-folder, specific to a model. This folder contains CSV files with the raw pipeline scores for each video",
    )
    args.add_argument(
        "--config_file",
        type=Path,
        help="Path to the configuration file",
    )
    args.add_argument(
        "--update",
        action="store_true",
        help="If set, will recompute the metrics for all videos in the folder",
    )
    args.add_argument(
        "--aggregate",
        action="store_true",
        help="If set, will aggregate the metrics for all videos in the folder (but only if --aggregate is set)",
    )
    args.add_argument(
        "--make_plots",
        action="store_true",
        help="If set, will make plots for the summary assessment",
    )
    args = args.parse_args()
    # args = {}
    # args["results_path"] = Path(
    #     "/data/shared/hummingbird-classifier/outputs/video_scores/densenet161-v1"
    # )
    # args["config_file"] = Path(
    #     "/data/shared/hummingbird-classifier/configs/configuration_hummingbirds.yaml"
    # )
    # args["update"] = True
    # args["aggregate"] = True
    # args = cfg_to_arguments(args)

    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = cfg_to_arguments(config)

    # compute per video assessment
    video_scores_list = sorted(list(Path(args.results_path).glob("*.csv")))
    if args.update:
        for video_score in video_scores_list[:]:
            # try:
            # print(f"Processing {video_score}")
            per_video_assessment(video_score, top_k=config.infe_top_k, config=config)
            # except:
            # print(f"Error processing {video_score}")
            # continue
    else:
        print(f"Skipping video assessment, update set to {args.update}")

    # aggregate results at same top_k rate
    if args.update and args.aggregate:
        video_metrics_list = sorted(list(Path(args.results_path).glob("*.json")))
        # remove _aggregated_metrics.json if present
        video_metrics_list = [
            x for x in video_metrics_list if "_aggregated_metrics.json" not in str(x)
        ]
        if args.update:
            aggregate_assessments(video_metrics_list, config=config)

    if args.make_plots and args.aggregate:
        plot_aggregated_metrics(args.results_path, config=config)
