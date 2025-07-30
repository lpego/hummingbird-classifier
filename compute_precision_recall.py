#!/usr/bin/env python3
"""
compute_precision_recall.py

Compute precision and recall for video anomaly detection for:
- Color Histogram
- Triplet Frame Analysis
- Running Mean
- Combined Scores
- Deep Learning Only

Ground truth is loaded from Weinstein2018MEE_ground_truth.csv.

Plots are from copilot :D
"""

import os
import argparse
import yaml

from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# could be made argument but it's mostly for debugging, not important
VERBOSE = False


def load_ground_truth(video_name, gt_file="./data/cleaned_ground_truth.csv"):
    """
    Load ground truth data for the specified video. If the CSV changes, the logic here fails
    Gotta double check:
        - Column names
        - Video names
        - Frame numbers

    Args:
        video_name: Name of the video (e.g., 'FH102_02')
        gt_folder: Folder containing the ground truth data (default: './data')
    Returns:
        gt_video: DataFrame with ground truth for the specified video
        positives: DataFrame with positive ground truth frames
        negatives: DataFrame with negative ground truth frames
    """

    # first check if files exists
    gt_file = os.path.abspath(gt_file)
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")

    gt = pd.read_csv(gt_file)

    # extract video GT
    gt_video = gt[gt["Video"] == video_name]
    gt_video = gt_video.set_index("Frame", drop=False)

    # shift frame index - 1 to make it 0-based
    gt_video.index = gt_video.index - 1

    # Deduplicate the index of ground truth
    gt_video = gt_video[~gt_video.index.duplicated(keep="first")]

    # problably unnecessary but was handy
    positives = gt_video[gt_video["Truth"].str.lower() == "positive"]
    negatives = gt_video[gt_video["Truth"].str.lower() == "negative"]

    return gt_video, positives, negatives


def load_diff_data(video_name, method="colorhist", results_folder="."):
    """
    Load processed difference data for the specified video.
    If filenames change, the logic here fails.
    Filenames are hardcoded, but could be passed as args but I'm too lazy.
    Not sure configs are necessary, but cheap to read in case.
    Args:
        video_name: Name of the video (e.g., 'FH102_02')
        method: 'colorhist', 'triplet', 'running_mean'
        results_folder: Folder containing the processed results

    Returns:
        df_change: DataFrame with processed difference data
        config: Configuration dictionary loaded from YAML file (if exists)

    """

    # First create filenames to read, then read
    if method == "colorhist":
        csv_file = os.path.join(
            results_folder,
            "color_histogram",  # this is a subfolder, should be consistent
            f"{video_name}_processed_chist_diff.csv",
        )

        # not sure is needed
        config_file = os.path.join(
            results_folder, "color_histogram", f"{video_name}_chist_config.yaml"
        )

    elif method == "triplet":
        csv_file = os.path.join(
            results_folder,
            "triplet_analysis",  # this is a subfolder, should be consistent
            f"{video_name}_triplet_diff.csv",
        )
        config_file = os.path.join(
            results_folder,
            "triplet_analysis",  # this is a subfolder, should be consistent
            f"{video_name}_triplet_config.yaml",
        )
    elif method == "running_mean":
        csv_file = os.path.join(
            results_folder,
            "running_mean",  # this is a subfolder, should be consistent
            f"{video_name}_running_mean_diff.csv",
        )
        config_file = os.path.join(
            results_folder, "running_mean", f"{video_name}_running_mean_config.yaml"
        )
    else:
        raise ValueError("method must be 'colorhist', 'triplet', or 'running_mean'")

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Processed data file not found: {csv_file}")

    df_change = pd.read_csv(csv_file)
    df_change.index = df_change["center_idx"]
    df_change = df_change.drop(columns=["center_idx"])

    config = None
    if os.path.exists(config_file):
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

    return df_change, config


def load_combined_data(video_name, results_folder="."):
    """
    Load combined scores data for the specified video, if pre-existing combined data CSV and dl-only CSV exists.

    Args:
        video_name: Name of the video (e.g., 'FH102_02')
        results_folder: Folder containing the combined results
    """

    csv_file = os.path.join(results_folder, f"{video_name}_combined_scores.csv")

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Combined scores file not found: {csv_file}")

    df_combined = pd.read_csv(csv_file)
    df_combined.index = df_combined["center_idx"]
    df_combined = df_combined.drop(
        columns=["center_idx", "video_name"], errors="ignore"
    )

    return df_combined


def compute_aggregated_diff(df_change, method="colorhist"):
    """
    Compute aggregated difference score based on the method, in case a script returns multiple scores.
    Potentially, this should not be used and first go through the score combination script.
    Right now is basically for renaming columns.

    Args:
        df_change: dataframe containing anomaly
        method: method name for defining cols to rename ('colorhist', 'triplet', 'running_mean')

    Returns:
        df_change: DataFrame with aggregated_diff column added

    Raises:
        ValueError if column name is wrong.
    """
    if method == "colorhist":
        # For color histogram analysis: use the stdev_magn_diff_chist column directly
        df_change["aggregated_diff"] = df_change["stdev_magn_diff_chist"]

    elif method == "triplet":
        # For triplet frame analysis: aggregate all std_diff_rgb_{frame_skip} columns
        triplet_cols = [
            col for col in df_change.columns if col.startswith("std_diff_rgb_")
        ]
        if triplet_cols:
            df_change["aggregated_diff"] = df_change[triplet_cols].mean(axis=1)
        else:
            raise ValueError("Triplet analysis data missing std_diff_rgb_ columns")

    elif method == "running_mean":
        # For running mean analysis: aggregate all std_diff_running_mean_{N} columns
        running_mean_cols = [
            col for col in df_change.columns if col.startswith("std_diff_running_mean_")
        ]
        if running_mean_cols:
            df_change["aggregated_diff"] = df_change[running_mean_cols].mean(axis=1)
        else:
            raise ValueError(
                "Running mean analysis data missing std_diff_running_mean_ columns"
            )

    elif method == "combined":
        # For combined scores: use the combined_score column directly (already normalized)
        if "combined_score" in df_change.columns:
            df_change["aggregated_diff"] = df_change["combined_score"]
        else:
            raise ValueError("Combined scores data missing 'combined_score' column")
        # Skip normalization since combined_score is already 0-1 normalized
        return df_change

    elif method == "dl_only":
        # For deep learning only: use the dl_score column directly (already normalized)
        if "dl_score" in df_change.columns:
            df_change["aggregated_diff"] = df_change["dl_score"]
        else:
            raise ValueError("Combined scores data missing 'dl_score' column")
        # Skip normalization since dl_score is already 0-1 normalized
        return df_change

    # 0-1 normalization
    # this might be delicate -- but as we assess by sorting and not by thresholding, it should be fine
    min_val = df_change["aggregated_diff"].min()
    max_val = df_change["aggregated_diff"].max()
    df_change["aggregated_diff"] = (df_change["aggregated_diff"] - min_val) / (
        max_val - min_val
    )

    return df_change


def compute_precision_recall(df_change, positives, k_values, buffer=1):
    """
    Compute precision and recall for different k values.
    k defines the number of top frames to consider for precision/recall after sorting the score
    The Score is now in the column "aggregated_diff" and is already normalized to 0-1, _for all the methods_.

    Args:
        df_change: DataFrame with aggregated_diff scores
        positives: DataFrame with positive ground truth frames
        k_values: integer or list of integers. List of k values to compute precision/recall for
        buffer: int. Buffer around ground truth frames for matching (e.g. 1 means exact frame-to-frame match, 3 means counted as poisitive if prediction is within 3 frames of the ground truth)
            I did the above as I had the feeling the GT is a bit inconsistent, but I use 1 nonetheless (same for all methods -- pensalises equally)

    Returns:
        DataFrame with precision and recall for each k value
    """
    # Sort frames by aggregated_diff score (descending)
    sorted_scores = df_change.sort_values(by="aggregated_diff", ascending=False)

    # Create results DataFrame to store whether each frame is within buffer of positive
    # this is the way predictionsa are scored. Negatives are just given by the number of frames _not_ matched, for each k value.
    # Uses logic from instance retrieval evaluation
    results = []
    for frame in sorted_scores.index:
        # do the matching with the buffer
        is_within_range = any(
            (positives["Frame"] >= frame - buffer)
            & (positives["Frame"] <= frame + buffer)
        )
        results.append({"Frame": frame, f"within_{buffer}_positive": is_within_range})

    results_df = pd.DataFrame(results)

    # Compute precision and recall for different k values
    precision_recall_data = []

    for k in k_values:
        if k > len(results_df):  # stop if k is larger than the number of frames
            continue

        # Precision: fraction of retrieved frames that are correct (assuming buffer)
        precision_at_k = results_df[f"within_{buffer}_positive"].iloc[:k].sum() / k

        # Recall: fraction of positive frames that are retrieved and are within the buffer
        true_positives = results_df.loc[:, "Frame"].iloc[:k].isin(positives["Frame"])
        recall_at_k = true_positives.sum() / len(positives)

        precision_recall_data.append(
            {
                "k": k,
                "Precision": precision_at_k,
                "Recall": recall_at_k,
                "num_positives": len(positives),
            }
        )
    return pd.DataFrame(precision_recall_data)


def create_precision_recall_plots(precision_recall_df, output_folder, video_name):
    """
    Create and save precision-recall plots for both methods. Gladly helped by copilot.

    Args:
        precision_recall_df: DataFrame with precision and recall data for both methods
        output_folder: Folder to save the plots
        video_name: Name of the video

    Returns:
        None, saves plots to the specified output folder

    """

    # Separate data by method
    methods = precision_recall_df["method"].unique()
    colors = {
        "colorhist": "red",
        "triplet": "blue",
        "running_mean": "purple",
        "combined": "green",
        "dl_only": "orange",
    }
    markers = {
        "colorhist": "s",
        "triplet": "o",
        "running_mean": "^",
        "combined": "d",
        "dl_only": "v",
    }

    # Calculate random baseline precision (proportion of positive samples).
    # This ought to be abysmally low, as the dataset is highly imbalanced.
    num_positives = precision_recall_df["num_positives"].iloc[0]
    total_frames = precision_recall_df["total_frames"].iloc[0]
    random_precision = num_positives / total_frames

    if VERBOSE:
        print(f"Dataset statistics for {video_name}:")
        print(f"++Total frames: {total_frames}")
        print(f"++Positive frames: {num_positives}")
        print(
            f"++Random baseline precision: {random_precision:.4f} ({random_precision*100:.2f}%)"
        )

    # Create comparison plots
    plt.figure(figsize=(16, 10))
    plt.suptitle(
        f"Precision-Recall assessment for {video_name}",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Precision vs k for both methods
    plt.subplot(2, 3, 1)
    for method in methods:
        method_data = precision_recall_df[precision_recall_df["method"] == method]
        plt.plot(
            method_data["k"],
            method_data["Precision"],
            label=f"Precision ({method})",
            marker=markers[method],
            linewidth=2,
            markersize=4,
            color=colors[method],
            alpha=0.7,
        )

    # Add horizontal line for random baseline precision
    plt.axhline(
        y=random_precision,
        color="gray",
        linestyle=":",
        label=f"Random baseline ({random_precision:.4f})",
        alpha=0.7,
    )

    # Add vertical line for actual positives (same for both methods)
    plt.axvline(
        x=num_positives,
        color="gray",
        linestyle="--",
        label=f"GT positives ({num_positives})",
        alpha=0.7,
    )

    plt.title(f"Precision @ k")
    plt.xlabel("Top-k frames retrieved")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, 1.05)

    # Remove top and right spines
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Plot 2: Recall vs k for both methods
    plt.subplot(2, 3, 2)
    for method in methods:
        method_data = precision_recall_df[precision_recall_df["method"] == method]
        plt.plot(
            method_data["k"],
            method_data["Recall"],
            label=f"Recall ({method})",
            marker=markers[method],
            linewidth=2,
            markersize=4,
            color=colors[method],
            alpha=0.7,
        )

    plt.axvline(
        x=num_positives,
        color="gray",
        linestyle="--",
        label=f"GT positives ({num_positives})",
        alpha=0.7,
    )

    plt.title(f"Recall @ k")
    plt.xlabel("Top-k frames retrieved")
    plt.ylabel("Recall")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, 1.05)

    # Remove top and right spines
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Plot 3: Precision-Recall curves for both methods
    plt.subplot(2, 3, 3)
    for method in methods:
        method_data = precision_recall_df[precision_recall_df["method"] == method]
        plt.plot(
            method_data["Recall"],
            method_data["Precision"],
            marker=markers[method],
            linewidth=2,
            markersize=3,
            color=colors[method],
            label=f"{method}",
            alpha=0.8,
        )

    plt.title(f"Precision-Recall curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)

    # Add horizontal line for random baseline (correct for imbalanced datasets)
    plt.axhline(
        y=random_precision,
        color="gray",
        linestyle=":",
        alpha=0.7,
        label=f"Random baseline ({random_precision:.4f})",
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Remove top and right spines
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Plot 4: F1 Score comparison
    plt.subplot(2, 3, 4)
    for method in methods:
        method_data = precision_recall_df[
            precision_recall_df["method"] == method
        ].reset_index(drop=True)
        f1_scores = (
            2
            * (method_data["Precision"] * method_data["Recall"])
            / (method_data["Precision"] + method_data["Recall"])
        )
        f1_scores = f1_scores.fillna(0)

        plt.plot(
            method_data["k"],
            f1_scores,
            marker=markers[method],
            linewidth=2,
            markersize=4,
            color=colors[method],
            label=f"F1 ({method})",
            alpha=0.8,
        )

        # Mark the best F1 score for this method (with proper error handling)
        if len(f1_scores) > 0 and f1_scores.max() > 0:
            best_f1_idx = f1_scores.idxmax()
            best_k = method_data.iloc[best_f1_idx]["k"]
            best_f1 = f1_scores.iloc[best_f1_idx]

            plt.scatter(
                best_k,
                best_f1,
                color=colors[method],
                s=100,
                zorder=5,
                edgecolor="black",
                linewidth=1,
            )
            # Add text box at top of plot instead of annotations
            if method == "colorhist":
                plt.text(
                    0.02,
                    0.98,
                    f"Color Hist - Best F1: \n {best_f1:.3f} @ k={best_k}",
                    transform=plt.gca().transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor=colors[method], alpha=0.3
                    ),
                )
            elif method == "triplet":
                plt.text(
                    0.02,
                    0.88,
                    f"Triplet - Best F1: \n {best_f1:.3f} @ k={best_k}",
                    transform=plt.gca().transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor=colors[method], alpha=0.3
                    ),
                )
            elif method == "running_mean":
                plt.text(
                    0.02,
                    0.78,
                    f"Running Mean - Best F1: \n {best_f1:.3f} @ k={best_k}",
                    transform=plt.gca().transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor=colors[method], alpha=0.3
                    ),
                )
            elif method == "combined":
                plt.text(
                    0.02,
                    0.68,
                    f"Combined - Best F1: \n {best_f1:.3f} @ k={best_k}",
                    transform=plt.gca().transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor=colors[method], alpha=0.3
                    ),
                )
            elif method == "dl_only":
                plt.text(
                    0.02,
                    0.58,
                    f"DL Only - Best F1: \n {best_f1:.3f} @ k={best_k}",
                    transform=plt.gca().transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor=colors[method], alpha=0.3
                    ),
                )

    plt.title(f"F1 score @ k")
    plt.xlabel("Top-k frames retrieved")
    plt.ylabel("F1 score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, 1.05)

    # Remove top and right spines
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Plot 5: Log scale precision/recall
    plt.subplot(2, 3, 5)
    for method in methods:
        method_data = precision_recall_df[precision_recall_df["method"] == method]
        plt.semilogx(
            method_data["k"],
            method_data["Precision"],
            label=f"Precision ({method})",
            marker=markers[method],
            linewidth=2,
            markersize=3,
            color=colors[method],
            alpha=0.7,
        )
        plt.semilogx(
            method_data["k"],
            method_data["Recall"],
            label=f"Recall ({method})",
            marker=markers[method],
            linewidth=2,
            markersize=3,
            color=colors[method],
            alpha=0.7,
            linestyle="--",
        )

    plt.axvline(
        x=num_positives,
        color="gray",
        linestyle="--",
        label=f"GT rositives ({num_positives})",
        alpha=0.7,
    )

    plt.title(f"Precision/Recall @ k (log scale)")
    plt.xlabel("Top-k frames retrieved (log scale)")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)

    # Remove top and right spines
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Plot 6: Summary metrics comparison
    plt.subplot(2, 3, 6)
    # Calculate summary metrics for each method
    summary_data = []
    for method in methods:
        method_data = precision_recall_df[precision_recall_df["method"] == method]
        f1_scores = (
            2
            * (method_data["Precision"] * method_data["Recall"])
            / (method_data["Precision"] + method_data["Recall"])
        )
        f1_scores = f1_scores.fillna(0)

        # Get precision and recall at k=200 if available
        kf = 200
        at_kf_data = method_data[method_data["k"] >= kf]
        prec_at_kf = at_kf_data["Precision"].iloc[0] if len(at_kf_data) > 0 else 0
        recall_at_kf = at_kf_data["Recall"].iloc[0] if len(at_kf_data) > 0 else 0

        # Get k values for max precision, max recall, and best F1
        max_prec_k = method_data.loc[method_data["Precision"].idxmax(), "k"]
        max_recall_k = method_data.loc[method_data["Recall"].idxmax(), "k"]
        best_f1_k = (
            method_data.loc[f1_scores.idxmax(), "k"] if f1_scores.max() > 0 else 0
        )

        summary_data.append(
            {
                "Method": method,
                "Max precision": method_data["Precision"].max(),
                "Max precision k": max_prec_k,
                "Max recall": method_data["Recall"].max(),
                "Max recall k": max_recall_k,
                "Best F1": f1_scores.max(),
                "Best F1 k": best_f1_k,
                f"Precision@{kf}": prec_at_kf,
                f"Recall@{kf}": recall_at_kf,
            }
        )

    summary_df = pd.DataFrame(summary_data)

    # Create grouped bar plot
    x = np.arange(5)  # 5 metrics to display
    width = 0.2

    metrics = [
        "Max precision",
        "Max recall",
        "Best F1",
        f"Precision@{kf}",
        f"Recall@{kf}",
    ]

    for i, method in enumerate(methods):
        method_row = summary_df[summary_df["Method"] == method].iloc[0]
        method_values = [method_row[metric] for metric in metrics]

        # Get corresponding k values
        k_values = [
            method_row["Max precision k"],
            method_row["Max recall k"],
            method_row["Best F1 k"],
            50,  # Fixed k=50 for @50 metrics
            50,
        ]

        plt.bar(
            x + i * width,
            method_values,
            width,
            label=method,
            color=colors[method],
            alpha=0.5,
            edgecolor="black",
        )
        # Add value labels on bars with k values
        for j, (value, k_val) in enumerate(zip(method_values, k_values)):
            # Position text on top of bar if value is below 0.6, otherwise inside the bar
            if value < 0.6:
                y_pos = value + 0.01  # Above the bar
                va = "bottom"
            else:
                y_pos = value - 0.13  # Inside the bar
                va = "center"

            plt.text(
                x[j] + i * width,
                y_pos,
                f"{value:.3f} (k={k_val})",
                ha="center",
                va=va,
                fontsize=7,
                fontweight="bold",
                rotation=90,
                color="black",
            )

    plt.title(f"Summary")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)  # Increased to accommodate text labels
    plt.grid(True, alpha=0.3, axis="y")
    plt.xticks(x + width / 2, metrics, rotation=45)
    plt.legend()

    # Remove top and right spines
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Save the comparison plot
    plot_filename = os.path.join(
        output_folder, f"{video_name}_comparison_precision_recall_curves.png"
    )
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()

    if VERBOSE:
        print(f"Comparison plots saved to: {plot_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute precision and recall for video anomaly detection using both methods"
    )
    parser.add_argument(
        "video_name", type=str, help="Name of the video (e.g., FH102_02)"
    )
    parser.add_argument(
        "--method",
        type=str,
        nargs="+",  # Allow multiple values
        choices=["colorhist", "triplet", "running_mean", "combined", "dl_only", "all"],
        default=["all"],
        help="Method(s) to use: colorhist, triplet, running_mean, combined, dl_only, or all. Can specify multiple methods (default: all)",
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=1,
        help="Buffer around ground truth frames for matching (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file path (default: {video_name}_precision_recall_comparison.csv)",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Create and save precision-recall plots"
    )
    parser.add_argument(
        "--plot-folder",
        type=str,
        help="Folder to save plots (default: same folder as CSV output)",
    )
    parser.add_argument(
        "--results-folder",
        type=str,
        default=".",
        help="Folder containing the processed video results (default: current directory)",
    )
    parser.add_argument(
        "--gt-folder",
        type=str,
        default="./data",
        help="Folder containing the ground truth data (default: ./data)",
    )

    args = parser.parse_args()

    # Set default output filename if not provided
    if args.output is None:
        if args.method == "all":
            args.output = f"{args.video_name}_precision_recall_comparison.csv"
        else:
            args.output = f"{args.video_name}_precision_recall.csv"
    else:
        # Check if the provided output is a directory
        if os.path.isdir(args.output):
            # If it's a directory, append the appropriate filename
            if args.method == "all":
                filename = f"{args.video_name}_precision_recall_comparison.csv"
            else:
                filename = f"{args.video_name}_precision_recall.csv"
            args.output = os.path.join(args.output, filename)

    # Ensure output is an absolute path
    args.output = os.path.abspath(args.output)
    if VERBOSE:
        print(f"Output will be saved to: {args.output}")

    # Set plot folder to same directory as CSV output
    plot_folder = os.path.dirname(args.output)
    if not plot_folder:
        plot_folder = "."

    if VERBOSE:
        print(f"Computing precision and recall for video: {args.video_name}")
        print(f"Method(s): {args.method}")
        print(f"Buffer: {args.buffer}")

    # Determine which methods to run
    if "all" in args.method:
        methods_to_run = ["colorhist", "triplet", "running_mean", "combined", "dl_only"]
    else:
        methods_to_run = args.method

    all_results = []

    # try:
    # Load ground truth (same for all methods)
    if VERBOSE:
        print("Loading ground truth data...")

    gt_video, positives, negatives = load_ground_truth(args.video_name, args.gt_folder)

    if VERBOSE:
        print(
            f"Found {len(positives)} positive frames and {len(negatives)} negative frames"
        )

    if len(positives) == 0:
        print("Warning: No positive frames found in ground truth")
        return

    # Process each method
    for method in methods_to_run:
        if VERBOSE:
            print(f"\n--- Processing {method} ---")

        try:
            # Load processed difference data
            if VERBOSE:
                print(f"Loading {method} data...")

            if method in ["combined", "dl_only"]:
                df_change = load_combined_data(args.video_name, args.results_folder)
                config = None  # No config file for combined scores
            else:
                df_change, config = load_diff_data(
                    args.video_name, method, args.results_folder
                )

            if VERBOSE:
                print(f"Loaded data with {len(df_change)} frames")
                print("Computing aggregated difference scores...")

            df_change = compute_aggregated_diff(df_change, method)

            # Define k values for precision/recall computation
            k_values = (
                list(range(10, 201, 10))
                + list(range(200, 1001, 100))
                + list(range(1000, 2001, 500))
            )

            # Filter k values to not exceed the number of frames
            k_values = [k for k in k_values if k <= len(df_change)]

            if VERBOSE:
                print(
                    f"Computing precision and recall for {len(k_values)} different k values..."
                )

            # Compute precision and recall
            precision_recall_df = compute_precision_recall(
                df_change, positives, k_values, args.buffer
            )

            # Add metadata columns
            precision_recall_df["video_name"] = args.video_name
            precision_recall_df["method"] = method
            precision_recall_df["buffer"] = args.buffer
            precision_recall_df["total_frames"] = len(df_change)

            all_results.append(precision_recall_df)

            if VERBOSE:
                print(
                    f"{method} - Max Precision: {precision_recall_df['Precision'].max():.4f}"
                )
                print(
                    f"{method} - Max Recall: {precision_recall_df['Recall'].max():.4f}"
                )

        except FileNotFoundError as e:
            print(f"Warning: Could not process {method} - {e}")
            continue

    if not all_results:
        print("Error: No methods could be processed successfully")
        return 1

    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)

    # Reorder columns
    cols = [
        "video_name",
        "method",
        "buffer",
        "k",
        "Precision",
        "Recall",
        "num_positives",
        "total_frames",
    ]
    combined_df = combined_df[cols]

    # Save combined results
    combined_df.to_csv(args.output, index=False)

    if VERBOSE:
        print(f"\nCombined results saved to: {args.output}")

    # Create comparison plots if requested
    if args.plot:
        if VERBOSE:
            print("Creating comparison plots...")
        create_precision_recall_plots(combined_df, plot_folder, args.video_name)

    # Print summary statistics for comparison
    if VERBOSE:
        print("\n=== COMPARISON SUMMARY ===")

    for method in combined_df["method"].unique():
        method_data = combined_df[combined_df["method"] == method].reset_index(
            drop=True
        )

        # Calculate F1 scores
        f1_scores = (
            2
            * (method_data["Precision"] * method_data["Recall"])
            / (method_data["Precision"] + method_data["Recall"])
        )
        f1_scores = f1_scores.fillna(0)

        # Find the best F1 score
        if len(f1_scores) > 0 and f1_scores.max() > 0:
            best_f1_idx = f1_scores.idxmax()
            best_f1_value = f1_scores.iloc[best_f1_idx]
            best_k = method_data.iloc[best_f1_idx]["k"]
            best_precision = method_data.iloc[best_f1_idx]["Precision"]
            best_recall = method_data.iloc[best_f1_idx]["Recall"]
        else:
            best_f1_value = 0
            best_k = 0
            best_precision = 0
            best_recall = 0

        if VERBOSE:
            print(f"\n{method.upper()}:")
            print(f"  Max Precision: {method_data['Precision'].max():.4f}")
            print(f"  Max Recall: {method_data['Recall'].max():.4f}")
            print(f"  Best F1 Score: {best_f1_value:.4f} at k={best_k}")
            print(f"  Precision: {best_precision:.4f}")
            print(f"  Recall: {best_recall:.4f}")

    # except Exception as e:
    #     print(f"Error: {e}")
    #     return 1

    return 0


if __name__ == "__main__":
    exit(main())
