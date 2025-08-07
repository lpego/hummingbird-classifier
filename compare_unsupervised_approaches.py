#!/usr/bin/env python3
"""
compare_unsupervised_approaches.py

Script to compare unsupervised approaches for hummingbird detection:
- Color histogram differences
- Triplet frame differences
- Running mean background subtraction
- Combined multi-criteria scoring

Computes precision and recall for each approach and creates comparison plots.
"""

import argparse
from argparse import Namespace

import pandas as pd
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt

global VERBOSE


def load_ground_truth(
    video_name: str, gt_file: str = "./data/cleaned_ground_truth.csv"
) -> tuple:
    """
    Load ground truth data for the specified video. If the CSV changes, the logic here fails
    Gotta double check:
        - Column names
        - Video names
        - Frame numbers

    Args:
        video_name: Name of the video (e.g., 'FH102_02')
        gt_file: Path to the ground truth CSV file (default: './data/cleaned_ground_truth.csv')
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


def load_unsupervised_data(
    video_name: str,
    method: str = "euclidean",
    results_folder: str = "./results/hummingbird",
) -> tuple:
    """
    Load processed data for unsupervised approaches.

    Args:
        video_name: Name of the video (e.g., 'FH102_02')
        method: 'euclidean', 'triplet', 'running_mean', 'combined', 'wasserstein', 'chi_square'
        results_folder: Base folder containing the analysis results

    Returns:
        df_change: DataFrame with processed change scores
        config: Configuration dictionary loaded from YAML file (if exists)
    """

    # First create filenames to read, then read
    if method == "euclidean":
        # Load from euclidean subfolder
        csv_file = os.path.join(
            results_folder,
            "euclidean",
            f"{video_name}_euclidean_diff.csv",
        )
        config_file = os.path.join(
            results_folder, "euclidean", f"{video_name}_euclidean_config.yaml"
        )
    elif method == "triplet":
        # Load from triplet_analysis subfolder
        csv_file = os.path.join(
            results_folder,
            "triplet_analysis",
            f"{video_name}_triplet_diff.csv",
        )
        config_file = os.path.join(
            results_folder, "triplet_analysis", f"{video_name}_triplet_config.yaml"
        )
    elif method == "running_mean":
        # Load from running_mean subfolder
        csv_file = os.path.join(
            results_folder,
            "running_mean",
            f"{video_name}_running_mean_diff.csv",
        )
        config_file = os.path.join(
            results_folder, "running_mean", f"{video_name}_running_mean_config.yaml"
        )
    elif method == "wasserstein":
        # Load from wasserstein subfolder
        csv_file = os.path.join(
            results_folder,
            "wasserstein",
            f"{video_name}_wasserstein_diff.csv",
        )
        config_file = os.path.join(
            results_folder, "wasserstein", f"{video_name}_wasserstein_config.yaml"
        )
    elif method == "chi_square":
        # Load from chi_square subfolder
        csv_file = os.path.join(
            results_folder,
            "chi_square",
            f"{video_name}_chi_square_diff.csv",
        )
        config_file = os.path.join(
            results_folder, "chi_square", f"{video_name}_chi_square_config.yaml"
        )
    elif method == "combined":
        # Load from combined subfolder
        csv_file = os.path.join(
            results_folder,
            "combined",
            f"{video_name}_combined_scores.csv",
        )
        config_file = None  # No config for combined scores
    else:
        raise ValueError(
            "method must be 'euclidean', 'triplet', 'running_mean', 'wasserstein', 'chi_square', or 'combined'"
        )

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Processed data file not found: {csv_file}")

    df_change = pd.read_csv(csv_file)
    df_change.index = df_change["center_idx"]
    df_change = df_change.drop(columns=["center_idx"])

    config = None
    if config_file and os.path.exists(config_file):
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

    return df_change, config


def compute_aggregated_diff(
    df_change: pd.DataFrame, method: str = "euclidean"
) -> pd.DataFrame:
    """
    Compute aggregated difference score based on the method, in case a script returns multiple scores.
    Potentially, this should not be used and first go through the score combination script.
    Right now is basically for renaming columns and checking consistency

    Args:
        df_change: dataframe containing anomaly
        method: method name for defining cols to rename ('euclidean', 'triplet', 'running_mean', 'wasserstein', 'chi_square')

    Returns:
        df_change: DataFrame with aggregated_diff column added

    Raises:
        ValueError if column name is wrong.
    """

    if method == "euclidean":
        # For color histogram analysis: use the stdev_magn_diff_chist column directly
        if "stdev_magn_diff_chist" in df_change.columns:
            df_change["aggregated_diff"] = df_change["stdev_magn_diff_chist"]
        else:
            raise ValueError(
                "Color histogram data missing 'stdev_magn_diff_chist' column"
            )

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

    elif method == "wasserstein":
        # For Wasserstein distance analysis: check multiple possible column names
        if "wasserstein_distance" in df_change.columns:
            df_change["aggregated_diff"] = df_change["wasserstein_distance"]
        elif "stdev_magn_diff_wasserstein" in df_change.columns:
            df_change["aggregated_diff"] = df_change["stdev_magn_diff_wasserstein"]
        elif "stdev_magn_diff_chist" in df_change.columns:
            # Fallback: the actual column name found in the data
            df_change["aggregated_diff"] = df_change["stdev_magn_diff_chist"]
        else:
            raise ValueError(
                "Wasserstein analysis data missing expected columns. Available columns: "
                + str(list(df_change.columns))
            )

    elif method == "chi_square":
        # For chi-square analysis: check multiple possible column names
        if "chi_square_statistic" in df_change.columns:
            df_change["aggregated_diff"] = df_change["chi_square_statistic"]
        elif "stdev_magn_diff_chi_square" in df_change.columns:
            df_change["aggregated_diff"] = df_change["stdev_magn_diff_chi_square"]
        elif "stdev_magn_diff_chist" in df_change.columns:
            # Fallback: the actual column name found in the data
            df_change["aggregated_diff"] = df_change["stdev_magn_diff_chist"]
        else:
            raise ValueError(
                "Chi-square analysis data missing expected columns. Available columns: "
                + str(list(df_change.columns))
            )

    elif method == "combined":
        # For combined scores: use the combined_score column directly (already normalized)
        if "combined_score" in df_change.columns:
            df_change["aggregated_diff"] = df_change["combined_score"]
        else:
            raise ValueError("Combined scores data missing 'combined_score' column")
        # Skip normalization since combined_score is already 0-1 normalized
        return df_change

    # 0-1 normalization
    min_val = df_change["aggregated_diff"].min()
    max_val = df_change["aggregated_diff"].max()
    if max_val > min_val:
        df_change["aggregated_diff"] = (df_change["aggregated_diff"] - min_val) / (
            max_val - min_val
        )
    else:
        df_change["aggregated_diff"] = 0.0

    return df_change


def compute_precision_recall(
    df_change: pd.DataFrame, positives: pd.DataFrame, k_values: list, buffer: int = 1
) -> pd.DataFrame:
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
    results = []
    for frame in sorted_scores.index:
        is_within_range = any(
            (positives["Frame"] >= frame - buffer)
            & (positives["Frame"] <= frame + buffer)
        )
        results.append({"Frame": frame, f"within_{buffer}_positive": is_within_range})

    results_df = pd.DataFrame(results)

    # Compute precision and recall for different k values
    precision_recall_data = []

    for k in k_values:
        if k > len(results_df):
            continue

        # Precision: fraction of retrieved frames that are relevant (within buffer of positive)
        precision_at_k = results_df[f"within_{buffer}_positive"].iloc[:k].sum() / k

        # Recall: fraction of relevant frames that are retrieved (exact matches)
        true_positives = results_df["Frame"].iloc[:k].isin(positives["Frame"])
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


def create_precision_recall_plots(
    precision_recall_df: pd.DataFrame, output_folder: str, video_name: str
) -> None:
    """
    Create and save precision-recall comparison plots for unsupervised approaches.

    Args:
        precision_recall_df: DataFrame with precision and recall data for all methods
        output_folder: Folder to save the plots
        video_name: Name of the video

    """

    # Separate data by method
    methods = precision_recall_df["method"].unique()
    colors = {
        "euclidean": "red",
        "triplet": "blue",
        "running_mean": "purple",
        "combined": "green",
        "wasserstein": "orange",
        "chi_square": "brown",
    }
    markers = {
        "euclidean": "s",
        "triplet": "o",
        "running_mean": "^",
        "combined": "d",
        "wasserstein": "v",
        "chi_square": "x",
    }

    # Calculate random baseline precision (proportion of positive samples)
    num_positives = precision_recall_df["num_positives"].iloc[0]
    total_frames = precision_recall_df["total_frames"].iloc[0]
    random_precision = num_positives / total_frames

    if VERBOSE:
        print(f"Dataset statistics for {video_name}:")
        print(f"  Total frames: {total_frames}")
        print(f"  Positive frames: {num_positives}")
        print(
            f"  Random baseline precision: {random_precision:.4f} ({random_precision*100:.2f}%)"
        )

    # Create comparison plots with adjusted layout for bottom legend
    fig = plt.figure(figsize=(16, 12))  # Increased height for legend space
    plt.suptitle(
        f"Comparison for {video_name}",
        fontsize=16,
        fontweight="bold",
    )

    # Adjust subplot layout to leave space for legend at bottom
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.2], hspace=0.3, wspace=0.3)

    # Plot 1: Precision vs k for all methods
    ax1 = fig.add_subplot(gs[0, 0])
    for method in methods:
        method_data = precision_recall_df[precision_recall_df["method"] == method]
        ax1.plot(
            method_data["k"],
            method_data["Precision"],
            # Removed label for methods - shown in master legend
            marker=markers[method],
            linewidth=2,
            markersize=4,
            color=colors[method],
            alpha=0.7,
        )

    # Add horizontal line for random baseline precision
    ax1.axhline(
        y=random_precision,
        color="gray",
        linestyle=":",
        label=f"Random baseline ({random_precision:.4f})",
        alpha=0.7,
    )

    # Add vertical line for actual positives
    ax1.axvline(
        x=num_positives,
        color="gray",
        linestyle="--",
        label=f"Actual Positives ({num_positives})",
        alpha=0.7,
    )

    ax1.set_title(f"Precision @ k")
    ax1.set_xlabel("Top-k Frames Retrieved")
    ax1.set_ylabel("Precision")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim(0, 1.05)

    # Remove top and right spines
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Plot 2: Recall vs k for all methods
    ax2 = fig.add_subplot(gs[0, 1])
    for method in methods:
        method_data = precision_recall_df[precision_recall_df["method"] == method]
        ax2.plot(
            method_data["k"],
            method_data["Recall"],
            # Removed label for methods - shown in master legend
            marker=markers[method],
            linewidth=2,
            markersize=4,
            color=colors[method],
            alpha=0.7,
        )

    ax2.axvline(
        x=num_positives,
        color="gray",
        linestyle="--",
        label=f"Actual Positives ({num_positives})",
        alpha=0.7,
    )

    ax2.set_title(f"Recall @ k")
    ax2.set_xlabel("Top-k Frames Retrieved")
    ax2.set_ylabel("Recall")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim(0, 1.05)

    # Remove top and right spines
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Plot 3: Precision-Recall curves for all methods
    ax3 = fig.add_subplot(gs[0, 2])
    for method in methods:
        method_data = precision_recall_df[precision_recall_df["method"] == method]
        ax3.plot(
            method_data["Recall"],
            method_data["Precision"],
            marker=markers[method],
            linewidth=2,
            markersize=3,
            color=colors[method],
            # Removed label for methods - shown in master legend
            alpha=0.8,
        )

    ax3.set_title(f"Precision-Recall Curves")
    ax3.set_xlabel("Recall")
    ax3.set_ylabel("Precision")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1.05)
    ax3.set_ylim(0, 1.05)

    # Add horizontal line for random baseline
    ax3.axhline(
        y=random_precision,
        color="gray",
        linestyle=":",
        alpha=0.7,
        label=f"Random baseline ({random_precision:.4f})",
    )
    ax3.legend()

    # Remove top and right spines
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Plot 4: F1 Score comparison
    ax4 = fig.add_subplot(gs[1, 0])
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

        ax4.plot(
            method_data["k"],
            f1_scores,
            marker=markers[method],
            linewidth=2,
            markersize=4,
            color=colors[method],
            # Removed label for methods - shown in master legend
            alpha=0.8,
        )

        # Mark the best F1 score for this method
        if len(f1_scores) > 0 and f1_scores.max() > 0:
            best_f1_idx = f1_scores.idxmax()
            best_k = method_data.iloc[best_f1_idx]["k"]
            best_f1 = f1_scores.iloc[best_f1_idx]

            ax4.scatter(
                best_k,
                best_f1,
                color=colors[method],
                s=100,
                zorder=5,
                edgecolor="black",
                linewidth=1,
            )

            # Add text annotations for best F1 scores - spread to corners
            corner_positions = {
                "euclidean": (0.02, 0.98),  # Top left
                "triplet": (0.65, 0.98),  # Top right
                "running_mean": (0.02, 0.87),  # Middle left
                "wasserstein": (0.65, 0.87),  # Middle right
                "chi_square": (0.65, 0.75),  # Bottom left
            }
            if method in corner_positions:
                x_pos, y_pos = corner_positions[method]
                ax4.text(
                    x_pos,
                    y_pos,
                    f"{method.replace('_', ' ').title()} - Best F1:\n{best_f1:.3f} @ k={best_k}",
                    transform=ax4.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor=colors[method], alpha=0.3
                    ),
                )

    ax4.set_title(f"F1 Score @ k")
    ax4.set_xlabel("Top-k Frames Retrieved")
    ax4.set_ylabel("F1 Score")
    # Remove legend since no labeled items remain
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(left=0)
    ax4.set_ylim(0, 1.05)

    # Remove top and right spines
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    # Plot 5: Log scale precision/recall
    ax5 = fig.add_subplot(gs[1, 1])
    for method in methods:
        method_data = precision_recall_df[precision_recall_df["method"] == method]
        ax5.semilogx(
            method_data["k"],
            method_data["Precision"],
            # Removed label for methods - shown in master legend
            marker=markers[method],
            linewidth=2,
            markersize=3,
            color=colors[method],
            alpha=0.7,
        )
        ax5.semilogx(
            method_data["k"],
            method_data["Recall"],
            # Removed label for methods - shown in master legend
            marker=markers[method],
            linewidth=2,
            markersize=3,
            color=colors[method],
            alpha=0.7,
            linestyle="--",
        )

    ax5.axvline(
        x=num_positives,
        color="gray",
        linestyle="--",
        label=f"Actual Positives ({num_positives})",
        alpha=0.7,
    )

    # Add legend entries for line style explanation
    ax5.plot([], [], "k-", label="Precision (solid)")
    ax5.plot([], [], "k--", label="Recall (dashed)")

    ax5.set_title(f"Precision/Recall @ k (Log Scale)")
    ax5.set_xlabel("Top-k Frames Retrieved (log scale)")
    ax5.set_ylabel("Score")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1.05)

    # Remove top and right spines
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)

    # Plot 6: Summary metrics comparison
    ax6 = fig.add_subplot(gs[1, 2])
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

        # Get precision and recall at closest threshold of k=num_positives
        kf = num_positives
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
                "Max Precision": method_data["Precision"].max(),
                "Max Precision k": max_prec_k,
                "Max Recall": method_data["Recall"].max(),
                "Max Recall k": max_recall_k,
                "Best F1": f1_scores.max(),
                "Best F1 k": best_f1_k,
                f"Precision@{kf}": prec_at_kf,
                f"Recall@{kf}": recall_at_kf,
            }
        )

    summary_df = pd.DataFrame(summary_data)

    # Create grouped bar plot
    x = np.arange(5)  # 5 metrics to display
    width = 0.15

    metrics = [
        "Max Precision",
        "Max Recall",
        "Best F1",
        f"Precision@{kf}",
        f"Recall@{kf}",
    ]

    for i, method in enumerate(methods):
        method_row = summary_df[summary_df["Method"] == method].iloc[0]
        method_values = [method_row[metric] for metric in metrics]

        # Get corresponding k values
        k_values = [
            method_row["Max Precision k"],
            method_row["Max Recall k"],
            method_row["Best F1 k"],
            kf,
            kf,
        ]

        ax6.bar(
            x + i * width,
            method_values,
            width,
            # Removed label for methods - shown in master legend
            color=colors[method],
            alpha=0.7,
            edgecolor="black",
        )

        # Add value labels on bars
        for j, (value, k_val) in enumerate(zip(method_values, k_values)):
            if value > 0:
                ax6.text(
                    x[j] + i * width,
                    value + 0.01,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                    rotation=90,
                )

    ax6.set_title(f"Summary Metrics")
    ax6.set_ylabel("Score")
    ax6.set_ylim(0, 1.1)
    ax6.grid(True, alpha=0.3, axis="y")
    ax6.set_xticks(x + width * 1.5)
    ax6.set_xticklabels(metrics, rotation=45)
    # Remove legend since methods are shown in master legend

    # Remove top and right spines
    ax6.spines["top"].set_visible(False)
    ax6.spines["right"].set_visible(False)

    # Create master legend at bottom in 2x2 grid
    legend_ax = fig.add_subplot(gs[2, :])
    legend_ax.axis("off")  # Hide axis

    # Create legend handles for methods only
    legend_handles = []
    method_labels = {
        "euclidean": "Euclidean Histogram Distance",
        "triplet": "Frame Triplet Difference",
        "running_mean": "Running Mean",
        "combined": "Combined Score",
        "wasserstein": "Wasserstein Distance",
        "chi_square": "Chi-Square Distance",
    }

    for method in methods:
        if method in colors:
            handle = plt.Line2D(
                [0],
                [0],
                color=colors[method],
                marker=markers[method],
                linewidth=3,
                markersize=8,
                label=method_labels.get(method, method),
            )
            legend_handles.append(handle)

    # Create legend in 2x2 grid layout
    legend = legend_ax.legend(
        handles=legend_handles,
        loc="lower left",
        ncol=3,  # 2 columns for 2x2 grid
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
        title="",
        title_fontsize=14,
    )
    legend.get_title().set_fontweight("bold")

    # Save the comparison plot
    plot_filename = os.path.join(
        output_folder, f"{video_name}_unsupervised_comparison.png"
    )
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()

    if VERBOSE:
        print(f"Unsupervised comparison plots saved to: {plot_filename}")


def get_available_videos(results_folder: str = "./results/hummingbird") -> list:
    """
    Get list of available videos based on CSV files in the results folder.
    Uses euclidean folder as the primary source for video discovery.

    Args:
        results_folder: Base folder containing analysis results

    Returns:
        List of video names (e.g., ['FH102_02', 'FH103_01'])
    """
    videos = set()

    # Check euclidean folder
    euclidean_folder = os.path.join(results_folder, "euclidean")
    if os.path.exists(euclidean_folder):
        for csv_file in os.listdir(euclidean_folder):
            if csv_file.endswith("_euclidean_diff.csv"):
                video_name = csv_file.replace("_euclidean_diff.csv", "")
                videos.add(video_name)

    # Also check ground truth for validation
    gt_path = os.path.join("./data", "cleaned_ground_truth.csv")
    if os.path.exists(gt_path):
        gt_df = pd.read_csv(gt_path)
        gt_videos = set(gt_df["Video"].unique())
        # Only include videos that have both analysis results and ground truth
        videos = videos.intersection(gt_videos)

    return sorted(list(videos))


def process_single_video(
    video_name: str,
    methods_to_run: list,
    gt_folder: str,
    results_folder: str,
    buffer: int,
) -> pd.DataFrame:
    """
    Process a single video for all specified methods.

    Args:
        video_name: Name of the video to process
        methods_to_run: List of methods to run
        gt_folder: Path to the ground truth folder
        results_folder: Path to the results folder
        buffer: Buffer size for precision/recall computation (consider +/- buffer frames around ground truth as match)

    Returns:
        Combined DataFrame with all results, or None if processing failed
    """
    if VERBOSE:
        print(f"\n{'='*60}")
        print(f"Processing video: {video_name}")
        print(f"{'='*60}")

    all_results = []

    # Load ground truth (same for all methods)
    try:
        gt_video, positives, negatives = load_ground_truth(video_name, gt_folder)

        if VERBOSE:
            print(
                f"Found {len(positives)} positive frames and {len(negatives)} negative frames"
            )

        if len(positives) == 0:
            print(f"Warning: No positive frames found in ground truth for {video_name}")
            return None

    except Exception as e:
        print(f"Error loading ground truth for {video_name}: {e}")
        return None

    # Process each method
    for method in methods_to_run:
        if VERBOSE:
            print(f"\n--- Processing {method} ---")

        try:
            # Load processed data
            if VERBOSE:
                print(f"Loading {method} data...")

            df_change, config = load_unsupervised_data(
                video_name, method, results_folder
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
                df_change, positives, k_values, buffer
            )

            # Add metadata columns
            precision_recall_df["video_name"] = video_name
            precision_recall_df["method"] = method
            precision_recall_df["buffer"] = buffer
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
            print(f"Warning: Could not process {method} for {video_name} - {e}")
            continue
        except Exception as e:
            print(f"Error processing {method} for {video_name}: {e}")
            continue

    if not all_results:
        print(f"Error: No methods could be processed successfully for {video_name}")
        return None

    # Combine all results for this video
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

    return combined_df


def main(
    video_name: str,
    method: str,
    buffer: int,
    output_folder: str,
    results_folder: str,
    gt_folder: str,
    plot: bool,
) -> int:

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Determine which methods to run
    if "all" in method:
        methods_to_run = [
            "euclidean",
            "triplet",
            "running_mean",
            # "combined",
            "wasserstein",
            "chi_square",
        ]
    else:
        methods_to_run = method

    # Determine which videos to process
    if video_name:
        # Single video specified
        videos_to_process = [video_name]
        if VERBOSE:
            print(f"Processing single video: {video_name}")
    else:
        # No video specified - process all available videos
        videos_to_process = get_available_videos(results_folder)
        if not videos_to_process:
            print(f"No videos found in {results_folder}")
            print("Make sure the results folder contains the expected subfolders:")
            print("  - euclidean/")
            print("  - triplet_analysis/")
            print("  - running_mean/")
            print("  - wasserstein/")
            print("  - chi_square/")
            return 1

        if VERBOSE:
            print(
                f"Found {len(videos_to_process)} videos to process: {videos_to_process}"
            )

    if VERBOSE:
        print(f"Method(s): {method}")
        print(f"Buffer: {buffer}")
        print(f"Output folder: {output_folder}")

    # Process videos
    all_combined_results = []
    successful_videos = 0
    failed_videos = []

    for video_name in videos_to_process:
        # Process single video
        video_results = process_single_video(
            video_name, methods_to_run, gt_folder, results_folder, buffer
        )

        if video_results is not None:
            # Save individual video results
            output_csv = os.path.join(
                output_folder, f"{video_name}_unsupervised_comparison.csv"
            )
            video_results.to_csv(output_csv, index=False)

            if VERBOSE:
                print(f"Results saved to: {output_csv}")

            # Create plots if requested
            if plot:
                if VERBOSE:
                    print("Creating unsupervised comparison plots...")
                create_precision_recall_plots(video_results, output_folder, video_name)

            # Add to combined results for summary
            all_combined_results.append(video_results)
            successful_videos += 1

            # Print summary for this video
            if VERBOSE:
                print(f"\n--- SUMMARY FOR {video_name} ---")
                for method in video_results["method"].unique():
                    method_data = video_results[
                        video_results["method"] == method
                    ].reset_index(drop=True)
                    f1_scores = (
                        2
                        * (method_data["Precision"] * method_data["Recall"])
                        / (method_data["Precision"] + method_data["Recall"])
                    )
                    f1_scores = f1_scores.fillna(0)

                    if len(f1_scores) > 0 and f1_scores.max() > 0:
                        best_f1_idx = f1_scores.idxmax()
                        best_f1_value = f1_scores.iloc[best_f1_idx]
                        best_k = method_data.iloc[best_f1_idx]["k"]
                        print(f"  {method}: Best F1 = {best_f1_value:.3f} @ k={best_k}")
        else:
            failed_videos.append(video_name)

    # Final summary
    if VERBOSE:
        print(f"\n{'='*80}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(
            f"Successfully processed: {successful_videos}/{len(videos_to_process)} videos"
        )

        if failed_videos:
            print(f"Failed videos: {failed_videos}")

    if all_combined_results:
        # Save combined results across all videos
        if len(videos_to_process) > 1:
            all_results_df = pd.concat(all_combined_results, ignore_index=True)
            combined_output_csv = os.path.join(
                output_folder, "all_videos_unsupervised_comparison.csv"
            )
            all_results_df.to_csv(combined_output_csv, index=False)
            print(f"Combined results for all videos saved to: {combined_output_csv}")

    print(f"Results saved to: {output_folder}")

    return 0 if successful_videos > 0 else 1


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Compare unsupervised approaches for hummingbird detection"
    )
    parser.add_argument(
        "video_name",
        type=str,
        nargs="?",  # Make video_name optional
        help="Name of the video (e.g., FH102_02). If not provided, processes all available videos.",
    )
    parser.add_argument(
        "--method",
        type=str,
        nargs="+",
        choices=[
            "euclidean",
            "triplet",
            "running_mean",
            # "combined",
            "wasserstein",
            "chi_square",
            "all",
        ],
        default=["all"],
        help="Method(s) to compare: euclidean, triplet, running_mean, combined, wasserstein, chi_square, or all (default: all)",
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=1,
        help="Buffer around ground truth frames for matching (default: 1)",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="./test",
        help="Output folder for results and plots (default: ./test)",
    )
    parser.add_argument(
        "--results-folder",
        type=str,
        default="./results/hummingbird",
        help="Folder containing the analysis results (default: ./results/hummingbird)",
    )
    # results/hummingbird/chi_square
    parser.add_argument(
        "--gt-folder",
        type=str,
        default="./data/cleaned_ground_truth.csv",
        help="Folder containing the ground truth data (default: ./data)",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Create and save precision-recall plots"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    VERBOSE = args.verbose

    exit(
        main(
            args.video_name,
            args.method,
            args.buffer,
            args.output_folder,
            args.results_folder,
            args.gt_folder,
            args.plot,
        )
    )
