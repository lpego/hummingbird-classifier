#!/usr/bin/env python3
"""
compute_precision_recall.py

Script to compute precision and recall for a given video using different numbers of retrieval frames.
Based on the logic from the plotting notebooks for frame difference and histogram difference analysis.
By default, computes results for both methods to enable comparison.
"""

import argparse
import pandas as pd
import numpy as np
import yaml

# from pathlib import Path
import os
import matplotlib.pyplot as plt

VERBOSE = False


def load_ground_truth(video_name, gt_folder="./data"):
    """Load ground truth data for the specified video."""
    gt_path = os.path.join(gt_folder, "Weinstein2018MEE_ground_truth.csv")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    gt = pd.read_csv(gt_path)
    gt_video = gt[gt["Video"] == video_name]
    gt_video = gt_video.set_index("Frame", drop=False)

    # Deduplicate the index of ground truth
    gt_video = gt_video[~gt_video.index.duplicated(keep="first")]

    positives = gt_video[gt_video["Truth"].str.lower() == "positive"]
    negatives = gt_video[gt_video["Truth"].str.lower() == "negative"]

    return gt_video, positives, negatives


def load_diff_data(video_name, method="frame_diff", results_folder="."):
    """
    Load processed difference data for the specified video.

    Args:
        video_name: Name of the video (e.g., 'FH102_02')
        method: 'frame_diff' or 'chist_diff'
        results_folder: Folder containing the processed results
    """
    if method == "frame_diff":
        csv_file = os.path.join(results_folder, f"{video_name}_processed_diff.csv")
        config_file = os.path.join(results_folder, f"{video_name}_diff_config.yaml")
    elif method == "chist_diff":
        csv_file = os.path.join(
            results_folder,
            f"{video_name}_processed_chist_diff.csv",
        )
        config_file = os.path.join(results_folder, f"{video_name}_chist_config.yaml")
    else:
        raise ValueError("method must be 'frame_diff' or 'chist_diff'")

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


def compute_aggregated_diff(df_change, method="frame_diff"):
    """Compute aggregated difference score based on the method."""
    if method == "frame_diff":
        # Match the updated notebook's calculation exactly:
        # Use only the running mean column directly (no rate_change)
        running_mean_col = [col for col in df_change.columns if "running_mean" in col]
        if running_mean_col:
            # Use the running mean column directly
            df_change["aggregated_diff"] = df_change[running_mean_col].mean(axis=1)
        else:
            # Fallback: use all std_diff columns if no running mean found
            df_change["aggregated_diff"] = df_change.filter(like="std_diff_").mean(
                axis=1
            )

    elif method == "chist_diff":
        # For histogram difference: use the stdev_magn_diff_chist column directly
        df_change["aggregated_diff"] = df_change["stdev_magn_diff_chist"]

    # 0-1 normalization
    min_val = df_change["aggregated_diff"].min()
    max_val = df_change["aggregated_diff"].max()
    df_change["aggregated_diff"] = (df_change["aggregated_diff"] - min_val) / (
        max_val - min_val
    )

    return df_change


def compute_precision_recall(df_change, positives, k_values, buffer=3):
    """
    Compute precision and recall for different k values.

    Args:
        df_change: DataFrame with aggregated_diff scores
        positives: DataFrame with positive ground truth frames
        k_values: List of k values to compute precision/recall for
        buffer: Buffer around ground truth frames for matching
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


def create_precision_recall_plots(precision_recall_df, output_folder, video_name):
    """
    Create and save precision-recall plots for both methods.

    Args:
        precision_recall_df: DataFrame with precision and recall data for both methods
        output_folder: Folder to save the plots
        video_name: Name of the video
    """

    # Separate data by method
    methods = precision_recall_df["method"].unique()
    colors = {"frame_diff": "blue", "chist_diff": "red"}
    markers = {"frame_diff": "o", "chist_diff": "s"}

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

    # Create comparison plots
    plt.figure(figsize=(16, 10))
    plt.suptitle(
        f"Precision-Recall Comparison for {video_name}",
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
        label=f"Actual Positives ({num_positives})",
        alpha=0.7,
    )

    plt.title(f"Precision @ k")
    plt.xlabel("Top-k Frames Retrieved")
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
        label=f"Actual Positives ({num_positives})",
        alpha=0.7,
    )

    plt.title(f"Recall @ k")
    plt.xlabel("Top-k Frames Retrieved")
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

    plt.title(f"Precision-Recall Curves")
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
    plt.legend()

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
            if method == "frame_diff":
                plt.text(
                    0.02,
                    0.98,
                    f"Frame Diff - Best F1: \n {best_f1:.3f} @ k={best_k}",
                    transform=plt.gca().transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor=colors[method], alpha=0.3
                    ),
                )
            else:  # chist_diff
                plt.text(
                    0.02,
                    0.88,
                    f"Hist Diff - Best F1: \n {best_f1:.3f} @ k={best_k}",
                    transform=plt.gca().transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor=colors[method], alpha=0.3
                    ),
                )

    plt.title(f"F1 Score @ k")
    plt.xlabel("Top-k Frames Retrieved")
    plt.ylabel("F1 Score")
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
        label=f"Actual Positives ({num_positives})",
        alpha=0.7,
    )

    plt.title(f"Precision/Recall @ k (Log Scale)")
    plt.xlabel("Top-k Frames Retrieved (log scale)")
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
    width = 0.35

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
        choices=["frame_diff", "chist_diff", "both"],
        default="both",
        help="Method to use: frame_diff, chist_diff, or both (default: both)",
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
        if args.method == "both":
            args.output = f"{args.video_name}_precision_recall_comparison.csv"
        else:
            args.output = f"{args.video_name}_precision_recall.csv"
    else:
        # Check if the provided output is a directory
        if os.path.isdir(args.output):
            # If it's a directory, append the appropriate filename
            if args.method == "both":
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
    methods_to_run = (
        ["frame_diff", "chist_diff"] if args.method == "both" else [args.method]
    )

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
