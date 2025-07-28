#!/usr/bin/env python3
"""
Unsupervised Multi-Criteria Scoring System for Hummingbird Detection.

This script loads CSV outputs from three different analysis approaches:
1. Color histogram differences (from video_frame_diff_colorhist.py)
2. Triplet frame differences (from video_frame_diff_triplet.py)
3. Running mean background subtraction (from video_frame_diff_running_mean.py)

It combines these scores using a linear combination and saves the results
with all individual scores plus the combined score.

Author: Generated for hummingbird-classifier project
"""

import os
import argparse
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")


class UnsupervisedMultiCriteriaScorer:
    """
    Unsupervised multi-criteria scorer that combines outputs from three analysis methods.
    """

    def __init__(
        self,
        results_folder: str = "./results/hummingbird/",
        verbose: bool = True,
    ):
        """
        Initialize the multi-criteria scorer.

        Args:
            results_folder: Path to folder containing analysis results subfolders
            verbose: Whether to print detailed progress information
        """
        self.results_folder = Path(results_folder)
        self.verbose = verbose

        # Define subfolders for each method
        self.colorhist_folder = self.results_folder / "color_histogram"
        self.triplet_folder = self.results_folder / "triplet_analysis"
        self.running_mean_folder = self.results_folder / "running_mean"

        if self.verbose:
            print(f"Initialized scorer with results folder: {self.results_folder}")
            print(f"  - Color histogram: {self.colorhist_folder}")
            print(f"  - Triplet analysis: {self.triplet_folder}")
            print(f"  - Running mean: {self.running_mean_folder}")

    def get_available_videos(self) -> List[str]:
        """Get list of available videos based on CSV files in color histogram folder."""
        videos = []
        if self.colorhist_folder.exists():
            for csv_file in self.colorhist_folder.glob("*_processed_chist_diff.csv"):
                video_name = csv_file.stem.replace("_processed_chist_diff", "")
                videos.append(video_name)
        return sorted(videos)

    def load_colorhist_scores(self, video_name: str) -> Optional[pd.DataFrame]:
        """
        Load color histogram difference scores from CSV file.

        Args:
            video_name: Name of the video (e.g., "FH102_02")

        Returns:
            DataFrame with center_idx as index and stdev_magn_diff_chist column
        """
        csv_file = self.colorhist_folder / f"{video_name}_processed_chist_diff.csv"

        if not csv_file.exists():
            if self.verbose:
                print(f"Color histogram file not found: {csv_file}")
            return None

        df = pd.read_csv(csv_file)

        # Set center_idx as index
        if "center_idx" in df.columns:
            df = df.set_index("center_idx", drop=True)

        # Check for expected column
        expected_col = "stdev_magn_diff_chist"
        if expected_col not in df.columns:
            if self.verbose:
                print(
                    f"Warning: Expected column '{expected_col}' not found in {csv_file}"
                )
                print(f"Available columns: {list(df.columns)}")
            return None

        if self.verbose:
            print(f"Loaded color histogram scores: {len(df)} frames")

        return df

    def load_triplet_scores(self, video_name: str) -> Optional[pd.DataFrame]:
        """
        Load triplet frame difference scores from CSV file.

        Args:
            video_name: Name of the video (e.g., "FH102_02")

        Returns:
            DataFrame with center_idx as index and aggregated triplet score
        """
        csv_file = self.triplet_folder / f"{video_name}_triplet_diff.csv"

        if not csv_file.exists():
            if self.verbose:
                print(f"Triplet analysis file not found: {csv_file}")
            return None

        df = pd.read_csv(csv_file)

        # Set center_idx as index
        if "center_idx" in df.columns:
            df = df.set_index("center_idx", drop=True)

        # Aggregate all std_diff_rgb_{frame_skip} columns
        triplet_cols = [col for col in df.columns if col.startswith("std_diff_rgb_")]

        if not triplet_cols:
            if self.verbose:
                print(f"Warning: No triplet columns found in {csv_file}")
                print(f"Available columns: {list(df.columns)}")
            return None

        # Compute mean across all triplet frame skip values
        df["aggregated_triplet"] = df[triplet_cols].mean(axis=1)

        if self.verbose:
            print(
                f"Loaded triplet scores: {len(df)} frames, aggregated {len(triplet_cols)} columns"
            )

        return df

    def load_running_mean_scores(self, video_name: str) -> Optional[pd.DataFrame]:
        """
        Load running mean background subtraction scores from CSV file.

        Args:
            video_name: Name of the video (e.g., "FH102_02")

        Returns:
            DataFrame with center_idx as index and aggregated running mean score
        """
        csv_file = self.running_mean_folder / f"{video_name}_running_mean_diff.csv"

        if not csv_file.exists():
            if self.verbose:
                print(f"Running mean file not found: {csv_file}")
            return None

        df = pd.read_csv(csv_file)

        # Set center_idx as index
        if "center_idx" in df.columns:
            df = df.set_index("center_idx", drop=True)

        # Aggregate all std_diff_running_mean_{N} columns
        running_mean_cols = [
            col for col in df.columns if col.startswith("std_diff_running_mean_")
        ]

        if not running_mean_cols:
            if self.verbose:
                print(f"Warning: No running mean columns found in {csv_file}")
                print(f"Available columns: {list(df.columns)}")
            return None

        # Compute mean across all running mean buffer sizes
        df["aggregated_running_mean"] = df[running_mean_cols].mean(axis=1)

        if self.verbose:
            print(
                f"Loaded running mean scores: {len(df)} frames, aggregated {len(running_mean_cols)} columns"
            )

        return df

    def normalize_scores(self, scores: pd.Series) -> pd.Series:
        """
        Normalize scores to 0-1 range using min-max normalization.

        Args:
            scores: Series of scores to normalize

        Returns:
            Normalized scores
        """
        min_val = scores.min()
        max_val = scores.max()

        if max_val > min_val:
            return (scores - min_val) / (max_val - min_val)
        else:
            return pd.Series(0.0, index=scores.index)

    def combine_scores(
        self, combined_df: pd.DataFrame, weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Combine the three normalized scores using linear combination.

        Args:
            combined_df: DataFrame with individual normalized scores
            weights: Dictionary of weights for each score type

        Returns:
            DataFrame with combined score added
        """
        if weights is None:
            # Default equal weights
            weights = {
                "colorhist_score_norm": 1 / 3,
                "triplet_score_norm": 1 / 3,
                "running_mean_score_norm": 1 / 3,
            }

        # Ensure weights sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        # Compute combined score
        combined_score = pd.Series(0.0, index=combined_df.index)

        for score_col, weight in weights.items():
            if score_col in combined_df.columns:
                combined_score += weight * combined_df[score_col]
                if self.verbose:
                    print(f"  Added {score_col} with weight {weight:.3f}")

        combined_df["combined_score"] = combined_score

        if self.verbose:
            print(
                f"Combined score range: {combined_score.min():.4f} to {combined_score.max():.4f}"
            )

        return combined_df

    def process_single_video(
        self, video_name: str, weights: Optional[Dict[str, float]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Process a single video and combine all available scores.

        Args:
            video_name: Name of the video
            weights: Dictionary of weights for score combination

        Returns:
            DataFrame with all scores and combined score, or None if no data
        """
        if self.verbose:
            print(f"\nProcessing video: {video_name}")

        # Load scores from all three methods
        colorhist_df = self.load_colorhist_scores(video_name)
        triplet_df = self.load_triplet_scores(video_name)
        running_mean_df = self.load_running_mean_scores(video_name)

        # Check if at least one method has data
        available_methods = []
        if colorhist_df is not None:
            available_methods.append("colorhist")
        if triplet_df is not None:
            available_methods.append("triplet")
        if running_mean_df is not None:
            available_methods.append("running_mean")

        if not available_methods:
            if self.verbose:
                print(f"No data available for video {video_name}")
            return None

        if self.verbose:
            print(f"Available methods: {available_methods}")

        # Collect all frame indices
        all_indices = set()
        if colorhist_df is not None:
            all_indices.update(colorhist_df.index)
        if triplet_df is not None:
            all_indices.update(triplet_df.index)
        if running_mean_df is not None:
            all_indices.update(running_mean_df.index)

        # Create combined DataFrame with all indices
        combined_df = pd.DataFrame(index=sorted(all_indices))
        combined_df.index.name = "center_idx"

        # Add raw scores
        if colorhist_df is not None:
            combined_df["colorhist_score"] = colorhist_df["stdev_magn_diff_chist"]
            combined_df["colorhist_score_norm"] = self.normalize_scores(
                combined_df["colorhist_score"]
            )

        if triplet_df is not None:
            combined_df["triplet_score"] = triplet_df["aggregated_triplet"]
            combined_df["triplet_score_norm"] = self.normalize_scores(
                combined_df["triplet_score"]
            )

        if running_mean_df is not None:
            combined_df["running_mean_score"] = running_mean_df[
                "aggregated_running_mean"
            ]
            combined_df["running_mean_score_norm"] = self.normalize_scores(
                combined_df["running_mean_score"]
            )

        # Fill NaN values with 0 (for methods that don't have data for all frames)
        combined_df = combined_df.fillna(0)

        # Combine scores
        if self.verbose:
            print("Combining scores...")
        combined_df = self.combine_scores(combined_df, weights)

        # Reset index for output (don't add video_name column)
        combined_df = combined_df.reset_index()

        if self.verbose:
            print(f"Final result: {len(combined_df)} frames with combined scores")

        return combined_df

    def save_results(self, results_df: pd.DataFrame, output_file: str):
        """Save combined results to CSV file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        if self.verbose:
            print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Unsupervised multi-criteria scoring for hummingbird detection"
    )
    parser.add_argument(
        "--results-folder",
        type=str,
        default="./results/hummingbird/",
        help="Path to folder containing analysis results subfolders (default: ./results/hummingbird/)",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="./results/hummingbird/combined/",
        help="Output folder for combined scores (default: ./results/hummingbird/combined/)",
    )
    parser.add_argument(
        "--videos",
        type=str,
        nargs="+",
        help="Specific videos to process (if not provided, processes all available)",
    )
    parser.add_argument(
        "--colorhist-weight",
        type=float,
        default=1 / 3,
        help="Weight for color histogram score (default: 1/3)",
    )
    parser.add_argument(
        "--triplet-weight",
        type=float,
        default=1 / 3,
        help="Weight for triplet frame score (default: 1/3)",
    )
    parser.add_argument(
        "--running-mean-weight",
        type=float,
        default=1 / 3,
        help="Weight for running mean score (default: 1/3)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with detailed progress information",
    )

    args = parser.parse_args()

    # Initialize scorer
    scorer = UnsupervisedMultiCriteriaScorer(
        results_folder=args.results_folder,
        verbose=args.verbose,
    )

    # Get videos to process
    if args.videos:
        videos_to_process = args.videos
    else:
        videos_to_process = scorer.get_available_videos()

    if not videos_to_process:
        print("No videos found to process. Check the results folder structure.")
        return

    print(f"Found {len(videos_to_process)} videos to process: {videos_to_process}")

    # Create output directory
    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define weights
    weights = {
        "colorhist_score_norm": args.colorhist_weight,
        "triplet_score_norm": args.triplet_weight,
        "running_mean_score_norm": args.running_mean_weight,
    }

    print(f"\nUsing weights: {weights}")

    # Process each video
    successful_videos = 0

    for video_name in videos_to_process:
        combined_df = scorer.process_single_video(video_name, weights)

        if combined_df is not None:
            # Save individual video results
            output_file = output_dir / f"{video_name}_combined_scores.csv"
            scorer.save_results(combined_df, str(output_file))
            successful_videos += 1
        else:
            print(f"Skipping {video_name} - no data available")

    print(f"\nProcessing complete!")
    print(f"Successfully processed {successful_videos}/{len(videos_to_process)} videos")
    print(f"Results saved to: {output_dir}")

    # Create a summary of the run
    summary = {
        "run_info": {
            "total_videos_found": len(videos_to_process),
            "successfully_processed": successful_videos,
            "results_folder": str(args.results_folder),
            "output_folder": str(args.output_folder),
        },
        "weights_used": weights,
        "processed_videos": videos_to_process[:successful_videos],
    }

    summary_file = output_dir / "processing_summary.yaml"
    with open(summary_file, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, indent=2)

    if args.verbose:
        print(f"Processing summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
