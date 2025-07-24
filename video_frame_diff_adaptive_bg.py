"""
video_frame_diff_adaptive_bg.py

Script to read a video using torchcodec, preprocess frames (normalization, histogram matching, blurring, cropping), and compute the difference using adaptive background subtraction methods. Also includes RGB visualization of the difference image.
"""

import numpy as np
import cv2
from skimage import exposure
from torchcodec.decoders import VideoDecoder
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from collections import deque
from typing import Optional, Tuple, Dict, Any


class AdaptiveBackgroundSubtractor:
    """
    Adaptive background subtraction using multiple algorithms.
    """

    def __init__(self, method="mog2", **kwargs):
        """
        Initialize background subtractor.

        Args:
            method: 'mog2', 'gmm', 'running_avg', 'median'
            **kwargs: method-specific parameters
        """
        self.method = method
        self.kwargs = kwargs
        self.background_model = None
        self.frame_count = 0
        self.history = deque(maxlen=kwargs.get("history_length", 200))

        if method == "mog2":
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=kwargs.get("history", 500),
                varThreshold=kwargs.get("varThreshold", 16),
                detectShadows=kwargs.get("detectShadows", True),
            )
        elif method == "knn":
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                history=kwargs.get("history", 500),
                dist2Threshold=kwargs.get("dist2Threshold", 400.0),
                detectShadows=kwargs.get("detectShadows", True),
            )
        elif method == "running_avg":
            self.learning_rate = kwargs.get("learning_rate", 0.01)
        elif method == "median":
            self.update_frequency = kwargs.get("update_frequency", 10)
        elif method == "gmm":
            from sklearn.mixture import GaussianMixture

            self.n_components = kwargs.get("n_components", 3)
            self.pixel_models = {}

    def update_background(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update background model and return foreground mask and background estimate.

        Args:
            frame: Input frame (H, W, C) normalized to [0, 1]

        Returns:
            Tuple of (foreground_mask, background_estimate)
        """
        self.frame_count += 1

        if self.method in ["mog2", "knn"]:
            return self._update_opencv_method(frame)
        elif self.method == "running_avg":
            return self._update_running_average(frame)
        elif self.method == "median":
            return self._update_median_background(frame)
        elif self.method == "gmm":
            return self._update_gmm_background(frame)
        else:
            raise ValueError(f"Unknown background subtraction method: {self.method}")

    def _update_opencv_method(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update using OpenCV background subtractors (MOG2/KNN)."""
        # Convert to uint8 for OpenCV
        frame_uint8 = (frame * 255).astype(np.uint8)

        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame_uint8)

        # Get background image
        bg_image = self.bg_subtractor.getBackgroundImage()
        if bg_image is not None:
            bg_image = bg_image.astype(np.float32) / 255.0
        else:
            # Fallback to current frame if background not ready
            bg_image = frame.copy()

        # Convert mask to float32 and normalize
        fg_mask = fg_mask.astype(np.float32) / 255.0

        return fg_mask, bg_image

    def _update_running_average(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Update using running average background model."""
        if self.background_model is None:
            self.background_model = frame.copy()
            fg_mask = np.zeros(frame.shape[:2], dtype=np.float32)
        else:
            # Update background model
            cv2.accumulateWeighted(frame, self.background_model, self.learning_rate)

            # Compute foreground mask
            diff = np.abs(frame - self.background_model)
            fg_mask = np.mean(diff, axis=2)

            # Threshold the difference
            threshold = np.std(fg_mask) * 2 + np.mean(fg_mask)
            fg_mask = (fg_mask > threshold).astype(np.float32)

        return fg_mask, self.background_model.copy()

    def _update_median_background(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Update using median background model."""
        self.history.append(frame.copy())

        if len(self.history) < 10:
            # Not enough history, return zero mask
            fg_mask = np.zeros(frame.shape[:2], dtype=np.float32)
            bg_image = frame.copy()
        else:
            # Update background every N frames
            if self.frame_count % self.update_frequency == 0:
                history_array = np.array(list(self.history))
                self.background_model = np.median(history_array, axis=0)

            if self.background_model is not None:
                # Compute foreground mask
                diff = np.abs(frame - self.background_model)
                fg_mask = np.mean(diff, axis=2)

                # Adaptive threshold
                threshold = np.percentile(fg_mask, 95)
                fg_mask = (fg_mask > threshold).astype(np.float32)
                bg_image = self.background_model.copy()
            else:
                fg_mask = np.zeros(frame.shape[:2], dtype=np.float32)
                bg_image = frame.copy()

        return fg_mask, bg_image

    def _update_gmm_background(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Update using Gaussian Mixture Model for each pixel."""
        # This is a simplified implementation
        # In practice, you'd want to use a more efficient implementation
        if self.frame_count < 50:
            # Collect initial samples
            self.history.append(frame.copy())
            fg_mask = np.zeros(frame.shape[:2], dtype=np.float32)
            bg_image = frame.copy()
        else:
            if self.frame_count == 50:
                # Initialize GMM models
                self._initialize_gmm_models()

            fg_mask, bg_image = self._apply_gmm_subtraction(frame)

        return fg_mask, bg_image

    def _initialize_gmm_models(self):
        """Initialize GMM models for each pixel (simplified version)."""
        # This is a placeholder for a more sophisticated implementation
        history_array = np.array(list(self.history))
        self.background_model = np.mean(history_array, axis=0)

    def _apply_gmm_subtraction(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply GMM-based background subtraction."""
        # Simplified implementation using statistical measures
        diff = np.abs(frame - self.background_model)
        fg_mask = np.mean(diff, axis=2)

        # Adaptive threshold based on local statistics
        threshold = np.std(fg_mask) * 2.5 + np.mean(fg_mask)
        fg_mask = (fg_mask > threshold).astype(np.float32)

        # Update background model slowly
        alpha = 0.005
        self.background_model = (1 - alpha) * self.background_model + alpha * frame

        return fg_mask, self.background_model.copy()


def normalize_frame(frame):
    """Normalize frame to [0, 1] float32."""
    return frame.astype(np.float32) / 255.0


def match_histogram(source, reference):
    """Match the histogram of the source frame to the reference frame."""
    matched = exposure.match_histograms(source, reference, channel_axis=-1)
    return matched


def blur_frame(frame, ksize=5):
    """Apply Gaussian blur to the frame."""
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)


def crop_frame(frame, crop_box):
    """Crop the frame to the given box: (x, y, w, h)."""
    x, y, w, h = crop_box
    return frame[y : y + h, x : x + w]


def preprocess_frame(frame, reference=None, crop_box=None, blur=True):
    """Apply normalization, optional histogram matching, blurring, and cropping."""
    frame = normalize_frame(frame)

    if reference is not None:
        # standardize_frame = lambda x: (x - np.mean(x)) / (np.std(x) + 1e-8)
        # frame = match_histogram(frame, reference)
        # standardize on statistics of a reference. For self-standardization, use as reference the frame itself
        frame = (frame - np.mean(reference)) / (np.std(reference) + 1e-8)

    if blur:
        frame = blur_frame(frame)
    if crop_box is not None:
        frame = crop_frame(frame, crop_box)
    return frame


def compute_adaptive_background_difference(
    bg_subtractor: AdaptiveBackgroundSubtractor, frame: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Compute adaptive background difference and extract features.

    Args:
        bg_subtractor: Adaptive background subtractor instance
        frame: Preprocessed frame

    Returns:
        Tuple of (foreground_mask, background_estimate, features_dict)
    """
    fg_mask, bg_estimate = bg_subtractor.update_background(frame)

    # Extract multiple features from the foreground mask
    features = {
        "fg_area_ratio": np.mean(fg_mask),
        "fg_std": np.std(fg_mask),
        "fg_max": np.max(fg_mask),
        "fg_entropy": _compute_entropy(fg_mask),
        "connected_components": _count_connected_components(fg_mask),
        "largest_component_area": _largest_component_area(fg_mask),
    }

    return fg_mask, bg_estimate, features


def _compute_entropy(mask: np.ndarray, bins: int = 256) -> float:
    """Compute entropy of the mask."""
    hist, _ = np.histogram(mask, bins=bins, range=(0, 1))
    hist = hist + 1e-10  # Avoid log(0)
    prob = hist / hist.sum()
    entropy = -np.sum(prob * np.log2(prob))
    return entropy


def _count_connected_components(mask: np.ndarray, threshold: float = 0.5) -> int:
    """Count connected components in binary mask."""
    binary_mask = (mask > threshold).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(binary_mask)
    return num_labels - 1  # Subtract background component


def _largest_component_area(mask: np.ndarray, threshold: float = 0.5) -> float:
    """Get the area of the largest connected component."""
    binary_mask = (mask > threshold).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(binary_mask)

    if num_labels <= 1:
        return 0.0

    # Find largest component (excluding background)
    areas = []
    for label in range(1, num_labels):
        area = np.sum(labels == label)
        areas.append(area)

    return max(areas) / mask.size if areas else 0.0


def compute_frame_difference(frame1, frame2, frame3):
    """Compute the sum of absolute differences between three frames."""
    diff21 = np.abs(frame2 - frame1)
    diff23 = np.abs(frame2 - frame3)
    return diff21, diff23


def visualize_adaptive_bg_results(
    frame: np.ndarray,
    fg_mask: np.ndarray,
    bg_estimate: np.ndarray,
    features: Dict[str, float],
    title_prefix: str = "Frame",
):
    """Visualize adaptive background subtraction results."""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(frame)
    plt.title(f"{title_prefix} - Original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(bg_estimate)
    plt.title("Background Estimate")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(fg_mask, cmap="hot")
    plt.title("Foreground Mask")
    plt.axis("off")
    plt.colorbar(shrink=0.8)

    plt.subplot(1, 4, 4)
    # Overlay foreground on original frame
    overlay = frame.copy()
    if len(overlay.shape) == 3:
        # Create colored overlay
        fg_colored = np.zeros_like(overlay)
        fg_colored[:, :, 0] = fg_mask  # Red channel for foreground
        overlay = cv2.addWeighted(overlay, 0.7, fg_colored, 0.3, 0)

    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    # Add feature text
    feature_text = "\n".join([f"{k}: {v:.3f}" for k, v in features.items()])
    plt.figtext(0.02, 0.02, feature_text, fontsize=8, verticalalignment="bottom")

    plt.tight_layout()
    plt.show()


def visualize_rgb_difference(
    frame1,
    frame2,
    frame3,
    diff_rgb,
    title1="Frame 1",
    title2="Frame 2",
    title3="Frame 3",
    titled="Frame diff",
):
    """Visualize the RGB difference between two frames."""
    plt.figure(figsize=(10, 4))
    plt.subplot(2, 3, 1)
    plt.imshow(frame1.astype(np.uint8))
    plt.title(title1)
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(frame2.astype(np.uint8))
    plt.title(title2)
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(frame3.astype(np.uint8))
    plt.title(title3)
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(frame2.astype(np.uint8))
    plt.title("")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(diff_rgb.astype(np.uint8))
    plt.title(titled)
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(frame2.astype(np.uint8))
    plt.imshow(diff_rgb.astype(np.uint8), alpha=0.5)
    plt.title("")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main(
    video_path,
    frame_skip=1,
    crop_box=None,
    visualize=False,
    use_running_mean=True,
    running_mean_N=20,
    bg_method="mog2",
    bg_params=None,
    output_folder=".",
):
    import pandas as pd
    from tqdm import tqdm
    from collections import deque

    # Extract video name for config file
    video_name = Path(video_path).stem

    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Default background subtraction parameters
    if bg_params is None:
        bg_params = {
            "mog2": {"history": 500, "varThreshold": 16, "detectShadows": True},
            "knn": {"history": 500, "dist2Threshold": 400.0, "detectShadows": True},
            "running_avg": {"learning_rate": 0.01},
            "median": {"update_frequency": 10, "history_length": 50},
            "gmm": {"n_components": 3},
        }

    # Create configuration dictionary
    config = {
        "video_processing": {
            "video_path": str(video_path),
            "video_name": video_name,
            "frame_skip": frame_skip if isinstance(frame_skip, list) else [frame_skip],
            "visualize": visualize,
            "extension": Path(video_path).suffix,
        },
        "preprocessing": {"crop_box": list(crop_box) if crop_box else None},
        "frame_difference_analysis": {
            "use_running_mean": use_running_mean,
            "running_mean_N": running_mean_N,
        },
        "adaptive_background": {
            "method": bg_method,
            "parameters": bg_params.get(bg_method, {}),
        },
        "output": {
            "csv_file": f"{video_name}_processed_adaptive_bg.csv",
            "config_file": f"{video_name}_adaptive_bg_config.yaml",
            "output_folder": str(output_path),
        },
    }

    # Save configuration to YAML file in the specified output folder
    config_filename = output_path / f"{video_name}_adaptive_bg_config.yaml"
    with open(config_filename, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"Configuration saved to: {config_filename}")

    # Initialize adaptive background subtractor
    bg_subtractor = AdaptiveBackgroundSubtractor(
        method=bg_method, **bg_params.get(bg_method, {})
    )

    reducer = np.sum

    decoder = VideoDecoder(video_path)
    num_frames = len(decoder)

    if isinstance(frame_skip, int):
        frame_skips = [frame_skip]
    else:
        frame_skips = list(frame_skip)

    all_idxs = list(range(num_frames))
    std_dict = {idx: {"center_idx": idx} for idx in all_idxs}

    # For running mean: keep a buffer of last N preprocessed frames
    if use_running_mean:
        running_buffer = deque(maxlen=running_mean_N)

    # Per-frame processing with adaptive background subtraction
    for idx in tqdm(range(num_frames), desc="Processing frames"):
        frame = decoder[idx].permute(1, 2, 0).cpu().numpy()
        pframe = preprocess_frame(frame, crop_box=crop_box)

        # Adaptive background subtraction
        fg_mask, bg_estimate, features = compute_adaptive_background_difference(
            bg_subtractor, pframe
        )

        # Store adaptive background features
        for feature_name, feature_value in features.items():
            std_dict[idx][f"adaptive_bg_{feature_name}"] = feature_value

        # Store background subtraction specific metrics
        std_dict[idx]["adaptive_bg_method"] = bg_method

        # Running mean difference computation (original method)
        if use_running_mean:
            if len(running_buffer) > 0:
                avg_frame = np.mean(list(running_buffer), axis=0)
                diff = np.abs(pframe - avg_frame)
                std_val = np.std(reducer(diff, axis=2))
                std_dict[idx][f"std_diff_running_mean_{running_mean_N}"] = std_val
            else:
                std_dict[idx][f"std_diff_running_mean_{running_mean_N}"] = np.nan
            running_buffer.append(pframe)

        # Per-frame difference (triplet) computation for all frame_skips
        for fs in frame_skips:
            idx1 = idx - fs
            idx2 = idx
            idx3 = idx + fs
            if idx1 >= 0 and idx3 < num_frames:
                frame1 = decoder[idx1].permute(1, 2, 0).cpu().numpy()
                frame3 = decoder[idx3].permute(1, 2, 0).cpu().numpy()
                p1 = preprocess_frame(frame1, crop_box=crop_box)
                p2 = pframe
                p3 = preprocess_frame(frame3, crop_box=crop_box)
                diff21, diff23 = compute_frame_difference(p1, p2, p3)
                diff_rgb = np.abs(diff21 + diff23)
                std_val = np.std(reducer(diff_rgb, axis=2))
                std_dict[idx2][f"std_diff_rgb_{fs}"] = std_val

                if visualize and idx % 100 == 0:  # Visualize every 100th frame
                    normalize_frame_vis = lambda x: (x - np.min(x)) / (
                        np.max(x) - np.min(x) + 1e-8
                    )
                    diff_rgb_vis = normalize_frame_vis(diff_rgb)

                    # Show traditional difference
                    visualize_rgb_difference(
                        p1 * 255,
                        p2 * 255,
                        p3 * 255,
                        diff_rgb=diff_rgb_vis * 255,
                        title1=f"Frame {idx1}",
                        title2=f"Frame {idx2}",
                        title3=f"Frame {idx3}",
                        titled=f"Traditional Diff (skip={fs})",
                    )

                    # Show adaptive background results
                    visualize_adaptive_bg_results(
                        pframe,
                        fg_mask,
                        bg_estimate,
                        features,
                        title_prefix=f"Frame {idx2}",
                    )
            else:
                std_dict[idx2][f"std_diff_rgb_{fs}"] = np.nan

    df_std = (
        pd.DataFrame(list(std_dict.values()))
        .sort_values("center_idx")
        .reset_index(drop=True)
    )
    df_std.index.name = "frame_idx"
    print(df_std.head())
    return df_std


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Video frame difference analysis with adaptive background subtraction"
    )
    parser.add_argument(
        "--bg-method",
        choices=["mog2", "knn", "running_avg", "median", "gmm"],
        default="mog2",
        help="Background subtraction method",
    )
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument(
        "--crop-box",
        nargs=4,
        type=int,
        default=[0, 0, 1280, 700],
        help="Crop box as x y w h",
    )
    parser.add_argument(
        "--output-folder", "-o", type=str, default=".", help="Output folder for results"
    )

    args = parser.parse_args()

    crop_box = tuple(args.crop_box)
    frame_skip = 1
    visualize = args.visualize
    use_running_mean = True
    running_mean_N = 20
    bg_method = args.bg_method
    output_folder = args.output_folder

    test_size = False

    # Get all FH videos in the /data/ directory
    video_dir = "data/"
    video_files = sorted([str(f) for f in Path(video_dir).rglob("FH102*.avi")])

    print(f"Found {len(video_files)} videos in {video_dir}")
    print(f"Using background subtraction method: {bg_method}")

    # Loop over the list of videos
    for video_path in video_files[:1]:  # Process only first video for testing
        print(f"Processing video: {video_path}")

        if test_size:
            # Load the first frame of the first video to define the crop box
            decoder = VideoDecoder(video_path)
            first_frame = decoder[0].permute(1, 2, 0).cpu().numpy()
            first_frame = (first_frame * 255).astype(np.uint8)

            plt.figure(figsize=(10, 5))
            plt.imshow(first_frame)
            plt.title("First Frame with Crop Box")
            plt.gca().add_patch(
                plt.Rectangle(
                    (crop_box[0], crop_box[1]),
                    crop_box[2],
                    crop_box[3],
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none",
                )
            )
            plt.show()
            exit()

        df_change = main(
            video_path,
            frame_skip=frame_skip,
            crop_box=crop_box,
            visualize=visualize,
            use_running_mean=use_running_mean,
            running_mean_N=running_mean_N,
            bg_method=bg_method,
            output_folder=output_folder,
        )

        # Save results with method-specific filename in the specified output folder
        output_path = Path(output_folder)
        fname = (
            output_path
            / f"{video_path.split('/')[-1].split('.')[0]}_processed_adaptive_bg_{bg_method}.csv"
        )
        df_change.to_csv(fname, index=False)
        print(f"Results saved to {fname}")
