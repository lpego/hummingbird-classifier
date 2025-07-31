"""
video_frame_chist_diff.py

Script to read a video using torchcodec, preprocess frames (normalization, blurring, cropping), and compute local color histogram differences between three frames at different time intervals. Also includes visualization of histogram differences.
"""

import numpy as np
import cv2
from torchcodec.decoders import VideoDecoder
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# import os
from pathlib import Path
import yaml
from collections import deque

# Try to import numba for JIT compilation
try:
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba is not available. Using fallback implementations.")

    # Create dummy decorators if numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    prange = range


# def normalize_frame(frame):
#     """Normalize frame to [0, 1] float32."""
#     return frame.astype(np.float32) / 255.0


# ====== PLOT FUNCTIONS ======
def plot_original_triplet(pframe1, pframe2, pframe3):
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    axes[0].imshow(pframe1)
    axes[0].set_title("Frame 1")
    axes[0].axis("off")
    axes[1].imshow(pframe2)
    axes[1].set_title("Frame 2")
    axes[1].axis("off")
    axes[2].imshow(pframe3)
    axes[2].set_title("Frame 3")
    axes[2].axis("off")
    plt.tight_layout()
    plt.show()


def highlight_high_diff_regions(
    pframe2, normalized_diff, patch_size, viz_threshold=0.5
):
    high_diff_indices = np.where(normalized_diff > viz_threshold)[0]
    mask = np.zeros_like(pframe2)

    # Calculate the actual number of patches
    patches_y = pframe2.shape[0] // patch_size[1]  # number of patches vertically
    patches_x = pframe2.shape[1] // patch_size[0]  # number of patches horizontally

    patch_idx = 0
    for y_idx in range(patches_y):
        for x_idx in range(patches_x):
            if patch_idx < len(normalized_diff) and patch_idx in high_diff_indices:
                # Calculate pixel coordinates
                y_start = y_idx * patch_size[1]
                y_end = y_start + patch_size[1]
                x_start = x_idx * patch_size[0]
                x_end = x_start + patch_size[0]

                mask[y_start:y_end, x_start:x_end] = [1, 0, 0]
            patch_idx += 1

    highlighted_frame = pframe2.copy()
    highlighted_frame[mask > 0] = mask[mask > 0]
    return highlighted_frame


def visualize_histogram_difference_patch_image(
    pframe1, pframe2, normalized_diff, patch_size, viz_threshold=0.5
):
    colored_patches = np.zeros_like(pframe1)

    # Calculate the actual number of patches
    patches_y = pframe1.shape[0] // patch_size[1]  # number of patches vertically
    patches_x = pframe1.shape[1] // patch_size[0]  # number of patches horizontally

    patch_idx = 0
    for y_idx in range(patches_y):
        for x_idx in range(patches_x):
            if patch_idx < len(normalized_diff):  # Add bounds check
                # Calculate pixel coordinates
                y_start = y_idx * patch_size[1]
                y_end = y_start + patch_size[1]
                x_start = x_idx * patch_size[0]
                x_end = x_start + patch_size[0]

                color_value = normalized_diff[patch_idx]
                colored_patches[y_start:y_end, x_start:x_end] = [
                    color_value,
                    color_value,
                    color_value,
                ]
                patch_idx += 1

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    axes[0].imshow(pframe2)
    axes[0].set_title("Original Frame 2")
    axes[0].axis("off")
    axes[1].imshow(colored_patches)
    axes[1].set_title("Patches Colored by Histogram Difference")
    axes[1].axis("off")
    framehlight = highlight_high_diff_regions(
        pframe2, normalized_diff, patch_size, viz_threshold=viz_threshold
    )
    axes[2].imshow(framehlight)
    axes[2].set_title("Frame 2 with Histogram Differences")
    axes[2].axis("off")
    plt.tight_layout()
    plt.show()


# ====== END PLOT FUNCTIONS ======


def blur_frame(frame, ksize=5):
    """Apply Gaussian blur to the frame."""
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)


def crop_frame(frame, crop_box):
    """Crop the frame to the given box: (x, y, w, h)."""
    x, y, w, h = crop_box
    return frame[y : y + h, x : x + w]


def preprocess_frame(frame, reference=None, crop_box=None, blur=False):
    """Apply normalization, optional standardization, blurring, and cropping."""
    frame = frame.astype(np.float32) / 255.0
    # normalize_frame(frame)
    if reference is not None:
        standardize_frame = lambda x: (x - np.mean(x)) / (np.std(x) + 1e-8)
        # frame = match_histogram(frame, reference)
        frame = standardize_frame(frame)
    if blur:
        frame = blur_frame(frame)
    if crop_box is not None:
        frame = crop_frame(frame, crop_box)
    return frame


@jit(nopython=True, cache=True, parallel=True)
def compute_patch_histograms_jit_exact(frame_flat, h, w, c, patch_size, bins):
    """JIT function that exactly replicates the original compute_patch_histograms logic with optimizations."""
    patch_h, patch_w = patch_size

    # Calculate number of patches and pre-allocate output
    num_patches_y = (h - patch_h) // patch_h + 1
    num_patches_x = (w - patch_w) // patch_w + 1
    total_patches = num_patches_y * num_patches_x

    # Pre-allocate 2D array for better memory access patterns
    histograms = np.zeros((total_patches, bins * c), dtype=np.float32)

    # Calculate bin width for density normalization
    bin_width = 1.0 / bins
    patch_pixels = patch_h * patch_w
    normalization = patch_pixels * bin_width

    # Use parallel processing for patches
    for patch_idx in prange(total_patches):
        # Convert linear patch index to 2D coordinates
        y_idx = patch_idx // num_patches_x
        x_idx = patch_idx % num_patches_x

        y = y_idx * patch_h
        x = x_idx * patch_w

        # Process all channels for this patch
        for channel in range(c):
            hist_start = channel * bins

            # Process pixels in patch with optimized memory access
            for py in range(patch_h):
                row_start = ((y + py) * w + x) * c + channel
                for px in range(patch_w):
                    pos = row_start + px * c
                    pixel_val = frame_flat[pos]

                    # Bin assignment with bounds checking
                    bin_idx = min(int(pixel_val * bins), bins - 1)
                    histograms[patch_idx, hist_start + bin_idx] += 1.0

            # Normalize density for this channel
            for bin_idx in range(bins):
                histograms[patch_idx, hist_start + bin_idx] /= normalization

    return histograms


@jit(nopython=True, cache=True)
def compute_histogram_differences_jit(hist1, hist2, hist3, distance_metric="euclidean"):
    """
    JIT-compiled function for fast histogram difference computation.

    Parameters:
    -----------
    hist1, hist2, hist3 : numpy.ndarray
        Histogram arrays for three consecutive frames
    distance_metric : str
        Distance metric to use: "euclidean", "wasserstein", or "chi_square"

    Returns:
    --------
    numpy.ndarray
        Combined distance differences between frame pairs
    """
    n_patches = hist1.shape[0]
    n_bins = hist1.shape[1]

    # Ensure consistent float32 data type for all operations
    diff1 = np.zeros(n_patches, dtype=np.float32)
    diff2 = np.zeros(n_patches, dtype=np.float32)

    if distance_metric == "euclidean":
        # Original Euclidean distance (L2 norm)
        for i in range(n_patches):
            sum_sq_1 = 0.0
            sum_sq_2 = 0.0
            for j in range(n_bins):
                diff_val_1 = hist2[i, j] - hist1[i, j]
                diff_val_2 = hist3[i, j] - hist2[i, j]
                sum_sq_1 += diff_val_1 * diff_val_1
                sum_sq_2 += diff_val_2 * diff_val_2
            diff1[i] = np.sqrt(sum_sq_1)
            diff2[i] = np.sqrt(sum_sq_2)

    elif distance_metric == "wasserstein":
        # Wasserstein (Earth Mover's) distance - 1D approximation
        for i in range(n_patches):
            # Compute cumulative distributions
            cum1 = 0.0
            cum2 = 0.0
            cum3 = 0.0
            wasserstein_12 = 0.0
            wasserstein_23 = 0.0

            for j in range(n_bins):
                cum1 += hist1[i, j]
                cum2 += hist2[i, j]
                cum3 += hist3[i, j]

                # Wasserstein distance is the L1 norm of cumulative difference
                wasserstein_12 += abs(cum2 - cum1)
                wasserstein_23 += abs(cum3 - cum2)

            diff1[i] = wasserstein_12
            diff2[i] = wasserstein_23

    elif distance_metric == "chi_square":
        # Chi-square distance
        eps = np.float32(1e-10)

        for i in range(n_patches):
            chi_sq_12 = 0.0
            chi_sq_23 = 0.0

            for j in range(n_bins):
                # Chi-square between hist1 and hist2
                sum_hist_12 = hist1[i, j] + hist2[i, j] + eps
                diff_val_12 = hist2[i, j] - hist1[i, j]
                chi_sq_12 += (diff_val_12 * diff_val_12) / sum_hist_12

                # Chi-square between hist2 and hist3
                sum_hist_23 = hist2[i, j] + hist3[i, j] + eps
                diff_val_23 = hist3[i, j] - hist2[i, j]
                chi_sq_23 += (diff_val_23 * diff_val_23) / sum_hist_23

            diff1[i] = np.sqrt(chi_sq_12)
            diff2[i] = np.sqrt(chi_sq_23)

    else:
        # Default to Euclidean if unknown metric
        for i in range(n_patches):
            sum_sq_1 = 0.0
            sum_sq_2 = 0.0
            for j in range(n_bins):
                diff_val_1 = hist2[i, j] - hist1[i, j]
                diff_val_2 = hist3[i, j] - hist2[i, j]
                sum_sq_1 += diff_val_1 * diff_val_1
                sum_sq_2 += diff_val_2 * diff_val_2
            diff1[i] = np.sqrt(sum_sq_1)
            diff2[i] = np.sqrt(sum_sq_2)

    # Ensure the result is also float32
    result = np.zeros(n_patches, dtype=np.float32)
    for i in range(n_patches):
        result[i] = diff1[i] + diff2[i]

    return result


def compute_patch_histograms(frame, patch_size=(32, 32), bins=16):
    """Compute binned color histograms for local image patches.
    Only use patches fully contained within the frame.
    """
    h, w, _ = frame.shape
    histograms = []
    patch_w, patch_h = patch_size
    for y in range(0, h - patch_h + 1, patch_h):
        for x in range(0, w - patch_w + 1, patch_w):
            patch = frame[y : y + patch_h, x : x + patch_w]
            hist = []
            for channel in range(3):  # for RGB channels
                hist_channel, _ = np.histogram(
                    patch[:, :, channel], bins=bins, range=(0, 1), density=True
                )
                hist.append(hist_channel)
            histograms.append(np.concatenate(hist))
    return np.array(histograms)


def compute_patch_histograms_jit_wrapper(frame, patch_size=(32, 32), bins=16):
    """JIT-optimized wrapper that exactly matches the original function behavior."""
    h, w, c = frame.shape

    if NUMBA_AVAILABLE:
        # Use JIT version that exactly replicates original logic
        frame_flat = frame.ravel()
        histograms_list = compute_patch_histograms_jit_exact(
            frame_flat, h, w, c, patch_size, bins
        )

        return np.array(histograms_list)
    else:
        # Fallback to original function
        return compute_patch_histograms(frame, patch_size, bins)


def main(
    video_path,
    frame_skip=1,
    crop_box=None,
    patch_size=(32, 32),
    bins=8,
    threshold=None,
    visualize=False,
    verbose=False,
    output_folder=".",
    distance_metric="euclidean",
):
    # Extract video name for config file
    video_name = Path(video_path).stem

    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create configuration dictionary
    config = {
        "video_processing": {
            "video_path": str(video_path),
            "video_name": video_name,
            "frame_skip": frame_skip,
            "visualize": visualize,
            "extension": Path(video_path).suffix,
        },
        "preprocessing": {"crop_box": list(crop_box) if crop_box else None},
        "histogram_analysis": {
            "patch_size": {
                "width": patch_size[0],
                "height": patch_size[1],
                "coordinates": list(patch_size),
            },
            "bins": bins,
            "threshold": threshold,
            "distance_metric": distance_metric,
        },
        "output": {
            "csv_file": f"{video_name}_{distance_metric}_diff.csv",
            "config_file": f"{video_name}_{distance_metric}_config.yaml",
            "output_folder": str(output_path),
        },
    }

    # Save configuration to YAML file in the specified output folder
    config_filename = output_path / f"{video_name}_{distance_metric}_config.yaml"
    with open(config_filename, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    if verbose:
        print(f"Configuration saved to: {config_filename}")

    decoder = VideoDecoder(video_path)
    num_frames = len(decoder)

    if threshold is None:
        threshold = bins * 3  # Default threshold based on bins and channels

    if verbose:
        print(
            f"Processing {num_frames} frames with patch_size: {patch_size}, bins: {bins}"
        )
        print(f"Using distance metric: {distance_metric}")
        print(f"Using {'JIT optimizations' if NUMBA_AVAILABLE else 'NumPy fallback'}")

    # Use deque for O(1) append/popleft operations instead of list
    frame_buffer = deque(maxlen=3)
    hist_buffer = deque(maxlen=3)

    # Pre-allocate results array for better memory efficiency
    results = []
    results_capacity = num_frames - 2 * frame_skip

    # Preload first two frames and their histograms using JIT-optimized version
    for i in range(frame_skip * 2):
        frame = decoder[i].permute(1, 2, 0).cpu().numpy()
        pframe = preprocess_frame(
            frame, crop_box=crop_box
        )  # Use SAME preprocessing as original
        hist = compute_patch_histograms_jit_wrapper(
            pframe, patch_size=patch_size, bins=bins
        )  # Use JIT-optimized version
        frame_buffer.append(pframe)
        hist_buffer.append(hist)

    # Main processing loop with all optimizations
    for idx in tqdm(
        range(frame_skip, num_frames - frame_skip),
        desc="Processing frames",
        disable=not verbose,
    ):
        # Only load the next frame if needed
        if idx + frame_skip < num_frames:
            frame = decoder[idx + frame_skip].permute(1, 2, 0).cpu().numpy()
            pframe = preprocess_frame(
                frame, crop_box=crop_box
            )  # Use SAME preprocessing as original
            hist = compute_patch_histograms_jit_wrapper(
                pframe, patch_size=patch_size, bins=bins
            )  # Use JIT-optimized version
            frame_buffer.append(pframe)
            hist_buffer.append(hist)

        # Vectorized histogram difference computation using JIT
        hist1, hist2, hist3 = hist_buffer[0], hist_buffer[1], hist_buffer[2]

        # Use JIT-compiled function for maximum speed with selected distance metric
        if NUMBA_AVAILABLE:
            hist_diff = compute_histogram_differences_jit(
                hist1, hist2, hist3, distance_metric
            )
        else:
            # Fallback to explicit numpy operations (identical to JIT version)
            if distance_metric == "euclidean":
                diff1 = np.sqrt(np.sum((hist2 - hist1) ** 2, axis=1))
                diff2 = np.sqrt(np.sum((hist3 - hist2) ** 2, axis=1))
            elif distance_metric == "wasserstein":
                n_patches = hist1.shape[0]
                diff1 = np.zeros(n_patches)
                diff2 = np.zeros(n_patches)
                for i in range(n_patches):
                    cum1 = np.cumsum(hist1[i])
                    cum2 = np.cumsum(hist2[i])
                    cum3 = np.cumsum(hist3[i])
                    diff1[i] = np.sum(np.abs(cum2 - cum1))
                    diff2[i] = np.sum(np.abs(cum3 - cum2))
            elif distance_metric == "chi_square":
                n_patches = hist1.shape[0]
                diff1 = np.zeros(n_patches)
                diff2 = np.zeros(n_patches)
                eps = 1e-10
                for i in range(n_patches):
                    sum_hist = hist1[i] + hist2[i] + eps
                    chi_sq_12 = np.sum(((hist2[i] - hist1[i]) ** 2) / sum_hist)
                    diff1[i] = np.sqrt(chi_sq_12)
                    sum_hist = hist2[i] + hist3[i] + eps
                    chi_sq_23 = np.sum(((hist3[i] - hist2[i]) ** 2) / sum_hist)
                    diff2[i] = np.sqrt(chi_sq_23)
            else:
                # Default to Euclidean
                diff1 = np.sqrt(np.sum((hist2 - hist1) ** 2, axis=1))
                diff2 = np.sqrt(np.sum((hist3 - hist2) ** 2, axis=1))
            hist_diff = diff1 + diff2

        normalized_diff = np.minimum(hist_diff, threshold) / threshold
        std_dev = np.std(normalized_diff)

        results.append({"center_idx": idx, "stdev_magn_diff_chist": std_dev})

        if visualize:
            pframe1, pframe2, pframe3 = (
                frame_buffer[0],
                frame_buffer[1],
                frame_buffer[2],
            )
            plot_original_triplet(pframe1, pframe2, pframe3)
            visualize_histogram_difference_patch_image(
                pframe1, pframe2, normalized_diff, patch_size
            )

        # deque automatically removes oldest items when maxlen is reached

    # Create DataFrame directly from list of dicts (more efficient)
    df_std = pd.DataFrame(results)
    df_std.index.name = "frame_idx"
    if verbose:
        print(f"Processing complete. Results shape: {df_std.shape}")
        print(df_std.head())
    return df_std


if __name__ == "__main__":
    import argparse

    # Add command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Process videos for hummingbird detection using color histogram differences"
    )
    parser.add_argument(
        "--output-folder",
        "-o",
        type=str,
        default="results/humbs/",
        help="Output folder for config files and CSV results (default: results/humbs/)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization of processing results",
    )

    parser.add_argument(
        "--crop-box",
        nargs=4,
        type=int,
        default=[0, 0, 1280, 700],
        help="Crop box as x y w h (default: 0 0 1280 700)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Frame skip interval for triplet analysis (default: 1)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with detailed progress information",
    )
    parser.add_argument(
        "--distance-metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "wasserstein", "chi_square"],
        help="Distance metric for histogram comparison (default: euclidean)",
    )
    args = parser.parse_args()

    # Create output folder path
    outfolder = Path(args.output_folder)

    crop_box = args.crop_box
    frame_skip = args.frame_skip
    patch_size = (32, 32)
    bins = 16
    visualize = args.visualize
    verbose = args.verbose

    test_size = (
        False  # Set to True to visualize the first frame and crop box and then exit
    )
    # Get all FH videos in the /data/ directory
    if 0:
        video_dir = "data/insects/"
        video_files = sorted([str(f) for f in Path(video_dir).rglob("PICT7*.mp4")])
    elif 0:
        video_dir = "data/external_data/"
        video_files = sorted([str(f) for f in Path(video_dir).rglob("*.mp4")])
    else:
        video_dir = "data/"
        video_files = sorted([str(f) for f in Path(video_dir).rglob("FH*.avi")])
        # video_files = sorted([str(f) for f in Path(video_dir).rglob("ece*.avi")])

    if verbose:
        print(f"Found {len(video_files)} videos in {video_dir}")
        print(f"Output folder: {outfolder}")

    # Loop over the list of videos
    for video_path in video_files[:]:
        if verbose:
            print(f"Processing video: {video_path}")

        if crop_box is None:
            if verbose:
                print("Crop box is not defined. Please manually draw a bounding box.")

            # Load the first frame of the first video to define the crop box
            decoder = VideoDecoder(video_path)
            first_frame = decoder[0].permute(1, 2, 0).cpu().numpy()[..., ::-1]

            # Display the first frame using OpenCV
            window_name = "Draw a bounding box (Press ENTER when done)"
            cv2.imshow(window_name, first_frame)  # Convert RGB to BGR for OpenCV
            crop_box = cv2.selectROI(
                window_name,
                first_frame,
                fromCenter=False,
                showCrosshair=True,
            )
            cv2.destroyAllWindows()

            # Convert crop_box to (x, y, w, h)
            crop_box = (
                int(crop_box[0]),
                int(crop_box[1]),
                int(crop_box[2]),
                int(crop_box[3]),
            )
            if verbose:
                print(f"Crop box defined as: {crop_box}")

        if test_size:
            # Load the first frame of the first video to define the crop box
            decoder = VideoDecoder(video_path)
            first_frame = decoder[0].permute(1, 2, 0).cpu().numpy()
            # print(f"First frame shape: {first_frame.shape}")
            # Convert the frame to uint8 for OpenCV display
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

        df_hist_diff = main(
            video_path,
            frame_skip=frame_skip,
            crop_box=crop_box,
            patch_size=patch_size,
            bins=bins,
            visualize=visualize,
            verbose=verbose,
            output_folder=str(outfolder),
            distance_metric=args.distance_metric,
        )

        fname = (
            outfolder
            / f"{video_path.split('/')[-1].split('.')[0]}_{args.distance_metric}_diff.csv"
        )
        df_hist_diff.to_csv(
            fname,
            index=False,
        )
        if verbose:
            print(f"Results saved to {fname}")
