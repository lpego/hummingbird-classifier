"""
video_frame_diff_triplet.py

Script to read a video using torchcodec, preprocess frames (normalization, histogram matching, blurring, cropping),
and compute the difference between three frames at different time intervals (triplet analysis).
Optimized with JIT compilation for large frames.
"""

import numpy as np
import cv2
from skimage import exposure
from torchcodec.decoders import VideoDecoder
import matplotlib.pyplot as plt
import yaml
from pathlib import Path


# Try to import numba for selective JIT optimizations
try:
    from numba import jit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Create dummy decorator if numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


@jit(nopython=True, cache=True)
def compute_triplet_differences_jit(p1_flat, p2_flat, p3_flat, h, w, c):
    """JIT-optimized triplet difference computation for large arrays."""
    # Compute differences element-wise
    diff_sum = 0.0
    diff_sum_sq = 0.0

    for i in range(h):
        for j in range(w):
            # Sum across color channels for this pixel
            pixel_diff = 0.0
            for k in range(c):
                idx = i * w * c + j * c + k
                d21 = p2_flat[idx] - p1_flat[idx]
                d23 = p2_flat[idx] - p3_flat[idx]
                pixel_diff = pixel_diff + abs(d21) + abs(d23)

            diff_sum = diff_sum + pixel_diff
            diff_sum_sq = diff_sum_sq + pixel_diff * pixel_diff

    # Compute standard deviation
    mean_diff = diff_sum / (h * w)
    variance = (diff_sum_sq / (h * w)) - (mean_diff * mean_diff)
    return (variance**0.5) if variance > 0 else 0.0


@jit(nopython=True, cache=True)
def normalize_frame_jit(frame_flat, total_pixels):
    """JIT-optimized frame normalization."""
    normalized = np.empty(total_pixels, dtype=np.float32)
    for i in range(total_pixels):
        normalized[i] = frame_flat[i] / 255.0
    return normalized


def normalize_frame(frame):
    """Normalize frame to [0, 1] float32 with optional JIT acceleration."""
    if NUMBA_AVAILABLE and frame.size > 10000:
        flat_frame = frame.astype(np.uint8).ravel()
        normalized_flat = normalize_frame_jit(flat_frame, frame.size)
        return normalized_flat.reshape(frame.shape).astype(np.float32)
    else:
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


def preprocess_frame(frame, reference=None, crop_box=None, blur=False):
    """Apply normalization, optional histogram matching, blurring, and cropping."""
    frame = normalize_frame(frame)

    if reference is not None:
        frame = (frame - np.mean(reference)) / (np.std(reference) + 1e-8)

    if blur:
        frame = blur_frame(frame)
    if crop_box is not None:
        frame = crop_frame(frame, crop_box)
    return frame


def compute_frame_difference(frame1, frame2, frame3):
    """Compute the sum of absolute differences between three frames."""
    diff21 = np.abs(frame2 - frame1)
    diff23 = np.abs(frame2 - frame3)
    return diff21, diff23


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


def process_frame_triplets_optimized(
    decoder, frame_skips, crop_box, num_frames, visualize=False, verbose=False
):
    """
    Optimized core function to process frame triplets for difference computation.

    Args:
        decoder: Video decoder instance
        frame_skips: List of frame skip values
        crop_box: Crop box coordinates (x, y, w, h)
        num_frames: Total number of frames
        visualize: Whether to show visualizations
        verbose: Whether to print detailed progress information

    Returns:
        Dictionary mapping frame indices to computed statistics
    """
    from tqdm import tqdm

    # Pre-allocate results dictionary
    results = {}

    # Cache for preprocessed frames to avoid redundant processing
    frame_cache = {}
    cache_size_limit = max(frame_skips) * 2 + 10

    def get_preprocessed_frame(idx):
        """Get preprocessed frame with caching."""
        if idx in frame_cache:
            return frame_cache[idx]

        # Load and preprocess frame
        raw_frame = decoder[idx].permute(1, 2, 0).cpu().numpy()
        pframe = preprocess_frame(raw_frame, crop_box=crop_box)

        # Cache management - keep cache size reasonable
        if len(frame_cache) >= cache_size_limit:
            oldest_key = min(frame_cache.keys())
            del frame_cache[oldest_key]

        frame_cache[idx] = pframe
        return pframe

    def compute_triplet_differences_vectorized(p1, p2, p3):
        """Vectorized computation of triplet differences with optional JIT acceleration."""
        if NUMBA_AVAILABLE and p1.size > 50000:  # ~224x224x3 threshold
            h, w, c = p1.shape
            p1_flat = p1.ravel()
            p2_flat = p2.ravel()
            p3_flat = p3.ravel()
            return compute_triplet_differences_jit(p1_flat, p2_flat, p3_flat, h, w, c)
        else:
            # Original vectorized NumPy implementation for smaller frames
            diff21 = p2 - p1
            diff23 = p2 - p3
            diff_rgb = np.abs(diff21) + np.abs(diff23)
            diff_sum = np.sum(diff_rgb, axis=2)
            return np.std(diff_sum)

    if verbose:
        print(f"Starting triplet analysis with frame cache limit: {cache_size_limit}")
        print(f"JIT optimization available: {NUMBA_AVAILABLE}")

    # Main processing loop with real-time progress
    for idx in tqdm(range(num_frames), desc="Triplet analysis", disable=not verbose):
        frame_results = {"center_idx": idx}

        # Process all frame skips for current center frame
        for fs in frame_skips:
            idx1 = idx - fs
            idx2 = idx
            idx3 = idx + fs

            if idx1 >= 0 and idx3 < num_frames:
                try:
                    # Get preprocessed frames (with caching)
                    p1 = get_preprocessed_frame(idx1)
                    p2 = get_preprocessed_frame(idx2)
                    p3 = get_preprocessed_frame(idx3)

                    # Compute difference with vectorized operations
                    std_val = compute_triplet_differences_vectorized(p1, p2, p3)
                    frame_results[f"std_diff_rgb_{fs}"] = std_val

                    # Optional visualization
                    if visualize and idx % 100 == 0:
                        diff21, diff23 = compute_frame_difference(p1, p2, p3)
                        diff_rgb = np.abs(diff21 + diff23)
                        normalize_frame_vis = lambda x: (x - np.min(x)) / (
                            np.max(x) - np.min(x) + 1e-8
                        )
                        diff_rgb_vis = normalize_frame_vis(diff_rgb)
                        visualize_rgb_difference(
                            p1,
                            p2,
                            p3,
                            diff_rgb=diff_rgb_vis,
                            title1=f"Frame {idx1}",
                            title2=f"Frame {idx2}",
                            title3=f"Frame {idx3}",
                            titled=f"Frame diff (frame_skip={fs})",
                        )

                except Exception as e:
                    if verbose:
                        print(
                            f"Error processing frame triplet {idx1}-{idx2}-{idx3}: {e}"
                        )
                    frame_results[f"std_diff_rgb_{fs}"] = np.nan
            else:
                frame_results[f"std_diff_rgb_{fs}"] = np.nan

        results[idx] = frame_results

    if verbose:
        print(f"Triplet analysis complete. Processed {len(results)} frames.")

    return results


def main(
    video_path,
    frame_skip=1,
    crop_box=None,
    visualize=False,
    verbose=False,
    output_folder=".",
):
    import pandas as pd

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
            "frame_skip": frame_skip if isinstance(frame_skip, list) else [frame_skip],
            "visualize": visualize,
            "extension": Path(video_path).suffix,
        },
        "preprocessing": {"crop_box": list(crop_box) if crop_box else None},
        "frame_difference_analysis": {
            "method": "triplet",
            "frame_skips": frame_skip if isinstance(frame_skip, list) else [frame_skip],
        },
        "output": {
            "csv_file": f"{video_name}_triplet_diff.csv",
            "config_file": f"{video_name}_triplet_config.yaml",
            "output_folder": str(output_path),
        },
    }

    # Save configuration to YAML file
    config_filename = output_path / f"{video_name}_triplet_config.yaml"
    with open(config_filename, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    if verbose:
        print(f"Configuration saved to: {config_filename}")

    decoder = VideoDecoder(video_path)
    num_frames = len(decoder)

    if isinstance(frame_skip, int):
        frame_skips = [frame_skip]
    else:
        frame_skips = list(frame_skip)

    if verbose:
        print(f"Processing {num_frames} frames with frame_skips: {frame_skips}")

    # Initialize results dictionary
    results_dict = {idx: {"center_idx": idx} for idx in range(num_frames)}

    # TRIPLET FRAME ANALYSIS
    if verbose:
        print("Computing triplet frame differences...")
    triplet_results = process_frame_triplets_optimized(
        decoder, frame_skips, crop_box, num_frames, visualize=visualize, verbose=verbose
    )

    # Update results with triplet analysis
    for idx, frame_results in triplet_results.items():
        results_dict[idx].update(frame_results)

    # Convert to DataFrame
    df_std = (
        pd.DataFrame(list(results_dict.values()))
        .sort_values("center_idx")
        .reset_index(drop=True)
    )
    df_std.index.name = "frame_idx"
    if verbose:
        print(f"\nProcessing complete. Results shape: {df_std.shape}")
        print(df_std.head())
    return df_std


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process videos for hummingbird detection using triplet frame differences"
    )
    parser.add_argument(
        "--output-folder",
        "-o",
        type=str,
        default=".",
        help="Output folder for config files and CSV results (default: current directory)",
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
        help="Frame skip value for triplet analysis (default: 1)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with detailed progress information",
    )

    args = parser.parse_args()

    # Set parameters from command line arguments
    crop_box = tuple(args.crop_box)
    frame_skip = args.frame_skip
    visualize = args.visualize
    verbose = args.verbose
    output_folder = args.output_folder

    if verbose:
        print(f"Output folder: {output_folder}")
        print(f"Crop box: {crop_box}")
        print(f"Frame skip: {frame_skip}")

    test_size = (
        False  # Set to True to visualize the first frame and crop box and then exit
    )

    # Get all FH videos in the /data/ directory
    if 0:
        video_dir = "data/insects/"
        video_files = sorted([str(f) for f in Path(video_dir).rglob("PICT7*.mp4")])
    else:
        video_dir = "data/"
        video_files = sorted([str(f) for f in Path(video_dir).rglob("FH*.avi")])

    if verbose:
        print(f"Found {len(video_files)} videos in {video_dir}")

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
            cv2.imshow(window_name, first_frame)
            crop_box = cv2.selectROI(
                window_name, first_frame, fromCenter=False, showCrosshair=True
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

        df_change = main(
            video_path,
            frame_skip=frame_skip,
            crop_box=crop_box,
            visualize=visualize,
            verbose=verbose,
            output_folder=output_folder,
        )

        # Save results to the specified output folder
        output_path = Path(output_folder)
        fname = (
            output_path / f"{video_path.split('/')[-1].split('.')[0]}_triplet_diff.csv"
        )
        df_change.to_csv(fname, index=False)
        if verbose:
            print(f"Results saved to {fname}")
