"""
video_frame_diff_running_mean.py

Script to read a video using torchcodec, preprocess frames (normalization, histogram matching, blurring, cropping),
and compute the difference between current frame and running mean background (background subtraction).
Optimized with JIT compilation and incremental averaging for large frames.
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
def normalize_frame_jit(frame_flat, total_pixels):
    """JIT-optimized frame normalization."""
    normalized = np.empty(total_pixels, dtype=np.float32)
    for i in range(total_pixels):
        normalized[i] = frame_flat[i] / 255.0
    return normalized


@jit(nopython=True, cache=True)
def update_running_mean_jit(buffer_sum, old_frame_flat, new_frame_flat, buffer_size):
    """JIT-optimized running mean update using incremental computation."""
    # Update running sum by removing old frame and adding new frame
    for i in range(len(new_frame_flat)):
        buffer_sum[i] = buffer_sum[i] - old_frame_flat[i] + new_frame_flat[i]
    return buffer_sum


@jit(nopython=True, cache=True)
def compute_running_mean_diff_jit_optimized(
    current_frame_flat, buffer_sum, buffer_size, h, w, c
):
    """JIT-optimized running mean difference computation using incremental mean."""
    total_pixels = h * w

    # Compute difference and standard deviation
    diff_sum = 0.0
    diff_sum_sq = 0.0

    for i in range(h):
        for j in range(w):
            # Sum across color channels for this pixel
            pixel_diff = 0.0
            for k in range(c):
                idx = i * w * c + j * c + k
                # Average value from buffer sum
                avg_val = buffer_sum[idx] / float(buffer_size)
                diff_val = abs(current_frame_flat[idx] - avg_val)
                pixel_diff = pixel_diff + diff_val

            diff_sum = diff_sum + pixel_diff
            diff_sum_sq = diff_sum_sq + pixel_diff * pixel_diff

    # Compute standard deviation
    mean_diff = diff_sum / float(total_pixels)
    variance = (diff_sum_sq / float(total_pixels)) - (mean_diff * mean_diff)
    return (variance**0.5) if variance > 0.0 else 0.0


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


def preprocess_frame(frame, reference=None, crop_box=None, blur=True):
    """Apply normalization, optional histogram matching, blurring, and cropping."""
    frame = normalize_frame(frame)

    if reference is not None:
        frame = (frame - np.mean(reference)) / (np.std(reference) + 1e-8)

    if blur:
        frame = blur_frame(frame)
    if crop_box is not None:
        frame = crop_frame(frame, crop_box)
    return frame


def visualize_running_mean_difference(
    current_frame, avg_frame, diff_frame, frame_idx, buffer_size
):
    """Visualize the running mean background subtraction."""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(current_frame.astype(np.uint8))
    plt.title(f"Current Frame {frame_idx}")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(avg_frame.astype(np.uint8))
    plt.title(f"Background (N={buffer_size})")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(diff_frame.astype(np.uint8))
    plt.title("Difference")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(current_frame.astype(np.uint8))
    plt.imshow(diff_frame.astype(np.uint8), alpha=0.5)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def process_running_mean_optimized(
    decoder, num_frames, crop_box, running_mean_N, visualize=False, verbose=False
):
    """
    Highly optimized running mean computation with JIT and incremental averaging.

    Uses circular buffer and incremental mean computation to avoid expensive
    array conversions and mean calculations every frame.
    """
    from collections import deque
    from tqdm import tqdm

    results = {}

    # Initialize buffers
    frame_shape = None
    use_jit_optimization = False

    # Standard deque for small frames or fallback
    running_buffer = deque(maxlen=running_mean_N)

    # JIT optimization buffers for large frames
    circular_buffer = None
    buffer_sum = None
    buffer_pos = 0
    buffer_full = False

    for idx in tqdm(
        range(num_frames),
        desc="Running mean background subtraction",
        disable=not verbose,
    ):
        # Load and preprocess frame
        raw_frame = decoder[idx].permute(1, 2, 0).cpu().numpy()
        pframe = preprocess_frame(raw_frame, crop_box=crop_box)

        # Initialize optimization on first frame
        if frame_shape is None:
            frame_shape = pframe.shape
            h, w, c = frame_shape
            frame_size = h * w * c

            # Use JIT optimization for large frames
            if NUMBA_AVAILABLE and frame_size > 50000:
                use_jit_optimization = True
                circular_buffer = np.zeros(
                    (running_mean_N, frame_size), dtype=np.float64
                )
                buffer_sum = np.zeros(frame_size, dtype=np.float64)
                if verbose:
                    print(f"Using JIT optimization for frames of size {frame_size}")
            else:
                if verbose:
                    print(f"Using NumPy fallback for frames of size {frame_size}")

        # Compute running mean difference
        if len(running_buffer) > 0:
            if use_jit_optimization:
                # JIT-optimized path for large frames
                h, w, c = frame_shape
                current_flat = pframe.ravel().astype(np.float64)

                if buffer_full:
                    # Remove oldest frame from sum and add new frame
                    old_frame_flat = circular_buffer[buffer_pos]
                    buffer_sum = update_running_mean_jit(
                        buffer_sum, old_frame_flat, current_flat, running_mean_N
                    )
                    effective_buffer_size = running_mean_N
                else:
                    # Still filling buffer - just add new frame
                    buffer_sum = buffer_sum + current_flat
                    effective_buffer_size = len(running_buffer)

                # Compute difference using incremental mean
                std_val = compute_running_mean_diff_jit_optimized(
                    current_flat, buffer_sum, effective_buffer_size, h, w, c
                )

                # Update circular buffer
                circular_buffer[buffer_pos] = current_flat.copy()
                buffer_pos = (buffer_pos + 1) % running_mean_N
                if buffer_pos == 0:
                    buffer_full = True

                # Optional visualization for JIT path
                if visualize and idx % 100 == 0:
                    avg_frame = (buffer_sum / effective_buffer_size).reshape(
                        frame_shape
                    )
                    diff = np.abs(pframe - avg_frame)
                    normalize_frame_vis = lambda x: (x - np.min(x)) / (
                        np.max(x) - np.min(x) + 1e-8
                    )
                    diff_vis = normalize_frame_vis(diff)
                    visualize_running_mean_difference(
                        pframe * 255,
                        avg_frame * 255,
                        diff_vis * 255,
                        idx,
                        effective_buffer_size,
                    )

            else:
                # NumPy fallback for smaller frames
                buffer_array = np.array(list(running_buffer))
                avg_frame = np.mean(buffer_array, axis=0)
                diff = np.abs(pframe - avg_frame)
                diff_sum = np.sum(diff, axis=2)
                std_val = np.std(diff_sum)

                # Optional visualization for NumPy path
                if visualize and idx % 100 == 0:
                    normalize_frame_vis = lambda x: (x - np.min(x)) / (
                        np.max(x) - np.min(x) + 1e-8
                    )
                    diff_vis = normalize_frame_vis(diff)
                    visualize_running_mean_difference(
                        pframe * 255,
                        avg_frame * 255,
                        diff_vis * 255,
                        idx,
                        len(running_buffer),
                    )

            results[idx] = {
                "center_idx": idx,
                f"std_diff_running_mean_{running_mean_N}": std_val,
            }
        else:
            results[idx] = {
                "center_idx": idx,
                f"std_diff_running_mean_{running_mean_N}": np.nan,
            }

        # Update standard buffer (for size tracking and fallback)
        running_buffer.append(pframe)

    if verbose:
        print(f"Running mean analysis complete. Processed {len(results)} frames.")

    return results


def main(
    video_path,
    running_mean_N=20,
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
            "visualize": visualize,
            "extension": Path(video_path).suffix,
        },
        "preprocessing": {"crop_box": list(crop_box) if crop_box else None},
        "frame_difference_analysis": {
            "method": "running_mean_background_subtraction",
            "running_mean_N": running_mean_N,
        },
        "output": {
            "csv_file": f"{video_name}_running_mean_diff.csv",
            "config_file": f"{video_name}_running_mean_config.yaml",
            "output_folder": str(output_path),
        },
    }

    # Save configuration to YAML file
    config_filename = output_path / f"{video_name}_running_mean_config.yaml"
    with open(config_filename, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    if verbose:
        print(f"Configuration saved to: {config_filename}")

    decoder = VideoDecoder(video_path)
    num_frames = len(decoder)

    if verbose:
        print(
            f"Processing {num_frames} frames with running mean buffer size: {running_mean_N}"
        )

    # Initialize results dictionary
    results_dict = {idx: {"center_idx": idx} for idx in range(num_frames)}

    # RUNNING MEAN BACKGROUND SUBTRACTION
    if verbose:
        print("Computing running mean background subtraction...")
    running_mean_results = process_running_mean_optimized(
        decoder,
        num_frames,
        crop_box,
        running_mean_N,
        visualize=visualize,
        verbose=verbose,
    )

    # Update results with running mean analysis
    for idx, frame_results in running_mean_results.items():
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
        description="Process videos for hummingbird detection using running mean background subtraction"
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
        "--running-mean-N",
        type=int,
        default=20,
        help="Buffer size for running mean analysis (default: 20)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with detailed progress information",
    )

    args = parser.parse_args()

    # Set parameters from command line arguments
    crop_box = tuple(args.crop_box)
    running_mean_N = args.running_mean_N
    visualize = args.visualize
    verbose = args.verbose
    output_folder = args.output_folder

    if verbose:
        print(f"Output folder: {output_folder}")
        print(f"Crop box: {crop_box}")
        print(f"Running mean buffer size: {running_mean_N}")

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

        if test_size:
            # ...existing code...
            exit()

        df_change = main(
            video_path,
            running_mean_N=running_mean_N,
            crop_box=crop_box,
            visualize=visualize,
            verbose=verbose,
            output_folder=output_folder,
        )

        # Save results to the specified output folder
        output_path = Path(output_folder)
        fname = (
            output_path
            / f"{video_path.split('/')[-1].split('.')[0]}_running_mean_diff.csv"
        )
        df_change.to_csv(fname, index=False)
        if verbose:
            print(f"Results saved to {fname}")
