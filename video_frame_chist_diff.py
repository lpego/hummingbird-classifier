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
import os
import cv2
from pathlib import Path
import yaml


def normalize_frame(frame):
    """Normalize frame to [0, 1] float32."""
    return frame.astype(np.float32) / 255.0


def blur_frame(frame, ksize=5):
    """Apply Gaussian blur to the frame."""
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)


def crop_frame(frame, crop_box):
    """Crop the frame to the given box: (x, y, w, h)."""
    x, y, w, h = crop_box
    return frame[y : y + h, x : x + w]


def preprocess_frame(frame, reference=None, crop_box=None, blur=True):
    """Apply normalization, optional standardization, blurring, and cropping."""
    frame = normalize_frame(frame)
    standardize_frame = lambda x: (x - np.mean(x)) / (np.std(x) + 1e-8)
    if reference is not None:
        # frame = match_histogram(frame, reference)
        frame = standardize_frame(frame)
    if blur:
        frame = blur_frame(frame)
    if crop_box is not None:
        frame = crop_frame(frame, crop_box)
    return frame


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


def main(
    video_path,
    frame_skip=1,
    crop_box=None,
    patch_size=(32, 32),
    bins=32,
    threshold=None,
    visualize=False,
):
    # Extract video name for config file
    video_name = Path(video_path).stem

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
                "coordinates": list(patch_size),  # Convert tuple to list
            },
            "bins": bins,
            "threshold": threshold,
        },
        "output": {
            "csv_file": f"{video_name}_processed_chist_diff.csv",
            "config_file": f"{video_name}_config.yaml",
        },
    }

    # Save configuration to YAML file
    config_filename = f"{video_name}_config.yaml"
    with open(config_filename, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"Configuration saved to: {config_filename}")

    decoder = VideoDecoder(video_path)
    num_frames = len(decoder)
    std_dict = {}

    if threshold is None:
        threshold = bins * 3  # Default threshold based on bins and channels

    # Buffer for preprocessed frames and histograms
    frame_buffer = []
    hist_buffer = []

    # Preload first two frames and their histograms
    for i in range(frame_skip * 2):
        frame = decoder[i].permute(1, 2, 0).cpu().numpy()
        pframe = preprocess_frame(frame, crop_box=crop_box)
        hist = compute_patch_histograms(pframe, patch_size=patch_size, bins=bins)
        frame_buffer.append(pframe)
        hist_buffer.append(hist)

    for idx in tqdm(
        range(frame_skip, num_frames - frame_skip), desc="Processing frames"
    ):
        # Only load the next frame if not already in buffer
        if idx + frame_skip < num_frames:
            frame = decoder[idx + frame_skip].permute(1, 2, 0).cpu().numpy()
            pframe = preprocess_frame(frame, crop_box=crop_box)
            hist = compute_patch_histograms(pframe, patch_size=patch_size, bins=bins)
            frame_buffer.append(pframe)
            hist_buffer.append(hist)

        # Use buffer for computation
        hist1 = hist_buffer[0]  # / hist_buffer[0].sum(axis=1, keepdims=True)
        hist2 = hist_buffer[1]  # / hist_buffer[1].sum(axis=1, keepdims=True)
        hist3 = hist_buffer[2]  # / hist_buffer[2].sum(axis=1, keepdims=True)

        pframe1 = frame_buffer[0]
        pframe2 = frame_buffer[1]
        pframe3 = frame_buffer[2]

        hist_diff = np.linalg.norm(hist2 - hist1, axis=1) + np.linalg.norm(
            hist3 - hist2, axis=1
        )
        normalized_diff = np.minimum(hist_diff, threshold) / threshold
        std_dev = np.std(normalized_diff)
        std_dict[idx] = {"center_idx": idx, "stdev_magn_diff_chist": std_dev}

        if visualize:
            plot_original_triplet(pframe1, pframe2, pframe3)
            visualize_histogram_difference_patch_image(
                pframe1, pframe2, normalized_diff, patch_size
            )

        # Remove oldest frame/histogram from buffer
        frame_buffer.pop(0)
        hist_buffer.pop(0)

    df_std = (
        pd.DataFrame.from_dict(std_dict, orient="index")
        .sort_values("center_idx")
        .reset_index(drop=True)
    )
    df_std.index.name = "frame_idx"
    print(df_std.head())
    return df_std


if __name__ == "__main__":

    crop_box = (0, 0, 1280, 700)
    frame_skip = 1
    patch_size = (32, 32)
    bins = 8
    visualize = False

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

    print(f"Found {len(video_files)} videos in {video_dir}")
    # Loop over the list of videos
    for video_path in video_files[:]:
        print(f"Processing video: {video_path}")

        if crop_box is None:
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
        )

        fname = f"./{video_path.split('/')[-1].split('.')[0]}_processed_chist_diff.csv"
        df_hist_diff.to_csv(
            fname,
            index=False,
        )
        print(f"Results saved to {fname}")
