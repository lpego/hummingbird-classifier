"""
video_frame_diff.py

Script to read a video using torchcodec, preprocess frames (normalization, histogram matching, blurring, cropping), and compute the difference between three frames at different time intervals. Also includes RGB visualization of the difference image.
"""

import numpy as np
import cv2
from skimage import exposure
from torchcodec.decoders import VideoDecoder
import matplotlib.pyplot as plt
import yaml
from pathlib import Path


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
        # frame = match_histogram(frame, reference)
        # standardize on statistics of a reference. For self-standardization, use as reference the frame itself
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


# def compute_avg_prev_frames_diff(frames, N):
#     """
#     For each frame (starting from the first), compute the average of the previous N frames (or all available if fewer than N),
#     and return the difference between the current frame and this average.
#     Returns a list of (frame_idx, std_diff) tuples.
#     """
#     diffs = []
#     for idx in range(len(frames)):
#         if idx == 0:
#             diffs.append((idx, np.nan))
#             continue
#         start = max(0, idx - N)
#         avg_frame = np.mean(frames[start:idx], axis=0)
#         diff = np.abs(frames[idx] - avg_frame)
#         std_val = np.std(np.linalg.norm(diff, axis=2))
#         diffs.append((idx, std_val))
#     return diffs


def main(
    video_path,
    frame_skip=1,
    crop_box=None,
    visualize=False,
    use_running_mean=True,
    running_mean_N=20,
):
    import pandas as pd
    from tqdm import tqdm
    from collections import deque

    # Extract video name for config file
    video_name = Path(video_path).stem

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
        "output": {
            "csv_file": f"{video_name}_processed_diff.csv",
            "config_file": f"{video_name}_config.yaml",
        },
    }

    # Save configuration to YAML file
    config_filename = f"{video_name}_diff_config.yaml"
    with open(config_filename, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    if VERBOSE:
        print(f"Configuration saved to: {config_filename}")

    reducer = np.sum

    decoder = VideoDecoder(video_path)
    num_frames = len(decoder)

    if isinstance(frame_skip, int):
        frame_skips = [frame_skip]
    else:
        frame_skips = list(frame_skip)

    # min_skip = min(frame_skips)
    # max_skip = max(frame_skips)
    all_idxs = list(range(num_frames))

    std_dict = {idx: {"center_idx": idx} for idx in all_idxs}

    # For running mean: keep a buffer of last N preprocessed frames
    if use_running_mean:
        running_buffer = deque(maxlen=running_mean_N)

    num_frames = len(decoder)
    # Per-frame difference (triplet) computation: process on the fly
    for idx in tqdm(range(num_frames), desc="Processing frames"):
        frame = decoder[idx].permute(1, 2, 0).cpu().numpy()
        pframe = preprocess_frame(frame, crop_box=crop_box)

        # Running mean difference computation
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
                # frame2 = frame  # already loaded
                frame3 = decoder[idx3].permute(1, 2, 0).cpu().numpy()
                p1 = preprocess_frame(frame1, crop_box=crop_box)
                p2 = pframe
                p3 = preprocess_frame(frame3, crop_box=crop_box)
                diff21, diff23 = compute_frame_difference(p1, p2, p3)
                diff_rgb = np.abs(diff21 + diff23)
                std_val = np.std(reducer(diff_rgb, axis=2))
                std_dict[idx2][f"std_diff_rgb_{fs}"] = std_val
                if visualize:
                    normalize_frame = lambda x: (x - np.min(x)) / (
                        np.max(x) - np.min(x)
                    )
                    diff_rgb_vis = normalize_frame(diff_rgb)
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
            else:
                std_dict[idx2][f"std_diff_rgb_{fs}"] = np.nan

    df_std = (
        pd.DataFrame(list(std_dict.values()))
        .sort_values("center_idx")
        .reset_index(drop=True)
    )
    df_std.index.name = "frame_idx"

    if VERBOSE:
        print(df_std.head())

    return df_std


if __name__ == "__main__":

    crop_box = (0, 0, 1280, 700)
    frame_skip = 1
    visualize = False
    use_running_mean = True
    running_mean_N = 20

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

    if VERBOSE:
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

        df_change = main(
            video_path,
            frame_skip=frame_skip,
            crop_box=crop_box,
            visualize=visualize,
            use_running_mean=use_running_mean,
            running_mean_N=running_mean_N,
        )

        # Use the same naming convention as video_frame_chist_diff.py
        fname = f"./{video_path.split('/')[-1].split('.')[0]}_processed_diff.csv"
        df_change.to_csv(fname, index=False)
        print(f"Results saved to {fname}")
