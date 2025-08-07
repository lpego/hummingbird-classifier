import cv2
import time
import os
from pathlib import Path
import argparse
import shutil

"""
This script will be potentially made useless by switching to torchcodec.decoders.VideoReader,
"""


def check_free_space(output_loc, required_space_gb):
    """Check if there is more than required_space_gb of free space in the output directory.

    Parameters:
    -----------
    output_loc: Path to the output directory.
    required_space_gb: Required free space in GB.

    Returns:
    --------
    bool: True if there is enough free space, False otherwise.


    """
    os.makedirs(output_loc, exist_ok=True)
    _, _, free = shutil.disk_usage(output_loc)
    free_gb = free / (1024**3)
    return free_gb > required_space_gb


def video_to_frames(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.

    Parameters:
    ----------
    input_loc: Path to the input video file.
    output_loc: Path to the output directory where frames will be saved.

    """
    # Calculate required space as 20 times the size of the input file
    # this is just an estimate and it may not be accurate
    input_file_size_gb = os.path.getsize(input_loc) / (1024**3)
    required_space_gb = 20 * input_file_size_gb

    # Check if there is enough free space
    if not check_free_space(output_loc, required_space_gb):
        print(
            f"Not enough free space in {output_loc}. Up to {required_space_gb:.2f} GB may be required."
        )
        user_input = input("Do you want to proceed anyway? (yes/no): ")
        if user_input.lower() != "yes":
            return

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_loc):
        os.makedirs(output_loc)

    # Extract the filename without its extension
    video_name = input_loc.stem

    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(str(input_loc))
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print("Number of frames: ", video_length)
    count = 0
    print("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            continue
        # Write the results back to output location.
        cv2.imwrite(str(output_loc / f"{video_name}_frame{count+1:06d}.jpg"), frame)
        count = count + 1
        # If there are no more frames left
        if count > (video_length - 1):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print("Done extracting frames.\n%d frames extracted" % count)
            elapsed_time = time_end - time_start
            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print(
                "It took {:0>2}:{:0>2}:{:05.2f} (hh:mm:ss) for conversion.".format(
                    int(hours), int(minutes), seconds
                )
            )
            break


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Extract frames from a video file.")
    args.add_argument(
        "--input_loc",
        "-i",
        type=Path,
        help="Input video file location",
    )
    args.add_argument(
        "--output_loc",
        "-o",
        type=Path,
        help="Output directory to save the frames",
    )
    args = args.parse_args()

    exit(video_to_frames(args.input_loc, args.output_loc))
