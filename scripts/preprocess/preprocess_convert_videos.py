import subprocess
import argparse

from pathlib import Path

# from src.utils import cfg_to_arguments


def convert_videos(video_path, output_path):
    """
    Convert all AVI videos to MP4 using ffmpeg, so that they can be read by OpenCV efficiently.
    """
    video_list = list(video_path.glob("**/*.AVI"))
    video_list = sorted(video_list)[:1]

    for v, video in enumerate(video_list):
        print(f"video {v+1}/{len(video_list)}, in: {video}")

        video_out = output_path / (video.name.split(".")[0] + ".mp4")
        cmd = [
            "ffmpeg",
            "-i",
            str(video),
            "-vcodec",
            "copy",
            "-acodec",
            "copy",
            str(video_out),
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input-video-path", type=str, help="Path to input video folder")
    args.add_argument(
        "--output-video-path", type=str, help="Path to output video folder"
    )
    # args.add_argument("--config", "-c", type=str, help="Path to config file")
    args = args.parse_args()

    # make new video folder, which contains only recoded AVI
    original_video_path = Path(
        args.input_video_path
    )  # Path("/data/shared/raw-video-import/data/AnnotatedVideos/")
    recoded_video_path = Path(
        args.output_video_path
    )  # Path("/data/shared/raw-video-import/data/RECODED_AnnotatedVideos/")
    recoded_video_path.mkdir(exist_ok=True, parents=True)

    # config = cfg_to_arguments(args.config)
    convert_videos(original_video_path, recoded_video_path)


# %% Recode using ffmpeg coded

# for v, video in enumerate(avi_video_list):
#     print(f"video {v+1}/{len(avi_video_list)}, in: {video}")

#     video_out = recoded_video_path / (video.name.split(".")[0] + ".avi")
#     cmd = [
#         "ffmpeg",
#         "-i",
#         str(video),
#         "-vcodec",
#         "copy",
#         "-acodec",
#         "copy",
#         str(video_out),
#     ]
#     subprocess.run(cmd, check=True)
