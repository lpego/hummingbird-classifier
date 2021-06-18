# %%
import subprocess, os
from pathlib import Path


# %%

# make new video folder, which contains only recoded AVI
original_video_path = Path("/data/shared/raw-video-import/data/HummingbirdVideo/")
recoded_video_path = Path(
    "/data/shared/raw-video-import/data/RECODED_HummingbirdVideo/"
)
recoded_video_path.mkdir(exist_ok=True, parents=True)

avi_video_list = list(original_video_path.glob("**/*.AVI"))

# %% Recode using ffmpeg coded

for v, video in enumerate(avi_video_list):

    print(f"video {v+1}/{len(avi_video_list)}, in: {video.name}")

    video_out = recoded_video_path / (video.name.split(".")[0] + ".avi")
    cmd = [
        "ffmpeg",
        "-i",
        "-v",
        "0",
        str(video),
        "-vcodec",
        "copy",
        "-acodec",
        "copy",
        str(video_out),
    ]
    subprocess.run(cmd, check=True)
# %%
