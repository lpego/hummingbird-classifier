# %%
# Remember to convert videos before running this!

import os, sys
# import cv2

# import shutil
# import pandas as pd
import numpy as np
from shutil import copyfile

from pathlib import Path
# from PIL import Image
import ffmpeg

from joblib import Parallel, delayed

# fixed for random selection, but not too random. first 10 are parsed with another seed.
np.random.seed(42)

# %%
INFER_FULL = True
vid_root = Path(f"/data/shared/raw-video-import/data/RECODED_HummingbirdVideo/")
vid_path = sorted(list(vid_root.glob("*.avi")))[100:]
# vid_path = [v for v in vid_path if "FH112_02" in v.name]
print(vid_path)

out_folder = Path(f"/data/shared/frame-diff-anomaly/data/no_annotation")
# Path(f"/data/shared/frame-prediction/data/")
# video_name = [Path("FH109_02.avi")]  # Out of sample not too bad
# np.random.shuffle(vid_path)

# df = pd.read_csv("/data/shared/raw-video-import/data/Weinstein2018MEE_ground_truth.csv")
# df["truth_bin"] = df.Truth.replace({"Positive": 1, "Negative": 0})

# video_positives = df.loc[df.Truth == "Positive", "Video"].unique()
# video_name = [vid_root / (a + ".avi") for a in video_positives]
# vid_with_something = ["FH301_02"]


# %%
tr_size = 0
for video in vid_path:
    out_path = out_folder / video.stem
    out_path.mkdir(exist_ok=True, parents=True)
    print(video)
    probe = ffmpeg.probe(video)
    time = float(probe["streams"][0]["duration"]) // 2
    width = probe["streams"][0]["width"]
    nb_frames = int(probe["streams"][0]["nb_frames"])

    cmd = f'ffmpeg -i "{str(video)}" "{out_path}/frame_%05d.jpg"'
    os.system(cmd)


# %%
