# %%
# %load_ext autoreload
# %autoreload 2

# standard ecosystem
import os, sys, time, copy
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import datetime
import time

from joblib import Parallel, delayed

import ffmpeg
import cv2

prefix = "../"
sys.path.append(f"{prefix}src")

# torch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# torchvision
from torchvision import transforms

from HummingBirdLoader import HummingBirdLoader, Denormalize
from learning_loops import infer_model

from matplotlib import pyplot as plt

hub_dir = Path(f"/data/shared/hummingbird-classifier/models/").resolve()
torch.hub.set_dir(hub_dir)

print(f"current torch hub directory: {torch.hub.get_dir()}")

# %% 0 - prepare model

device = "cuda" if torch.cuda.is_available() else "cpu"
architecture = "ResNet50_added_negatives"
model_folder = Path(f"{hub_dir}/{architecture}/")
save_pos_frames = model_folder / "extracted_video_frames"
save_pos_frames.mkdir(exist_ok=True, parents=True)

model_pars = torch.load(model_folder / "model_pars_best.pt", map_location="cpu",)
model_state = torch.load(model_folder / "model_state_best.pt", map_location="cpu",)

model = model_pars
model.load_state_dict(model_state)
model.to(device)
model.eval()

augment = transforms.Compose(
    [
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224, 224), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

# %% 1 - prepare video and run it through the model

# videos = list(
#     Path("/data/shared/raw-video-import/data/RECODED_HummingbirdVideo/").glob("*.avi")
# )
# videos.sort()

start = time.time()

video_name = "FH112_01"

video = Path(
    f"/data/shared/raw-video-import/data/RECODED_HummingbirdVideo/{video_name}.avi"
)
videos = [video]

model.eval()

SAVE_POSITIVES = False
FLAG = True
for iv, video in enumerate(videos[:1]):
    # tqdm.write(f"{video.name}, {iv+1}/{len(videos)}")
    # print()
    save_frames = save_pos_frames / (str(video.name)[:-4])
    save_frames.mkdir(exist_ok=True, parents=True)

    probe = ffmpeg.probe(video)
    n_frames = int(probe["streams"][0]["nb_frames"])
    framerate = float(
        eval(probe["streams"][0]["avg_frame_rate"])
    )  # * float(eval(probe["streams"][0]["time_base"]))
    duration_s = str(
        datetime.timedelta(seconds=float(probe["streams"][0]["duration"]))
    )[:-4]
    # print(n_frames, length, framerate, duration_s)

    # frame_list = np.arange(15000, 16000, 1)#n_frames)
    frame_list = np.arange(0, n_frames, 5)

    cap = cv2.VideoCapture(str(video))

    df = pd.DataFrame(
        [],
        index=frame_list,
        columns=[
            "frame_number",
            "timestamp_video",
            "predicted_class",
            "prob_0",
            "prob_1",
        ],
    )
    for i, ff in enumerate(
        tqdm(frame_list[:], desc=f"{video.name}, video {iv+1} of {len(videos)}")
        # frame_list[:10]
    ):
        # print(ff)
        cap.set(1, ff)
        _, frame = cap.read()
        frame = frame[:, :, [2, 1, 0]]
        pframe = Image.fromarray(frame.astype("uint8"), "RGB")
        frame = augment(pframe).to(device)

        outputs = model(frame[None, ...])
        proba = nn.Softmax(dim=1)(outputs).detach().squeeze()
        _, preds = torch.max(outputs, 1)

        # print(proba, preds, outputs)

        time_fr = str(datetime.timedelta(seconds=ff * 1 / framerate))
        df.iloc[i, :].loc["timestamp_video"] = time_fr[:-5] + "/" + duration_s
        df.iloc[i, :].loc["frame_number"] = ff
        df.iloc[i, :].loc["predicted_class"] = preds.cpu().numpy().squeeze()
        df.iloc[i, :].loc["prob_0"] = proba[0].cpu().numpy()
        df.iloc[i, :].loc["prob_1"] = proba[1].cpu().numpy()

        if SAVE_POSITIVES:
            if df.iloc[i, :].loc["prob_1"] > 0.5:
                pframe.save(save_frames / (str(ff) + ".jpg"))
                if FLAG:
                    plt.figure()
                    plt.imshow(pframe)
                    FLAG = False
    #  save DF
    df.to_csv(save_frames / "summary.csv")

end = time.time()
elapsed = str(datetime.timedelta(seconds=(end - start)))
print(f"Time for plain video inference: {elapsed}")
# %%

plt.figure(figsize=(15, 5))
plt.plot(df.prob_0)
# plt.xticks(df.index[::100], df.frame_number[::100], rotation=90);
plt.xticks(df.index[::100], df.index[::100], rotation=90)

# %%
if 1:
    # ONCE INFERENCE IS DONE, THIS RETRIEVES FRAMES BASED ON DETECTION PROBABILITIES
    video_name = "FH112_01"

    df_folder = Path(
        f"/data/shared/hummingbird-classifier/models/{architecture}/extracted_video_frames/{video_name}/summary.csv"
    )
    df = pd.read_csv(df_folder)

    video = Path(
        f"/data/shared/raw-video-import/data/RECODED_HummingbirdVideo/{video_name}.avi"
    )
    # df.prob_1.astype(float).plot()
    frame_list = df.iloc[800:900].loc[df.prob_1.astype(float) > 0.7].frame_number
    # )  # (-df[df.predicted_class == 1].prob_1).sort_values().index.astype(int)

    cap = cv2.VideoCapture(str(video))
    plt.figure()
    denorm = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    for ff in frame_list[:]:
        # print(ff)
        cap.set(1, ff)
        _, frame = cap.read()
        frame = frame[:, :, [2, 1, 0]]
        pframe = Image.fromarray(frame.astype("uint8"), "RGB")

        p1 = df[df.frame_number == ff].prob_1.values[0]
        ts = df[df.frame_number == ff].timestamp_video.values[0].split("/")[0]
        # pframe = augment(pframe)
        plt.title(f"{ff}, p_bird = {p1:.2f} @ {ts}")
        # plt.imshow(denorm(pframe).permute(1,2,0))
        plt.imshow(np.array(pframe))

        plt.show()


# %%
if 0:
    # This is slower. Needs to be parallelized somwhere. I suspect frame extraction can be sped up easy 20x but cv2.imwrite() calls do not respond.

    # Fast inference option 2:
    # 	- save frames to temp folder in parallel,
    # 	- build dataloader, infer,
    #  	- build and store df,
    # 	- remove temp folder,
    #   - loop videos (this loop should allow best parallelism until I figure out streaming pytorch dataloaders...)

    # 1 video at a time, GPU parallel over frames.
    def extract_frame(cap, save_fold, vname, fra):
        cap.set(1, fra)
        _, frame = cap.read()
        cv2.imwrite(f"{save_fold}/{vname}_neg_{fra}.jpg", frame)

    def extract_frames_from_video(save_fold, video, frame_list):
        # probe = ffmpeg.probe(video)
        # n_frames = int(probe["streams"][0]["nb_frames"])

        vname = str(video).split("/")[-1][:-4]

        # frame_list = np.arange(n_frames)
        # frame_list = frame_list[::FREQ]

        pool = Parallel(n_jobs=8, verbose=11, backend="threading")

        cap = cv2.VideoCapture(str(video))
        # for ff in frame_list:
        #  print(save_fold)
        # pool(delayed(extract_frame)(cap, save_fold, vname, ff) for ff in frame_list)
        for ff in frame_list:
            extract_frame(cap, save_fold, vname, ff)

        cap.release()

    # %%

    start = time.time()
    video_name = "FH101_02"

    video = Path(
        f"/data/shared/raw-video-import/data/RECODED_HummingbirdVideo/{video_name}.avi"
    )

    probe = ffmpeg.probe(video)
    n_frames = int(probe["streams"][0]["nb_frames"])
    framerate = float(
        eval(probe["streams"][0]["avg_frame_rate"])
    )  # * float(eval(probe["streams"][0]["time_base"]))
    duration_s = str(
        datetime.timedelta(seconds=float(probe["streams"][0]["duration"]))
    )[:-4]
    # print(n_frames, length, framerate, duration_s)
    # frame_list = np.arange(15000, 16000, 1)#n_frames)
    frame_list = np.arange(0, n_frames, 2)
    cap = cv2.VideoCapture(str(video))
    df = pd.DataFrame(
        [],
        index=frame_list,
        columns=[
            "frame_number",
            "timestamp_video",
            "predicted_class",
            "prob_0",
            "prob_1",
        ],
    )

    temp_save_fold = Path(
        f"/data/shared/hummingbird-classifier/models/temporary_inference/{video_name}/"
    )
    temp_save_fold.mkdir(exist_ok=True, parents=True)

    # store frames into temp folder
    extract_frames_from_video(temp_save_fold, video, frame_list)

    dir_dict_inf = {
        "unlabelled": Path(
            f"/data/shared/hummingbird-classifier/models/temporary_inference/{video_name}"
        ),
        "meta_data": Path(),
    }

    augment = transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((224, 224), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    inf_hummingbirds = HummingBirdLoader(
        dir_dict_inf, learning_set="infer", ls_inds=[], transforms=augment
    )
    inf_loader = DataLoader(inf_hummingbirds, shuffle=False, batch_size=4)
    device = "cuda"
    yhat, probs, gt = infer_model(
        model.to(device), inf_loader, criterion=None, device=device
    )

    yhat = np.asarray(yhat).squeeze()
    probs = np.asarray(probs).squeeze()

    df = pd.DataFrame(
        [],
        index=frame_list,
        columns=[
            "frame_number",
            "timestamp_video",
            "predicted_class",
            "prob_0",
            "prob_1",
        ],
    )

    for i, ff in enumerate(
        tqdm(frame_list[:], desc=f"{video.name}")  # , video {iv+1} of {len(videos)}")
        # frame_list[:10]
    ):
        # print(ff)

        time_fr = str(datetime.timedelta(seconds=ff * 1 / framerate))
        df.iloc[i, :].loc["timestamp_video"] = time_fr[:-5] + "/" + duration_s
        df.iloc[i, :].loc["frame_number"] = ff
        df.iloc[i, :].loc["predicted_class"] = yhat[i].squeeze()
        df.iloc[i, :].loc["prob_0"] = probs[i, 0]  # .cpu().numpy()
        df.iloc[i, :].loc["prob_1"] = probs[i, 1]  # .cpu().numpy()

    df.to_csv(save_frames / "summary_2.csv")

    end = time.time()
    elapsed = str(datetime.timedelta(seconds=(end - start)))
    print(f"Time for dataloader video conversion + inference: {elapsed}")
