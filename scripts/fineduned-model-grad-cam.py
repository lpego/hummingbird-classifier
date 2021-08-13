# %%

%load_ext autoreload
%autoreload 2
import os, sys, time, copy
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

# from tqdm import tqdm
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
from torchvision import transforms, models

from HummingBirdLoader import HummingBirdLoader, Denormalize
from learning_loops import infer_model

from matplotlib import pyplot as plt

######
sys.path.append(f"{prefix}src/pytorch-grad-cam/")

from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
)
from pytorch_grad_cam.utils.image import show_cam_on_image

hub_dir = Path(f"/data/shared/hummingbird-classifier/models/").resolve()
torch.hub.set_dir(hub_dir)

print(f"current torch hub directory: {torch.hub.get_dir()}")

# %% 

video_name = "FH403_01"

df_anno = pd.read_csv(
    "/data/shared/raw-video-import/data/Weinstein2018MEE_ground_truth.csv"
)

df_vi = df_anno[df_anno.Video == video_name]
df_vi.Truth = df_vi.Truth.replace({"Negative": 0, "Positive": 1})
df_vi = df_vi[df_vi.Truth == 1].drop_duplicates()

# %% 0 - prepare model

device = "cuda" if torch.cuda.is_available() else "cpu"
architecture = "DenseNet161_more_negatives"
model_folder = Path(f"{hub_dir}/{architecture}/")
save_pos_frames = model_folder / "extracted_video_frames"
save_pos_frames.mkdir(exist_ok=True, parents=True)

model_pars = torch.load(model_folder / "model_pars_best.pt", map_location="cpu",)
model_state = torch.load(model_folder / "model_state_best.pt", map_location="cpu",)

model = model_pars
model.load_state_dict(model_state)
model.to(device)
# model.eval()

for par in model.features[-1].parameters():
    par.requires_grad = True

model.zero_grad()

augment = transforms.Compose(
    [
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((700, 700), interpolation=Image.BILINEAR),  # AT LEAST 224
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)
# %% 
BSIZE = 16
set_type = "more_negatives"  # "balanced"
dir_dict_trn = {
    "negatives": Path(f"{prefix}data/{set_type}/training_set/class_0"),
    "positives": Path(f"{prefix}data/{set_type}/training_set/class_1"),
    "meta_data": Path(f"{prefix}data/positives_verified.csv"),
}

augment_tr = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((500, 500), interpolation=Image.BILINEAR),  # AT LEAST 224
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

trn_hummingbirds = HummingBirdLoader(
    dir_dict_trn, learning_set="trn", ls_inds=[], transforms=augment_tr
)
trn_loader = DataLoader(
    trn_hummingbirds, batch_size=BSIZE, shuffle=True, drop_last=True
)
cl, clc = np.unique(trn_hummingbirds.labels, return_counts=True)
class_weights = torch.Tensor(np.sum(clc) / (2 * clc)).float()
# %% 
# I supect we need to do one forward pass, and one backward pass to fill gradients.
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)

model.train()  # Set model to training mode
for i, (inputs, labels) in enumerate(trn_loader):
    print(f"\r ({i}/{len(trn_loader)})", end="")
    inputs = inputs.to(device)
    labels = labels.to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward
    # track history if only in train
    with torch.set_grad_enabled(True):
        outputs = model(inputs)
        proba = nn.Softmax(dim=1)(outputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

    # if i == 200: 
        # break
# %% 1 - prepare video and run it through the model

# videos = list(
#     Path("/data/shared/raw-video-import/data/RECODED_HummingbirdVideo/").glob("*.avi")
# )
# videos.sort()
denorm = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

target_category = 1
target_layer = model.features[-1]

start = time.time()

# video_name = "FH403_01"

video = Path(
    f"/data/shared/raw-video-import/data/RECODED_HummingbirdVideo/{video_name}.avi"
)
videos = [video]

# model.eval()

SAVE_POSITIVES = False
FLAG = True

# model_rn = models.resnet50(pretrained=True)
# target_layer = model_rn.layer4[-1]

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
    frame_list = np.arange(987, 995)  # n_frames, 20)

    cap = cv2.VideoCapture(str(video))

    for fi, ff in enumerate(
        frame_list[:]
    ):
        print(f"{fi} / {len(frame_list)}", end="\n")
        cap.set(1, ff)
        _, frame = cap.read()
        frame = frame[:, :, [2, 1, 0]]
        pframe = Image.fromarray(frame.astype("uint8"), "RGB")
        frame = augment(pframe).to(device)
        frame = frame[None, ...]

        # outputs = model(frame)

        # Construct the CAM object once, and then re-use it on many images:
        cam = GradCAM(model=model, target_layer=target_layer, use_cuda=True)

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=frame, target_category=target_category)

        # In this example grayscale_cam has only one image in the batch:
        # grayscale_cam = grayscale_cam[0, :]
        # visualization = show_cam_on_image(frame, grayscale_cam)

        f, a = plt.subplots(1, 2, figsize=(15, 45))
        a[0].imshow(np.transpose(grayscale_cam,(1,2,0)))
        a[1].imshow(denorm(frame.cpu().squeeze()).permute((1,2,0)))
        plt.show()

# %%
