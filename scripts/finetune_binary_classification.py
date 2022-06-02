# %%
prefix = ""
# %load_ext autoreload
# %autoreload 2

# standard ecosystem
import os, sys, time, copy
import numpy as np
from pathlib import Path
from PIL import Image
import datetime

sys.path.append(f"{prefix}src")

# torch imports
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import random_split as RandomSplit, DataLoader, BatchSampler

# torchvision
from torchvision import models, transforms

from HummingBirdLoader import HummingBirdLoader, Denormalize
from learning_loops import train_model, visualize_model
from utils import read_pretrained_model

from matplotlib import pyplot as plt

hub_dir = Path(f"{prefix}models/").resolve()
torch.hub.set_dir(hub_dir)

print(f"current torch hub directory: {torch.hub.get_dir()}")
# %%
BSIZE = 32
set_type = "balanced_classes_different_locations"  # "balanced_classes_same_locations", "balanced_classes_different_locations"
dir_dict_trn = {
    "negatives": Path(f"{prefix}data/{set_type}/training_set/class_0"),
    "positives": Path(f"{prefix}data/{set_type}/training_set/class_1"),
    "meta_data": Path(f"{prefix}data/positives_verified.csv"),
}

dir_dict_val = {
    "negatives": Path(f"{prefix}data/{set_type}/validation_set/class_0"),
    "positives": Path(f"{prefix}data/{set_type}/validation_set/class_1"),
    "meta_data": Path(f"{prefix}data/positives_verified.csv"),
}

dir_dict_tst = {
    "negatives": Path(f"{prefix}data/{set_type}/test_set/class_0"),
    "positives": Path(f"{prefix}data/{set_type}/test_set/class_1"),
    "meta_data": Path(f"{prefix}data/positives_verified.csv"),
}

# RandomPerspective(distortion_scale=0.6, p=1.0)
# ColorJitter(brightness=.5, hue=None, saturation=None, contrast=None)
# RandomAdjustSharpness(sharpness_factor=2)
# RandomAutocontrast()
# RandomEqualize()
# SIZE_IM = 224 # originao
SIZE_IM = 448  # Better res

augment_tr = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
        # transforms.RandomAdjustSharpness(sharpness_factor=2),
        # transforms.RandomEqualize(),
        # transforms.RandomAutocontrast(),
        transforms.ColorJitter(brightness=0.3, hue=0, saturation=0, contrast=0.2),
        transforms.Resize(
            (SIZE_IM, SIZE_IM), interpolation=Image.BILINEAR
        ),  # AT LEAST 224
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

augment_ts = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(
            (SIZE_IM, SIZE_IM), interpolation=Image.BILINEAR
        ),  # AT LEAST 224
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)
# mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
# %% set up loaders
trn_hummingbirds = HummingBirdLoader(
    dir_dict_trn, learning_set="trn", ls_inds=[], transforms=augment_tr
)

val_hummingbirds = HummingBirdLoader(
    dir_dict_val, learning_set="val", ls_inds=[], transforms=augment_ts
)

tst_hummingbirds = HummingBirdLoader(
    dir_dict_tst, learning_set="tst", ls_inds=[], transforms=augment_ts
)

trn_loader = DataLoader(
    trn_hummingbirds, batch_size=BSIZE, shuffle=True, drop_last=True
)

val_loader = DataLoader(val_hummingbirds, batch_size=BSIZE)
tst_loader = DataLoader(tst_hummingbirds, batch_size=BSIZE)

dataloaders = {"trn": trn_loader, "val": val_loader}

print("number of batches per loaders")
print(len(trn_loader), len(val_loader), len(tst_loader))

# %%
cl, clc = np.unique(trn_hummingbirds.labels, return_counts=True)
class_weights = torch.Tensor(np.sum(clc) / (2 * clc)).float()
# %% set up model for pretraining

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"

# architecture = "VGG"
# architecture = "ResNet50"
# architecture = "mobilenet"
# architecture = "DenseNet161"
architecture = "ResNet18"

append = (
    set_type
    + "_jitter_augmentation_"
    + datetime.datetime.now().strftime("%Y%m%d")
    + f"_{SIZE_IM}px"
)

model_folder = Path(f"{hub_dir}/{architecture}_{append}/")

print(f"folder: {model_folder.resolve()}")

model = read_pretrained_model(architecture, n_class=2)

# Define Loss
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), reduction="mean")

# Alternatively, associate them with a very low learning rate
# 10e-2 is a scaler to the original lr.
# pars = [
#     {"params": model.features.parameters(), "lr": 1e-2},
# {"params": model.classifier.parameters(), "lr": 1},
# ]

model.epochs = 50
model.model_folder = model_folder
model.learning_rate = 1e-5
model.weight_decay = 1e-8

optimizer_ft = optim.AdamW(
    model.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay
)  # , momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer_ft,
    mode="min",
    factor=0.1,
    patience=10,
    threshold=0.0001,
    threshold_mode="rel",
    cooldown=0,
    min_lr=0,
    eps=1e-08,
    verbose=False,
)

# MultiStepLR(optimizer_ft, milestones=[40], gamma=0.1)

# Send to CUDA
if not prefix:
    model = model.to(device)


# %% TRAIN
model_best, track_learning = train_model(
    model,
    dataloaders,
    criterion,
    optimizer_ft,
    exp_lr_scheduler,
    num_epochs=model.epochs,
    device=device,
    model_dir=model.model_folder,
)

print(f"saving learning curves to {model_folder / 'learning_curves.dict'}")
np.save(model_folder / "learning_curves.dict", track_learning)

# %%
