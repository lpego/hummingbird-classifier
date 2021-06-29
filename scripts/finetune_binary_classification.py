# %%
prefix = ""

# standard ecosystem
import os, sys, time, copy
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.append(f"{prefix}src")

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import random_split as RandomSplit, DataLoader, BatchSampler

# torchvision
from torchvision import models, transforms


from HummingBirdLoader import HummingBirdLoader, Denormalize
from learning_loops import train_model, visualize_model

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

hub_dir = Path(f"{prefix}models/").resolve()
torch.hub.set_dir(hub_dir)

print(f"current torch hub directory: {torch.hub.get_dir()}")
# %%
BSIZE = 32

dir_dict_trn = {
    "negatives": Path(f"{prefix}data/training_set/class_0"),
    "positives": Path(f"{prefix}data/training_set/class_1"),
    "meta_data": Path(f"{prefix}data/positives_verified.csv"),
}

dir_dict_val = {
    "negatives": Path(f"{prefix}data/validation_set/class_0"),
    "positives": Path(f"{prefix}data/validation_set/class_1"),
    "meta_data": Path(f"{prefix}data/positives_verified.csv"),
}

dir_dict_tst = {
    "negatives": Path(f"{prefix}data/test_set/class_0"),
    "positives": Path(f"{prefix}data/test_set/class_1"),
    "meta_data": Path(f"{prefix}data/positives_verified.csv"),
}

augment = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224, 224), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

# %% set up loaders
trn_hummingbirds = HummingBirdLoader(
    dir_dict_trn, learning_set="trn", ls_inds=[], transforms=augment
)

val_hummingbirds = HummingBirdLoader(
    dir_dict_val, learning_set="val", ls_inds=[], transforms=augment
)

tst_hummingbirds = HummingBirdLoader(
    dir_dict_tst, learning_set="tst", ls_inds=[], transforms=augment
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
print(cl, clc)

# loss functions
class_weights = torch.Tensor(np.sum(clc) / (2 * clc)).float()
print(class_weights)
# %% set up model for pretraining

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

architecture = "ResNet50"
append = "_v2"
model_folder = Path(f"{hub_dir}/{architecture}{append}/")

if architecture is "VGG":
    model = models.vgg16(pretrained=True)

    in_feat = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(in_features=in_feat, out_features=2, bias=True)

    # Freeze base feature extraction trunk:
    for param in model.features.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        if np.any([a == 2 for a in param.shape]):
            pass
        else:
            param.requires_grad = False

    # pars = []
    # for param in model.classifier.parameters():
    #     if np.any([a == 2 for a in param.shape]):
    #         pars.append({"param": param, "lr": 1})
    #     else:
    #         pars.append({"param": param, "lr": 1e-9})

elif architecture is "ResNet18":

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2, bias=True)

    # Freeze base feature extraction trunk:
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

elif architecture is "ResNet50":

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2, bias=True)

    # Freeze base feature extraction trunk:
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True


# Define Loss
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), reduction="mean")

# Alternatively, associate them with a very low learning rate
# 10e-2 is a scaler to the original lr.
# pars = [
#     {"params": model.features.parameters(), "lr": 1e-2},
# {"params": model.classifier.parameters(), "lr": 1},
# ]

optimizer_ft = optim.Adam(
    model.parameters(), lr=1e-4, weight_decay=1e-4
)  # , momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[10], gamma=0.1)

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
    num_epochs=20,
    device=device,
    model_dir=model_folder,
)

np.save(model_folder / "learning_curves.dict", track_learning)
