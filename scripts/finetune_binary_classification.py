# %%
prefix = "../"

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

dir_dict = {
    "positives": Path(f"{prefix}data/positive_frames"),
    "negatives": Path(f"{prefix}data/negative_frames"),
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
# %% make learning set inds
n_imgs = len(list(dir_dict["positives"].glob("*.jpg"))) + len(
    list(dir_dict["negatives"].glob("*.jpg"))
)

np.random.seed(42)
rands = np.arange(n_imgs)
np.random.shuffle(rands)
trn_s = int(0.7 * n_imgs)
val_s = int(0 * n_imgs)
tst_s = int(0.3 * n_imgs)
trn_inds = rands[:trn_s].astype(int)
val_inds = rands[trn_s : (trn_s + val_s)].astype(int)
tst_inds = rands[(trn_s + val_s) :].astype(int)

print(f"trn size: {trn_s}, val size {val_s}, tst size {tst_s}")

# %% set up loaders
trn_hummingbirds = HummingBirdLoader(
    dir_dict, learning_set="trn", ls_inds=trn_inds, transforms=augment
)

val_hummingbirds = HummingBirdLoader(
    dir_dict, learning_set="val", ls_inds=val_inds, transforms=augment
)

tst_hummingbirds = HummingBirdLoader(
    dir_dict, learning_set="tst", ls_inds=tst_inds, transforms=augment
)

trn_loader = DataLoader(
    trn_hummingbirds, batch_size=BSIZE, shuffle=True, drop_last=True
)

val_loader = DataLoader(val_hummingbirds, batch_size=BSIZE)
tst_loader = DataLoader(tst_hummingbirds, batch_size=BSIZE)

dataloaders = {"trn": trn_loader, "val": tst_loader}
# %%
cl, clc = np.unique(trn_hummingbirds.labels, return_counts=True)
print(cl, clc)

# loss functions
class_weights = torch.Tensor(np.sum(clc) / (2 * clc)).float()
print(class_weights)
# %% set up model for pretraining

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

architecture = "ResNet"
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

elif architecture is "ResNet":

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2, bias=True)

    # Freeze base feature extraction trunk:
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

# print(model)

# replace last linear layer of the classifier, change from 1000 classes to 2


# Define Loss
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), reduction="mean")


# Alternatively, associate them with a very low learning rate
# 10e-2 is a scaler to the original lr.
# pars = [
#     {"params": model.features.parameters(), "lr": 1e-2},
# {"params": model.classifier.parameters(), "lr": 1},
# ]

optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)  # , momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[3], gamma=0.1)

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
    num_epochs=10,
    device=device,
    model_dir=Path(f"{hub_dir}/resnet18/"),
)

# %%
# visualize_model(model_best, dataloaders, device=device)


# model_pars = torch.load(
#     Path(f"deapsnow_live/models/rnns/plain_cnn/logs/{MODEL_NAME}/model_pars.pt"),
#     map_location="cpu",
# )

# model_state = torch.load(
#     Path(f"deapsnow_live/models/rnns/plain_cnn/logs/{MODEL_NAME}/model_state.pt"),
#     map_location="cpu",
# )

# model = model_pars
# model.load_state_dict(model_state)
# %%
# denorm = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

# c = 0
# for batch in tst_loader:
#     c +=1
#     for x, y in batch:
#         plt.figure()
#         plt.title(f"label: {y}")
#         plt.imshow(denorm(x).permute((1,2,0)))
#     break
# %%

# x, y = dloader[0]
# denorm = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

# plt.figure()
# plt.title(f"label: {y}")
# plt.imshow(denorm(x).permute((1,2,0)))
# %%

# %%
