# %%
%load_ext autoreload
%autoreload 2

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
from learning_loops import train_model, visualize_model, infer_model


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
class_weights = (
    torch.tensor(
        np.sum(clc)
        / (2 * clc)
    )
    .float()
)
print(class_weights)
# %% set up model for inference

device = "cuda" if torch.cuda.is_available() else "cpu"
model_dir = Path("/data/users/michele/hummingbird-classifier/models/vgg_cp")

model_pars = torch.load(model_dir / "model_pars.pt", map_location="cpu",)
model_state = torch.load(model_dir / "model_state.pt", map_location="cpu",)

model = model_pars
model.load_state_dict(model_state)
model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device),reduce="mean")
yhat, probs, gt = infer_model(model, tst_loader, criterion, device=device)
# %% 
model.to("cpu")
# %%
denorm = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

visualize_model(model, tst_loader, device=device, num_images=100, denormalize=denorm, save_folder=Path("../models/vgg_cp_2/somefigs/"))
# %% 

from sklearn.metrics import classification_report

print(classification_report(gt, yhat))
# %%

plt.figure()
plt.plot(probs[:100,:])

# %%
