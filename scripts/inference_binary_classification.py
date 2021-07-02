# %%
%load_ext autoreload
%autoreload 2

# standard ecosystem
import os, sys, time, copy
import numpy as np
from pathlib import Path
from PIL import Image

prefix = "../"
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

hub_dir = Path(f"/data/shared/hummingbird-classifier/models/").resolve()
torch.hub.set_dir(hub_dir)

print(f"current torch hub directory: {torch.hub.get_dir()}")
# %% # %%
BSIZE = 32
set_type = "more_negatives" # "balanced" or "more_negatives" 
dir_dict_trn = {
    "negatives": Path(f"{prefix}data/training_set/{set_type}/class_0"),
    "positives": Path(f"{prefix}data/training_set/{set_type}/class_1"),
    "meta_data": Path(f"{prefix}data/positives_verified.csv"),
}

dir_dict_val = {
    "negatives": Path(f"{prefix}data/validation_set/{set_type}/class_0"),
    "positives": Path(f"{prefix}data/validation_set/{set_type}/class_1"),
    "meta_data": Path(f"{prefix}data/positives_verified.csv"),
}

dir_dict_tst = {
    "negatives": Path(f"{prefix}data/test_set/{set_type}/class_0"),
    "positives": Path(f"{prefix}data/test_set/{set_type}/class_1"),
    "meta_data": Path(f"{prefix}data/positives_verified.csv"),
}

augment = transforms.Compose(
    [
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224, 224), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

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

val_loader = DataLoader(val_hummingbirds, shuffle=False, batch_size=BSIZE)
tst_loader = DataLoader(tst_hummingbirds, shuffle=False, batch_size=BSIZE)

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

architecture = f"DenseNet161_{set_type}"
model_folder = Path(f"{hub_dir}/{architecture}/")

model_pars = torch.load(model_folder / "model_pars_best.pt", map_location="cpu",)
model_state = torch.load(model_folder / "model_state_best.pt", map_location="cpu",)

model = model_pars
model.load_state_dict(model_state)
model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device),reduce="mean")

# %%
yhat, probs, gt = infer_model(model, tst_loader, criterion, device=device)

model.to("cpu");
# %% 

learning_curves = np.load(model_folder / "learning_curves.dict.npy", allow_pickle=True).item()
n_epochs = len(learning_curves["trn"]["loss"]); skip = 5
plt.figure()
plt.plot(learning_curves["trn"]["loss"])
plt.plot(learning_curves["val"]["loss"])
plt.ylabel("Cross-entropy (mean)")
plt.xticks(range(0,n_epochs,skip),[a for a in range(0,n_epochs,skip)])
plt.xlabel("Epochs")

plt.figure()
plt.plot(learning_curves["trn"]["accuracy"])
plt.plot(learning_curves["val"]["accuracy"])
plt.ylabel("Accuracy [%]")
plt.xticks(range(0,n_epochs,skip),[a for a in range(0,n_epochs,skip)])
plt.xlabel("Epochs")
# %% 
from sklearn.metrics import classification_report, confusion_matrix

cm = confusion_matrix(gt, yhat)
print(classification_report(gt, yhat))
print(cm)
# %%
from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                            display_labels=["No Bird", "Bird"])
disp.plot()

# %% 
probs = probs.numpy()

plt.figure()
# plt.scatter(range(100), probs[:100,0])
plt.scatter(range(100), probs[:100,1])
# plt.scatter(range(100), gt[:100]-1)
plt.scatter(range(100), gt[:100])
plt.hlines(y=0.5, xmin=0, xmax=100, color="gray")

# %% 

plt.figure()
plt.hist(probs[:,0].ravel(), bins=50, density=False, histtype="step")
plt.hist(probs[:,1].ravel(), bins=50, density=False, histtype="step")
plt.vlines(x=0.5, ymin=0, ymax=1.5, color="gray")


# %%
plt.figure()
plt.hist(probs[:,0].ravel()-probs[:,1].ravel(), bins=50, density=False)#, histtype="step")
plt.vlines(x=0, ymin=0, ymax=1.5, color="gray")
# %%
entropy = np.sum(np.log10(probs+1e-8) * (probs+1e-8), axis=1)
 
plt.figure()
plt.hist(entropy, bins=50, density=True)#, histtype="step")
plt.vlines(x=0, ymin=0, ymax=1.5, color="gray")
# %%
plt.figure()
plt.hist(probs[:,0].ravel(), bins=100, density=True, cumulative=True, histtype="step")
plt.hist(probs[:,1].ravel(), bins=100, density=True, cumulative=True, histtype="step")
plt.xlim([0,1])
plt.vlines(x=0.5, ymin=0, ymax=1, color="gray")


# %%

# %%
cl = 0
tst_sub = probs[gt==cl,1]
labs = gt[gt==cl]
sind = np.argsort(1-tst_sub)
# possort = tst_positives[ii]
xfiles = tst_hummingbirds.img_paths[gt==cl]
xnames = tst_hummingbirds.img_paths[gt==cl]
# xsort = xsort[ii]

for i, ss in enumerate(sind[60:]): 
    with open(xfiles[ss], "rb") as f:
            img = Image.open(f).convert("RGB")

    plt.figure()
    plt.title(f"LABEL: {int(labs[i])}, 0: {probs[ss,0]:.4f} - 1: {probs[ss,1]:.4f}\n"
			  f"VIDEO: {xnames[ss].name}")
    plt.imshow(img)

    if i == 30: 
        break
# %%

# %%
# Redo with random shuffle so we get a mix of classes
tst_loader_sh = DataLoader(tst_hummingbirds, shuffle=True, batch_size=BSIZE)

# visualize_model(model, tst_loader_sh, device="cpu", num_images=BSIZE, denormalize=denorm, save_folder=model_folder / "example_figs")

denorm = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

for batch, (xb,yb) in enumerate(tst_loader_sh):
    for i, (x,y) in enumerate(zip(xb,yb)):
        # print(y)
        # x, y = p
        print(y, yhat[i], probs[i,:])
        x = denorm(x).permute((1,2,0))
        plt.figure()
		# plt.title(f"y {}, gt {}, yhat {}, ")
        plt.imshow(x)

        if i == 3: 
            break

    break

# # Define Loss
# criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), reduction="mean")

# Alternatively, associate them with a very low learning rate
# 10e-2 is a scaler to the original lr.
# pars = [
#     {"params": model.features.parameters(), "lr": 1e-2},
# {"params": model.classifier.parameters(), "lr": 1},
# ]
# %%
