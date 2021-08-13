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
# from utils import get_activation

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

hub_dir = Path(f"/data/shared/hummingbird-classifier/models/").resolve()
torch.hub.set_dir(hub_dir)

print(f"current torch hub directory: {torch.hub.get_dir()}")
# %% # %%
BSIZE = 32
set_type = "annotated_videos"  # "balanced", "more_negatives", "same_camera"
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

augment_tr = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
        # transforms.RandomAdjustSharpness(sharpness_factor=2),
        # transforms.RandomEqualize(),
        # transforms.RandomAutocontrast(),
        transforms.ColorJitter(brightness=0.5, hue=0.1),
        transforms.Resize((224, 224), interpolation=Image.BILINEAR),  # AT LEAST 224
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

augment_ts = transforms.Compose(
    [
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224, 224), interpolation=Image.BILINEAR),  # AT LEAST 224
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

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

# architecture = f"ResNet50_same_camera_jitter_augmentation_20210808_224x"
architecture = f"ResNet50_annotated_videos_jitter_augmentation_20210809_224x"
model_folder = Path(f"{hub_dir}/{architecture}/")

model_pars = torch.load(model_folder / "model_pars_best.pt", map_location="cpu",)
model_state = torch.load(model_folder / "model_state_best.pt", map_location="cpu",)

model = model_pars
model.load_state_dict(model_state)
model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device),reduce="mean")

# %% 
# make curves 
import pandas as pd
cur = pd.read_table("../outputs/curves.txt", header=None)
nd = pd.DataFrame(index=range(30),columns=["tloss", "accuracz"])

# track_learning = {}
# track_learning["trn"] = {}
# track_learning["trn"]["loss"].append(epoch_loss)
# track_learning["trn"]["accuracy"].append(epoch_acc)
# track_learning["val"] = {}
# track_learning["val"]["loss"].append(epoch_loss)
# track_learning["val"]["accuracy"].append(epoch_acc)
df_plt = pd.DataFrame([], index=range(30), columns=["trn_loss","trn_acc","val_loss","val_acc"])
for i, line in cur.iterrows(): 
    line = line.values[0]
    # print(line)
    if "Epoch" in line: 
        ind = int(line.split(" ")[1].split("/")[0])-1
    if "trn" in line:
        df_plt.iloc[ind,:].loc["trn_loss"] = line.split(" ")[2]
        df_plt.iloc[ind,:].loc["trn_acc"] = line.split(" ")[4]
    elif "val" in line: 
        df_plt.iloc[ind,:].loc["val_loss"] = line.split(" ")[2]
        df_plt.iloc[ind,:].loc["val_acc"] = line.split(" ")[4]
    
df_plt = df_plt.astype(float)

# %% 

learning_curves = np.load(model_folder / "learning_curves.dict.npy", allow_pickle=True).item()
n_epochs = len(learning_curves["trn"]["loss"]); skip = 5
plt.figure()
plt.title(f"N_epochs = {n_epochs}")
xtlo = np.arange(0,31,5)
xtlo[0] = 1
xtla = [(a) for a in xtlo]

# plt.plot(learning_curves["trn"]["loss"])
# plt.plot(learning_curves["val"]["loss"])
plt.plot(np.arange(1,31), df_plt["trn_loss"].values)
plt.plot(np.arange(1,31), df_plt["val_loss"].values)
plt.ylabel("Cross-entropy (mean)")
plt.xticks(xtlo, xtla)
plt.xlabel("Epochs")

plt.figure()
plt.title(f"N_epochs = {n_epochs}")
# plt.plot(learning_curves["trn"]["accuracy"])
# plt.plot(learning_curves["val"]["accuracy"])
plt.plot(np.arange(1,31),df_plt["trn_acc"].values)
plt.plot(np.arange(1,31),df_plt["val_acc"].values)
plt.ylabel("Accuracy [%]")
plt.xticks(xtlo, xtla)
plt.xlabel("Epochs");
# %%
if 0: 
    yhat, probs, gt = infer_model(model, tst_loader, criterion, device=device)
    model.to("cpu");
    np.save(model_folder / "predictions_proba.npy", probs)
    np.save(model_folder / "predictions_gt.npy", gt)
    np.save(model_folder / "predictions_yhat.npy", yhat)
else: 
    probs = np.load(model_folder / "predictions_proba.npy", allow_pickle=True)
    gt = np.load(model_folder / "predictions_gt.npy", allow_pickle=True)
    yhat = np.load(model_folder / "predictions_yhat.npy", allow_pickle=True)

## %% 
# model.to("cuda")
# yhat_val, probs_val, gt_val = infer_model(model, val_loader, criterion, device=device)
# model.to("cpu");

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

plt.figure(figsize=(15,5))
plt.scatter(range(len(gt)), probs[:,1])
# plt.scatter(range(100), probs[:100,1])
plt.scatter(range(len(gt)), gt)
# plt.scatter(range(100), gt[:100])
plt.hlines(y=0.5, xmin=0, xmax=100, color="gray");

# %% 
plt.figure(figsize=(15,5))
plt.hist(probs[gt==0,1].ravel(), bins=50, density=True, histtype="step", label="no hummingbird")
plt.hist(probs[gt==1,1].ravel(), bins=50, density=True, histtype="step", label="hummingbird")
plt.vlines(x=0.5, ymin=0, ymax=2, color="gray", linestyles=":")
plt.legend(loc=9)

# %%
plt.figure(figsize=(15,5))
plt.hist(probs[:,0].ravel()-probs[:,1].ravel(), bins=50, density=False)#, histtype="step")
plt.vlines(x=0, ymin=0, ymax=1.5, color="gray")
# %%
entropy = -np.sum(np.log10(probs+1e-8) * (probs+1e-8), axis=1)
plt.figure(figsize=(15,5))
# plt.hist(entropy, bins=50, density=True)#, histtype="step")
plt.scatter(range(len(gt)), entropy)
plt.scatter(range(len(gt)), gt*0.302)

# %% 
sind = np.argsort(-entropy)
# possort = tst_positives[ii]
xfiles = tst_hummingbirds.img_paths
xnames = tst_hummingbirds.img_paths
# xsort = xsort[ii]

c = 0
for i, ss in enumerate(sind[:]): 
    with open(xfiles[ss], "rb") as f:
            img = Image.open(f).convert("RGB")

    plt.figure()
    print(f"{c} : LABEL: {int(gt[ss])}, 0: {entropy[ss]:.4f}\n"
              f"FILE: {xnames[ss].name}")
    plt.title(f"Hummingbird entropy: {entropy[ss]:.2f}, GT: {int(gt[ss])}, p0: {probs[ss,0]:.4f} - p1: {probs[ss,1]:.4f}")
    plt.imshow(img)
    plt.axis("off")
    
    if c > 5: 
        break
    
    c += 1

 # %%
plt.figure()
plt.hist(probs[:,0].ravel(), bins=100, density=True, cumulative=True, histtype="step")
plt.hist(probs[:,1].ravel(), bins=100, density=True, cumulative=True, histtype="step")
plt.xlim([0,1])
plt.vlines(x=0.5, ymin=0, ymax=1, color="gray")

# %%
cl = 0
tst_sub = probs[gt==cl,:]
labs = gt[gt==cl]
sind = np.argsort(1-tst_sub[:,1])
# possort = tst_positives[ii]
xfiles = tst_hummingbirds.img_paths[gt==cl]
xnames = tst_hummingbirds.img_paths[gt==cl]
# xsort = xsort[ii]

c = 0
for i, ss in enumerate(sind[:]): 
    if "FH509" not in str(xfiles[ss]): 
        continue
    # elif "FH107" in str(xfiles[ss]): 
    #     continue

    with open(xfiles[ss], "rb") as f:
            img = Image.open(f).convert("RGB")

    plt.figure()
    print(f"{c} : {ss} : LABEL: {int(labs[ss])}, 0: {tst_sub[ss,0]:.4f} - 1: {tst_sub[ss,1]:.4f}\n"
              f"FILE: {xnames[ss].name}")
    plt.title(f"Hummingbird probability: {tst_sub[ss,1]:.2f}, GT: {int(labs[ss])}")
    plt.imshow(img)
    plt.axis("off")
    
    if c >= 10: 
        break
    
    c += 1

# %%

# %%
# Redo with random shuffle so we get a mix of classes
# tst_loader_sh = DataLoader(tst_hummingbirds, shuffle=True, batch_size=BSIZE)

# # visualize_model(model, tst_loader_sh, device="cpu", num_images=BSIZE, denormalize=denorm, save_folder=model_folder / "example_figs")

# denorm = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

# for batch, (xb,yb) in enumerate(tst_loader_sh):
#     for i, (x,y) in enumerate(zip(xb,yb)):
#         # print(y)
#         # x, y = p
#         print(y, yhat[i], probs[i,:])
#         x = denorm(x).permute((1,2,0))
        
#         plt.figure()
#         # plt.title(f"y {}, gt {}, yhat {}, ")
#         plt.imshow(x)

#         if i == 3: 
#             break

#     break

# # Define Loss
# criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), reduction="mean")

# Alternatively, associate them with a very low learning rate
# 10e-2 is a scaler to the original lr.
# pars = [
#     {"params": model.features.parameters(), "lr": 1e-2},
# {"params": model.classifier.parameters(), "lr": 1},
# ]
# %%

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.layer4[1].register_forward_hook(get_activation("layer4"))

# Redo with random shuffle so we get a mix of classes
tst_loader_sh = DataLoader(tst_hummingbirds, shuffle=True, batch_size=BSIZE)

# visualize_model(model, tst_loader_sh, device="cpu", num_images=BSIZE, denormalize=denorm, save_folder=model_folder / "example_figs")

denorm = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

for batch, (xb,yb) in enumerate(tst_loader_sh):
    for i, (x,y) in enumerate(zip(xb,yb)):
        # print(y)
        # x, y = p
        # x = denorm(x)#.permute((1,2,0))
        out = model(x[None, ...])#.to("cuda"))
        out = out.cpu()
        print(y.numpy(), out.max(dim=1)[1].numpy()[0], nn.Softmax(dim=-1)(out[0]).detach().numpy())

        # print(out)
        plt.figure()
        # plt.title(f"y {}, gt {}, yhat {}, ")
        plt.imshow(denorm(x).permute((1,2,0)))
        if i == 0: 
            break

    break

# output = model(img)
# %%
# act = torch.squeeze(activation['layer4']).cpu().permute((1,2,0)).numpy()#.max(dim=2)[1]
# 
# c = 0
# f, a = plt.subplots(1,2)
# a[0].imshow(x.cpu().permute((1,2,0)))
# a[1].imshow(act[:,:,c:c+3])

# %%
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 

act = torch.squeeze(activation['layer4']).cpu().permute((1,2,0)).numpy()#.max(dim=2)[1]

act_r = act.reshape(-1,2048)
pca = PCA(n_components=12, whiten=True)
x_tr = pca.fit_transform(act_r)

clu = KMeans(n_clusters=10)
clu = clu.fit(x_tr)
cind = clu.labels_.reshape(7,7)#[:,::-1]

x_2d = x_tr.reshape(7,7,-1)#[:,::-1,:]
x_2d = np.uint8(255 * (x_2d - np.min(x_2d)) / (np.max(x_2d) - np.min(x_2d)))

c = 0
f, a = plt.subplots(1,2)
a[0].imshow(x.cpu().permute((1,2,0)))
a[1].imshow(x_2d[:,:,c:c+3])

c = 3
f, a = plt.subplots(1,2)
a[0].imshow(x.cpu().permute((1,2,0)))
a[1].imshow(x_2d[:,:,c:c+3])

c = 6
f, a = plt.subplots(1,2)
a[0].imshow(x.cpu().permute((1,2,0)))
a[1].imshow(x_2d[:,:,c:c+3])

c = 9
f, a = plt.subplots(1,2)
a[0].imshow(x.cpu().permute((1,2,0)))
a[1].imshow(x_2d[:,:,c:c+3])

f, a = plt.subplots(1,2)
a[0].imshow(x.cpu().permute((1,2,0)))
a[1].imshow(cind)
# %%
from matplotlib import colors
c = 0
mag = np.sqrt(np.power(x_tr[:,c],2) + np.power(x_tr[:,c+1],2)).reshape(-1,1)
angle = np.arctan2(x_tr[:,c+1], x_tr[:,c]) 

# ax[0,1].scatter(X_embedded[:,0], X_embedded[:,1],c=angle,s=2,cmap=plt.cm.hsv)

angle = (angle-np.min(angle)) / (np.max(angle) - np.min(angle))
mag = np.sqrt(mag)
mag = (mag-np.min(mag)) / (np.max(mag) - np.min(mag))

colors = colors.hsv_to_rgb(np.concatenate((angle.reshape(-1,1),  
                            mag.reshape(-1,1), #mag.reshape(-1,1),
                            mag.reshape(-1,1)), axis=1))

# col2d = colors.reshape((-1,3))

# ii = 0
plt.figure()
plt.scatter(x_tr[:,c],x_tr[:,c+1], c=colors, #df['hs'].mean(axis=1),
                s=50, alpha=1)#, cmap=newcmap)

f, a = plt.subplots(1,2)
a[0].imshow(x.cpu().permute((1,2,0)))
a[1].imshow(colors.reshape(7,7,3))

# %%
