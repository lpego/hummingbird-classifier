# %% 
%load_ext autoreload
%autoreload 2

import os, sys #£, time, copy

os.environ['MKL_THREADING_LAYER'] = 'GNU'

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
# import datetime

import torch
import pytorch_lightning as pl
# from pytorch_lightning import Trainer
# from torch import nn
# from torch.nn import functional as F
# from torch.utils.data import DataLoader
# from torchmetrics import Accuracy, F1Score, ConfusionMatrix
# from torchvision import transforms
from torchmetrics import F1Score, PrecisionRecallCurve, ROC, ConfusionMatrix
from sklearn.metrics import classification_report

from matplotlib import pyplot as plt

try:
    __IPYTHON__
except:
    prefix = ""  # or "../"
else:
    prefix = "../"  # or "../"
    
sys.path.append(f"{prefix}src")


# from src.utils import read_pretrained_model, find_checkpoints
from utils import read_pretrained_model, find_checkpoints

from HummingbirdLoader_v2 import Denormalize # HummingbirdLoader, 
from HummingbirdLitModel import HummingbirdModel

# %% 
# /data/shared/hummingbird-classifier/hummingbirds-pil/3u92ydow
# dirs = find_checkpoints(Path(f"{prefix}lightning_logs"), version="version_0", log="last")#.glob("**/*.ckpt"))
dirs = find_checkpoints(Path(f"{prefix}hummingbirds-pil"), version="3tvjugl2", log="best")#.glob("**/*.ckpt"))

mod_path = dirs[0]

model = HummingbirdModel()  

model = model.load_from_checkpoint( 
        checkpoint_path=mod_path,
        # hparams_file= str(mod_path.parents[1] / 'hparams.yaml') #same params as args
    )


# model.data_dir= f"{prefix}{model.data_dir}"
model.pos_data_dir=f"{prefix}data/bal_cla_diff_loc_all_vid/" # bal_cla_diff_loc_all_vid/", "double_negs_bal_cla_diff_loc_all_vid/"
model.neg_data_dir=f"{prefix}data/plenty_negs_all_vid/" # bal_cla_diff_loc_all_vid/", "double_negs_bal_cla_diff_loc_all_vid/"

# model.pretrained_network="resnet50"
# model.learning_rate=1e-7
# model.batch_size=128
# model.weight_decay=0

torch.set_grad_enabled(False)   
model.eval()

print(f"Positive data dir: {model.pos_data_dir}, negative data dir: {model.neg_data_dir}")
# %% 
vname = "FH507_02" # "FH502_02"#"FH109_02" # "FH502_02"# "FH803_01"
# dataloader = model.val_dataloader()
# dataloader = model.tst_dataloader()
# dataloader = model.train_dataloader(shuffle=True)

if vname:
    dataloader = model.tst_external_dataloader(path=f"/data/shared/frame-diff-anomaly/data/{vname}/")
    annot = pd.read_csv(f"{prefix}data/Weinstein2018MEE_ground_truth.csv").drop_duplicates()
    annot = annot[annot.Video == vname].sort_values("Frame")
    annot.Truth = annot.Truth.apply(lambda x: x == "Positive")
    annot.Frame -= 1
# 
# FH102_02  FH109_02  FH207_02  FH308_01  FH403_01  FH408_02  FH503_01  FH508_01  FH509_02  FH603_01  FH608_01  FH707_01 FH802_02 
# FH107_01  FH202_02  FH303_01  FH402_01  FH403_02  FH502_01  FH507_01  FH508_02  FH602_01  FH603_02  FH703_02  FH707_02 FH803_01 
# FH108_01  FH207_01  FH303_02  FH402_02  FH408_01  FH502_02  FH507_02  FH509_01  FH602_02  FH604_01  FH706_01  FH802_01

# %% 
pbar_cb = pl.callbacks.progress.TQDMProgressBar(refresh_rate=5)

trainer = pl.Trainer(
    gpus=1, #[0,1],
    callbacks=[pbar_cb], 
    enable_checkpointing=False,
    logger=False
)

outs = trainer.predict(model=model, dataloaders=[dataloader], return_predictions=True)
# %% 
y = []; p = []; gt = []
for out in outs: 
    y.append(out[0].numpy().squeeze())
    p.append(out[1].numpy().squeeze())
    gt.append(out[2].numpy().squeeze())

try:
    yc = np.concatenate(y)
    pc = np.concatenate(p)
    gc = np.concatenate(gt)
except: 
    yc = np.array(y)
    pc = np.asarray(p)
    gc = np.asarray(gt)

if vname: 
    gc[annot[annot.Truth].Frame] = 1

plt.figure()
plt.plot(pc[:,1])
if vname: 
    plt.scatter(annot[~annot.Truth].Frame, -0*(np.ones(len(annot[~annot.Truth].Frame))), marker="^", color="k")
    plt.scatter(annot[annot.Truth].Frame, -0*(np.ones(len(annot[annot.Truth].Frame))), marker="^", color="r")
plt.grid(True)

# %% 
if 0: 
    FR_F = 0
    TO_F = -1

    # this needs to be ordered
    files = dataloader.dataset.img_paths.copy()

    yy = np.argsort(-pc[FR_F:TO_F,1])

    plt.figure()
    plt.plot(pc[FR_F:TO_F,1])
    if vname: 
        plt.scatter(annot[~annot.Truth].Frame - FR_F, -0*(np.ones(len(annot[~annot.Truth].Frame))), marker="^", color="k")
        plt.scatter(annot[ annot.Truth].Frame - FR_F, -0*(np.ones(len(annot[annot.Truth].Frame))), marker="^", color="r")
    plt.grid(True)
    
    if 0: 
        sub_ind = (pc[yy,1] > 0) & (gc[yy] == 1)

        # sub_ind = (pc[yy,1] > 0.7)

        # sub_ind = np.where(sub_ind)[0]
        # files = files[sub_ind]  
        ss = yy[sub_ind]  
        # sub_ind=np.arange(pc.shape[0])
        # sub_p = pc[:,1]
        # yy = np.argsort(-sub_p)

        denorm = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        for i, ti in enumerate(ss):
            if i < 10: 
                continue
            fi = files[ti] 
            x = Image.open(fi).convert("RGB")
            plt.figure(figsize=(6,6))
            plt.imshow(x)
            plt.title(f"{i}, {ti}: yhat {yc[ti]:.2f}, gt {gc[ti]:.2f}, p0 {pc[ti,0]:.2f}, p1 {pc[ti,1]:.2f}\n{fi.name}")
            plt.show()

            if i > 40:
                break
# %% 
if 1:
    # from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, 
    # AblationCAM, XGradCAM, EigenCAM, FullGrad

    from pytorch_grad_cam import GradCAMPlusPlus as CAM
    from pytorch_grad_cam.utils.image import *
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    torch.set_grad_enabled(False)   
    denorm = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    cam = CAM(model=model.model, target_layers=[model.model.layer4[-1]], use_cuda=False)
    files = dataloader.dataset.img_paths.copy()

    inc = 1
    CLIP = -1
    yy = np.argsort(-pc[:CLIP,inc])
    sub_ind = (pc[yy,inc] > 0)# & (gc[yy] == 1)
    ss = yy[sub_ind]

    plt.figure()
    plt.plot(pc[:CLIP,1])
    FR_F = 0
    if vname: 
        plt.scatter(annot[~annot.Truth].Frame - FR_F, -0*(np.ones(len(annot[~annot.Truth].Frame))), marker="^", color="k")
        plt.scatter(annot[ annot.Truth].Frame - FR_F, -0*(np.ones(len(annot[annot.Truth].Frame))), marker="^", color="r")
    plt.grid(True)

    FR = 0; N = 20
    for i, ti in enumerate(ss):
        if i < FR:
            continue
    
        fi = files[ti]
        im = Image.open(fi).convert("RGB")
        x = model.transform_ts(im)
        x = x[None, ...]
        
        with torch.set_grad_enabled(True):
            gcam = [cam(input_tensor=x, targets=[ClassifierOutputTarget(cl)]).squeeze() for cl in [0,1]]

        p = torch.softmax(model(x), dim = 1).cpu().numpy()
        pl_im = np.transpose(denorm(x.squeeze()),(1,2,0)).numpy()

        f, a = plt.subplots(1,3,figsize=(12,6))
        for c in range(2):
            g_viz = show_cam_on_image(pl_im, gcam[c], use_rgb=True)
            a[c].imshow(g_viz)
            a[c].set_title(f"CAM for Class {c}")
            # a[].imshow(g_viz)
        a[2].imshow(pl_im)

        plt.suptitle(f"{i}, {ti}: yhat {yc[ti]:.2f}, gt {gc[ti]:.2f}, p0 {pc[ti,0]:.2f}, p1 {pc[ti,1]:.2f} \n online p {p[0,1]:.5f}\n{fi.name}")
        plt.show()
        
        if i > FR+N:
            break

    # %%
# TODO
# Cross tab errors per video source, check if some videos are better modeled 
# than others for the negative class. 


# names = ["_".join(str(f.name).split("_")[:-2]) for f in files]
# pred_agg = pd.DataFrame({"fname": names, "gt": gc, "yha": yc}) 
# pred_agg["diff"] = pred_agg["gt"] != pred_agg["yha"]

# group = pred_agg[pred_agg["gt"] == 0].groupby("fname").agg({"gt": "count", "yha": "sum", "diff": "sum"})
# group["perc"] = group["diff"] / group["gt"]
# plt.figure()
# plt.plot(group["perc"]);



# %% 

if 0:
    if vname: 
        pred = pc[annot.Frame,:]
        grtr = gc[annot.Frame]
    else: 
        pred = pc
        grtr = gc

    fsc = F1Score(num_classes=2)
    sc = []
    threshs = np.arange(0.1,1,0.1)
    for t in threshs:
        sc.append(fsc(torch.tensor(pc[:,1] > t), torch.tensor(gc)))
        # print(f"THRESH = {t:.1f}: Acc: {np.sum(pc[:,1] > t):.1f}/{np.sum(gc == 1)}: {np.sum(pc[:,1] > t)/np.sum(gc == 1):.2f}%")

    plt.figure()
    plt.plot(sc)
    plt.xticks(np.arange(len(threshs)), [f"{a:.2f}" for a in  threshs]);
    plt.grid("on")
    plt.ylabel("F1 Score")

    pr_curve = PrecisionRecallCurve(num_classes=2)
    precision, recall, thresholds = pr_curve(torch.tensor(pc), torch.tensor(gc))
    plt.figure()
    for c in range(2):
        plt.plot(precision[c], recall[c], label=f"class {c}")
    plt.grid("on")
    plt.legend()
    plt.xlabel("Precision")
    plt.ylabel("Recall")

    roc = ROC(num_classes=2)
    fpr, tpr, thresholds = roc(torch.tensor(pc), torch.tensor(gc))
    plt.figure()
    for c in range(2):
        plt.plot(fpr[c], tpr[c], label=f"class {c}")
    plt.grid("on")
    plt.legend()
    plt.xlabel("FPR")
    plt.ylabel("TPR")

print(classification_report(pc[:,1] > 0.5, gc))
print(ConfusionMatrix(num_classes=2)(torch.tensor(pc), torch.tensor(gc)).numpy())
# %%
if 0:
    denorm = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    # dataloader = model.tst_external_dataloader()
    dataloader = model.train_dataloader()
    dliter = iter(dataloader)
    x, y, id = next(dliter)
    print(x.shape)

    # x = x[0,...]
    p = torch.softmax(model(x), dim=1)
    p = p.numpy()
    for i in range(10):#x.shape[0]):
        plt.figure()
        im = x[i,...]   
        plt.imshow(np.transpose(denorm(im),(1,2,0)))
        plt.title(f"{p[i,0]:.2f}, {p[i,1]:.2f}")

# %%

# fi = files[49862]
# im = Image.open(fi).convert("RGB")
# crop_img = im.crop((200,1,1080,700))

# x = model.transform_tr(im)
# pl_im = np.transpose(denorm(x.squeeze()),(1,2,0)).numpy()

# f, a = plt.subplots(1,2,figsize=(12,6))
# a[0].imshow(crop_img)
# a[1].imshow(pl_im)
# %%
