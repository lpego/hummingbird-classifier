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
from skimage import exposure

try:
    __IPYTHON__
except:
    prefix = ""  # or "../"
else:
    prefix = "../"  # or "../"
    
sys.path.append(f"{prefix}src")

# from src.utils import read_pretrained_model, find_checkpoints
from utils import read_pretrained_model, find_checkpoints
from triplet_diff_utils import main_triplet_difference

from HummingbirdLoader import Denormalize # HummingbirdLoader, 
from HummingbirdModel import HummingbirdModel

# %% 
EXPERIMENT = "ME_FULL"

# dirs = find_checkpoints(Path(f"{prefix}lightning_logs"), version="version_0", log="last")#.glob("**/*.ckpt"))
# dirs = find_checkpoints(Path(f"{prefix}hummingbirds-pil"), version="ixpfqgvo", log="best")#.glob("**/*.ckpt")) 

# dirs = find_checkpoints(Path(f"{prefix}hummingbirds-pil"), version="2wtmxr3l", log="best")#.glob("**/*.ckpt")) # Asymmetric DA
# dirs = find_checkpoints(Path(f"{prefix}hummingbirds-pil"), version="21aumca8", log="best")#.glob("**/*.ckpt")) # Symmetric DA
# dirs = find_checkpoints(Path(f"{prefix}hummingbirds-pil"), version="2cdynun2", log="best")#.glob("**/*.ckpt")) # LARGE DA
# dirs = find_checkpoints(Path(f"{prefix}hummingbirds-pil"), version="3anp5naw", log="best")#.glob("**/*.ckpt")) # LARGE DA

dirs = find_checkpoints(Path(f"{prefix}hummingbirds-pil"), version="1jartsnn", log="best")#.glob("**/*.ckpt")) # LARGE DA

# THIS WORKS: sfrfhnc3, 24zruk7z DENSENET161: 38tn45xv, 2col29g3, tba 130ch647  
# || GOOD ixpfqgvo very very long 3pau0qtg / very long 32tka2n9 / long 22m0pigr / mid bqoy698f / short 23rgsozp
# THIS WORKS LESS WELL: 1zh8fqdf
# dirs = find_checkpoints(Path(f"{prefix}hummingbirds-pil"), version="24zruk7z", log="last")#.glob("**/*.ckpt"))

mod_path = dirs[0]

model = HummingbirdModel()  

model = model.load_from_checkpoint( 
        checkpoint_path=mod_path,
        # hparams_file= str(mod_path.parents[1] / 'hparams.yaml') #same params as args
    )

model.pos_data_dir=f"{prefix}data/bal_cla_diff_loc_all_vid/" # bal_cla_diff_loc_all_vid/", "double_negs_bal_cla_diff_loc_all_vid/"
model.neg_data_dir=f"{prefix}data/plenty_negs_all_vid/" # bal_cla_diff_loc_all_vid/", "double_negs_bal_cla_diff_loc_all_vid/"

model.eval()

print(f"Positive data dir: {model.pos_data_dir}, negative data dir: {model.neg_data_dir}")
# %%
vname = "FH102_02" #"FH803_01" # "FH109_02"# "FH502_02"
#  THESE ONES PERFECT CHERRY PICK FH602_01 FH703_02 FH207_01
# dataloader = model.val_dataloader() # FH509_01, FH403_02, FH102_02 the worse FH408_02
# FH308_01 Wet camera, 
# FH107_01 Good example for why triplet socre is important. 

dataloader = model.tst_dataloader()
# dataloader = model.train_dataloader(shuffle=True)

if vname:
    dataloader = model.tst_external_dataloader(path=f"/data/shared/frame-diff-anomaly/data/{vname}/")
    annot = pd.read_csv(f"{prefix}data/Weinstein2018MEE_ground_truth.csv").drop_duplicates()
    annot = annot[annot.Video == vname].sort_values("Frame")
    annot.Truth = annot.Truth.apply(lambda x: x == "Positive")
    annot.Frame -= 1

# ANNOTATED VIDEOS AVAILABLE
# FH102_02 FH107_01 FH108_01 FH109_02 FH202_02 FH207_01 FH207_02 FH303_01 FH303_02 FH308_01 
# FH402_01 FH402_02 FH403_01 FH403_02 FH408_01 FH408_02 FH502_01 FH502_02 FH503_01 FH507_01 
# FH507_02 FH508_01 FH508_02 FH509_01 FH509_02 FH602_01 FH602_02 FH603_01 FH603_02 FH604_01 
# FH608_01 FH703_02 FH706_01 FH707_01 FH707_02 FH802_01 FH802_02 FH803_01 
# %% 
# pbar_cb = pl.callbacks.progress.TQDMProgressBar(refresh_rate=5)

score_csv = Path(f"{prefix}data/pred_csv/{EXPERIMENT}/{vname}.csv")

if 1 : #not score_csv.is_file():
    trainer = pl.Trainer(
        max_epochs=1,
        gpus=1, #[0,1],
        # callbacks=[pbar_cb], 
        enable_checkpointing=False,
        logger=False
    )

    outs = trainer.predict(model=model, dataloaders=[dataloader], return_predictions=True)

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

else :
    df_scores = pd.read_csv(score_csv)
    yc = (df_scores["probabilities"] > 0.5).astype(int).values
    gc = df_scores["gt"].values
    pc = np.zeros(shape=(df_scores.shape[0],2))
    pc[:,1] = df_scores["probabilities"].values
    pc[:,0] = 1-pc[:,1]

plt.figure()
plt.plot(pc[:,1])
if vname: 
    plt.scatter(annot[~annot.Truth].Frame, -0*(np.ones(len(annot[~annot.Truth].Frame))), marker="^", color="k")
    plt.scatter(annot[annot.Truth].Frame, -0*(np.ones(len(annot[annot.Truth].Frame))), marker="^", color="r")
plt.grid(True)


# %% 
if 1:
    # from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, 
    # AblationCAM, XGradCAM, EigenCAM, FullGrad

    from pytorch_grad_cam import GradCAMPlusPlus as CAM
    from pytorch_grad_cam.utils.image import *
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    denorm = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    compute_triplet = True

    for param in model.model.parameters():
            param.requires_grad = True

    # for DenseNet161
    cam = CAM(model=model.model, target_layers=[model.model.features[-1]], use_cuda=False)

    files = dataloader.dataset.img_paths.copy()

    fpath = Path(f"/data/shared/frame-diff-anomaly/data/{vname}/_scores_triplet.csv")
    
    if compute_triplet & (not fpath.is_file()):
        score_t_diff = main_triplet_difference(fpath.parent, save_csv="triplet")

    elif fpath.is_file():
        print(f"{vname} triplet loss exists already, loading it.")
        score_t_diff = pd.read_csv(fpath)

    score_t_diff = score_t_diff.mag_std.values.astype(float)
    score_t_diff = (score_t_diff - score_t_diff.min())/(score_t_diff.max() - score_t_diff.min())

    score_p_diff = np.concatenate(([0], np.abs(np.diff(pc[:,1]))))
    
    CLIP = len(score_t_diff)

    sort_score = (0.1*score_p_diff + 0.5*pc[:,1] + 0.4*score_t_diff)[:CLIP]
    # sort_score = (0.1*score_p_diff + 0.8*pc[:,1] + 0.1*score_t_diff)[:CLIP]

    sort_frames = np.argsort(-sort_score[:])

    if 0: # the heck is that here
        sub_ind = (score[yy] > 0)# & (gc[yy] == 1)
        ss = yy[sub_ind]# + 1 if score is the diff! starts at 1 and not at 0
        
    plt.figure(figsize=(15,5))
    plt.title(f"Proba time series, with N_positive = {annot.Truth.sum()}")
    plt.plot(sort_score, label="agg_score")
    plt.plot(0.33*pc[:CLIP,1], label="pc")
    plt.plot(0.33*score_p_diff[:CLIP], label="diff")
    plt.plot(0.33*score_t_diff[:CLIP], label="3diff")
    plt.legend(loc="lower right")

    FR_F = 0
    if vname: 
        plt.scatter(annot[~annot.Truth].Frame - FR_F, 1*(np.ones(len(annot[~annot.Truth].Frame))), marker="v", color="k")
        plt.scatter(annot[ annot.Truth].Frame - FR_F, 1*(np.ones(len(annot[annot.Truth].Frame))), marker="v", color="r")
    plt.grid(True)

    DETECT = 0
    DIFF = True
    FR = 0; N = 1
    for i, ti in enumerate(sort_frames):
        if i < FR:
            continue
        # if vname:
        # ti += 1
        
        fi = files[ti]
        im = Image.open(fi).convert("RGB")
        x = model.transform_ts(im)
        x = x[np.newaxis,...]

        im_ = Image.open(files[ti-1]).convert("RGB")
        x_ = model.transform_ts(im_)
        x_ = np.transpose(x_.squeeze(),(1,2,0))
        pl_im_pre = denorm(x_).numpy()
        
        im_ = Image.open(files[ti+1]).convert("RGB")
        x_ = model.transform_ts(im_)
        x_ = np.transpose(x_.squeeze(),(1,2,0))
        pl_im_pos = denorm(x_).numpy()

        # x_g = torch.clone(x)
        x.requires_grad = True
        with torch.set_grad_enabled(True):
            gcam = [cam(input_tensor=x, targets=[ClassifierOutputTarget(cl)]).squeeze() for cl in [0,1]]

        with torch.set_grad_enabled(False):
            p = torch.softmax(model(x), dim = 1).cpu().numpy()
            x_ = np.transpose(x.squeeze(),(1,2,0))
            pl_im = denorm(x_).numpy()

        f, a = plt.subplots(1,6,figsize=(17,4))
        for c in range(2):
            g_viz = show_cam_on_image(pl_im, gcam[c], use_rgb=True)
            a[c].imshow(g_viz)
            a[c].set_title(f"CAM for Class {c}")
            a[c].axis("off")
            # a[].imshow(g_viz)

        a[2].imshow(pl_im_pre)
        a[2].set_title(f"Pre-image: GT {gc[ti-1]}")
        a[2].axis("off")
        a[3].imshow(pl_im)
        a[3].set_title(f"Inference: GT {gc[ti]}")
        a[3].axis("off")
        a[4].imshow(pl_im_pos)
        a[4].set_title(f"Post-image: GT {gc[ti+1]}")
        a[4].axis("off")

        if DIFF:
            im_0 = exposure.match_histograms(pl_im_pre, pl_im, channel_axis=2)
            im_2 = exposure.match_histograms(pl_im_pos, pl_im, channel_axis=2)

            d1 = pl_im - im_0
            d2 = im_2 - pl_im
            dh = (1 + d1 - d2) / 2
            dh -= dh.min(); dh /= dh.max()
            a[5].imshow(dh)
            a[5].axis("off")

        plt.suptitle(
            f"{i}, {ti}: yhat {int(yc[ti])}, gt {int(np.any((gc[ti],gc[ti-1],gc[ti+1])))} ({int(gc[ti])}, {np.sum((gc[ti],gc[ti-1],gc[ti+1]))}), "\
            f"p0 {pc[ti,0]:.2f}, p1 {pc[ti,1]:.2f}\n"\
            f"online p {p[0,1]:.5f}, diff score {score_p_diff[ti]:.2f}, mag_triplet {score_t_diff[ti]:.2f}"\
            f"\n{fi.name}")
        plt.show()

        DETECT += int(np.any((gc[ti],gc[ti-1],gc[ti+1])))

        if i > FR+N:
            break

        
print(f"got {DETECT} out of {annot.Truth.sum()} in {N} frames")


# %% 

if 1:
    if vname: 
        pred = pc[annot.Frame,:]
        grtr = gc[annot.Frame]
    else: 
        pred = pc
        grtr = gc

    fsc = F1Score(num_classes=2)
    scosc = []; prosc = []; trisc = []
    threshs = np.arange(0.1,1,0.1)
    for t in threshs:
        scosc.append(fsc(torch.tensor(sort_score > t), torch.tensor(gc)))
        prosc.append(fsc(torch.tensor(pc[:,1] > t), torch.tensor(gc)))
        trisc.append(fsc(torch.tensor(score_t_diff > t), torch.tensor(gc)))

    plt.figure()
    plt.plot(scosc, label="bigscore")
    plt.plot(trisc, label="triplet")
    plt.plot(prosc, label="probas")
    plt.xticks(np.arange(len(threshs)), [f"{a:.2f}" for a in  threshs]);
    plt.grid("on")
    plt.ylabel("F1 Score")
    plt.legend()
    # pr_curve = PrecisionRecallCurve(num_classes=2)
    # precision, recall, thresholds = pr_curve(torch.tensor(pc), torch.tensor(gc))
    # plt.figure()
    # for c in range(2):
    #     plt.plot(precision[c], recall[c], label=f"class {c}")
    # plt.grid("on")
    # plt.legend()
    # plt.xlabel("Precision")
    # plt.ylabel("Recall")

    # roc = ROC(num_classes=2)
    # fpr, tpr, thresholds = roc(torch.tensor(pc), torch.tensor(gc))
    # plt.figure()
    # for c in range(2):
    #     plt.plot(fpr[c], tpr[c], label=f"class {c}")
    # plt.grid("on")
    # plt.legend()
    # plt.xlabel("FPR")
    # plt.ylabel("TPR")

# print("Pure p(y=bird|x)")
# print(classification_report(gc, pc[:,1] > 0.5))
# print("Mix score @ 0.5")
# print(classification_report(gc, sort_score > 0.5))
print("Pure p(y=bird|x)")
print(ConfusionMatrix(num_classes=2)(torch.tensor(pc[:,1]) > 0.5 , torch.tensor(gc)).numpy())
print("Mix score @ 0.5")
print(ConfusionMatrix(num_classes=2)(torch.tensor(sort_score) > 0.5, torch.tensor(gc)).numpy())
# print("Mix score @ 0.65")
# print(ConfusionMatrix(num_classes=2)(torch.tensor(sort_score) > 0.65, torch.tensor(gc)).numpy())
# print(ConfusionMatrix(num_classes=2)(torch.tensor(sort_score) > 0., torch.tensor(gc)).numpy())


# fi = files[49862]
# im = Image.open(fi).convert("RGB")
# crop_img = im.crop((200,1,1080,700))

# x = model.transform_tr(im)
# pl_im = np.transpose(denorm(x.squeeze()),(1,2,0)).numpy()

# f, a = plt.subplots(1,2,figsize=(12,6))
# a[0].imshow(crop_img)
# a[1].imshow(pl_im)
# %%
if 1: 
    fpath = Path(f"/data/shared/frame-diff-anomaly/data/{vname}/_scores_triplet.csv")
    if compute_triplet & (not fpath.is_file()):
        score_t_diff = main_triplet_difference(fpath.parent, save_csv="triplet")

    elif fpath.is_file():
        print(f"{vname} triplet loss exists already, loading it.")
        score_t_diff = pd.read_csv(fpath)

    score_t_diff = score_t_diff.mag_std.values.astype(float)
    score_t_diff = (score_t_diff - score_t_diff.min())/(score_t_diff.max() - score_t_diff.min())

    score_p_diff = np.concatenate(([0], np.abs(np.diff(pc[:,1]))))

    CLIP = len(score_t_diff)

    sort_score = (0.1*score_p_diff + 0.5*pc[:,1] + 0.4*score_t_diff)[:CLIP]
    sort_frames = np.argsort(-sort_score[:])

    if 0: # the heck is that here
        sub_ind = (score[yy] > 0)# & (gc[yy] == 1)
        ss = yy[sub_ind]# + 1 if score is the diff! starts at 1 and not at 0

    FROM = 20000
    plt.figure(figsize=(15,5))
    plt.title(f"Proba time series, with N_positive = {annot.Truth.sum()}")
    plt.plot(pc[:CLIP,1], label="pc")
    plt.plot(score_p_diff[:CLIP], label="diff")
    plt.plot(score_t_diff[:CLIP], label="3diff")
    plt.plot(sort_score, label="agg_score")

    plt.legend(loc="lower right")

    plt.scatter(annot[~annot.Truth].Frame - FR_F, 1*(np.ones(len(annot[~annot.Truth].Frame))), marker="v", color="k")
    plt.scatter(annot[ annot.Truth].Frame - FR_F, 1*(np.ones(len(annot[annot.Truth].Frame))), marker="v", color="r")
    plt.grid(True)
# plt.xlim(20000,21500)

# %%
f, a = plt.subplots(4,1,figsize=(15,9))
a[0].scatter(annot[~annot.Truth].Frame - FR_F, 1*(np.ones(len(annot[~annot.Truth].Frame))), marker="v", color="k")
a[0].scatter(annot[ annot.Truth].Frame - FR_F, 1*(np.ones(len(annot[annot.Truth].Frame))), marker="v", color="r")
a[0].plot(sort_score[:CLIP], label="anomaly score")
a[0].grid(True)
a[0].set_ylabel("Aggregated score")
a[1].plot(pc[:CLIP,1], label="p hummingbird")
a[1].grid(True)
a[1].set_ylabel("p bird")
a[2].plot(score_p_diff[:CLIP], label="p differential")
a[2].grid(True)
a[2].set_ylabel("p difference")
a[3].plot(score_t_diff[:CLIP], label="f difference")
a[3].grid(True)
a[3].set_ylabel("Frame difference")
a[3].set_xlabel("Frame number")

# %%
