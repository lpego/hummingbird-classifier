# %% 
%load_ext autoreload
%autoreload 2

import os, sys #time, copy

os.environ['MKL_THREADING_LAYER'] = 'GNU'

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import pytorch_lightning as pl
# from torchmetrics import F1Score, PrecisionRecallCurve, ROC, ConfusionMatrix
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

from HummingbirdLoader import Denormalize # HummingbirdLoader, 
from HummingbirdModel import HummingbirdModel        

# %% 
# /data/shared/hummingbird-classifier/hummingbirds-pil/3u92ydow
# dirs = find_checkpoints(Path(f"{prefix}lightning_logs"), version="version_0", log="last")#.glob("**/*.ckpt"))
dirs = find_checkpoints(Path(f"{prefix}hummingbirds-pil"), version="3pau0qtg", log="best")#.glob("**/*.ckpt")) 

# THIS WORKS: sfrfhnc3, 24zruk7z DENSENET161: 38tn45xv, 2col29g3, tba 130ch647  
# || GOOD very very long 3pau0qtg / very long 32tka2n9 / long 22m0pigr / mid bqoy698f / short 23rgsozp
# THIS WORKS LESS WELL: 1zh8fqdf
# dirs = find_checkpoints(Path(f"{prefix}hummingbirds-pil"), version="24zruk7z", log="last")#.glob("**/*.ckpt"))

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

# torch.set_grad_enabled(False)   
model.eval()

print(f"Positive data dir: {model.pos_data_dir}, negative data dir: {model.neg_data_dir}")
# %%mitch85

vname = "FH709_01" 

# ANNOTATED VIDEOS AVAILABLE: 
# dataloader = model.tst_external_dataloader(path=f"/data/shared/frame-diff-anomaly/data/{vname}/")

# FH102_02 FH107_01 FH108_01 FH109_02 FH202_02 FH207_01 FH207_02 FH303_01 FH303_02 FH308_01 
# FH402_01 FH402_02 FH403_01 FH403_02 FH408_01 FH408_02 FH502_01 FH502_02 FH503_01 FH507_01 
# FH507_02 FH508_01 FH508_02 FH509_01 FH509_02 FH602_01 FH602_02 FH603_01 FH603_02 FH604_01 
# FH608_01 FH703_02 FH706_01 FH707_01 FH707_02 FH802_01 FH802_02 FH803_01 

## Non Annotated videos: 
dataloader = model.tst_external_dataloader(path=f"/data/shared/frame-diff-anomaly/data/no_annotation/{vname}/")
# --- FH709_01 (HummingbirdParty), FH709_01 (HummingbirdParty 2), FH805_01 (good example with clutter)
# detected Events: FH101_01, +FH104_01, FH104_BU1, FH105_01 (butterfly? bird?), FH105_02, FH106_01, FH111_01 (1 event), 
# FH201_01 (1 event), FH205_01 (at least 3 events, then sun reflection), FH205_02, FH209_01, FH304_01, FH304_02, FH304_BU1, 
# FH304_BU2, FH306_01, FH312_01, FH404_01, FH404_02, FH409_01, FH409_02, FH410_01, FH410_02, FH411_01, FH411_02, FH511_01, 
# FH511_02, (FH512_01), (FH512_02), FH601_HR_01, FH607_02, FH611_01, +FH704_01, FH708_02, FH801_01, FH801_02, FH804_BU1, FH805_01, 
# FH808_02


# CHECK DEL FOR THOSE
# unclear (TO BE DELETED): FH104_BU2 (only persons prob), FH105_03 (a dog?), FH110_01 (at least one, but complex video _02 similarly hard), 
# FH204_BU1 (hard, small birds in few frames), FH211_02 (maybe a couple), FH301_02 (one big, then dark), 
# FH310_02 (maybe one), FH311_01 (maybe one), FH501_01 (not sure is a bird, 1-2 frames), FH505_01 (probably nothing), 
# FH505_02 (drop of watr), FH510_02 (1 visit), FH606_01 (1-2 visits), FH607_01 (1 visit), FH701_HR_01 (1 vis), 
# FH701_HR_02 (1 vis, large bird :)), FH705_01 (1 vis), FH705_02 (1 vis), FH708_01 (1 vis), FH710_02 (1 vis), FH801_03, 
# FH804_BU2 (uno zoccolo di capra?), FH809_01 (Maybe a couple but lots of wind), FH810_01 (1 vis + unclear glowing orb)

# no events (rm -r): FH101_02, FH106_02, FH109_01, FH111_02, FH112_01, FH112_02, FH201_02, FH201_03, 
# FH204_01, FH204_BU2, FH206_01, FH209_02, FH211_01, FH301_01, FH305_02, FH306_02, FH310_01, FH311_02,
# FH312_02, FH401_01, FH401_02, FH405_01, FH405_02, FH501_02, FH506_01 (rain), FH506_02 (rain), FH610_01, 
# FH610_02, FH611_02, FH704_02, FH704_BU1, FH704_BU2, FH804_01, FH804_02, FH805_02, FH808_01, FH809_02

# %% 
pbar_cb = pl.callbacks.progress.TQDMProgressBar(refresh_rate=5)

trainer = pl.Trainer(
    max_epochs=1,
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

# if vname: 
#     gc[annot[annot.Truth].Frame] = 1

plt.figure()
plt.plot(pc[:,1])
# if vname: 
    # plt.scatter(annot[~annot.Truth].Frame, -0*(np.ones(len(annot[~annot.Truth].Frame))), marker="^", color="k")
    # plt.scatter(annot[annot.Truth].Frame, -0*(np.ones(len(annot[annot.Truth].Frame))), marker="^", color="r")
plt.grid(True)

# %% 
if 1:
    
    # from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, 
    # AblationCAM, XGradCAM, EigenCAM, FullGrad

    from pytorch_grad_cam import GradCAMPlusPlus as CAM
    from pytorch_grad_cam.utils.image import *
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    # torch.set_grad_enabled(False)   
    denorm = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    for param in model.model.parameters():
            param.requires_grad = True

    # cam = CAM(model=model.model, target_layers=[model.model.layer4[-1]], use_cuda=False)
    cam = CAM(model=model.model, target_layers=[model.model.features[-1]], use_cuda=False)

    files = dataloader.dataset.img_paths.copy()

    inc = 1

    # score = pc[:,inc]
    if 1: #vname:
        score = np.concatenate(([0], np.abs(np.diff(pc[:,inc]))))
    else: 
        score = pc[:,inc]

    FR_F = 0; CLIP = len(score)
    sortsco = (0.5*score + 0.5*pc[:,inc])[FR_F:CLIP]
    yy = np.argsort(-sortsco)

    sub_ind = (score[yy] > 0)# & (gc[yy] == 1)
    ss = yy[sub_ind] + FR_F# + 1 if score is the diff! starts at 1 and not at 0

    plt.figure(figsize=(15,5))
    plt.title(f"Proba time series, with unkown N_positive")
    plt.plot(sortsco, label="agg_score")
    plt.plot(0.5*pc[FR_F:CLIP,1], label="pc")
    plt.plot(0.5*score[FR_F:CLIP], label="diff")
    plt.legend(loc="lower right")
    # if vname: 
    #     plt.scatter(annot[~annot.Truth].Frame - FR_F, 1*(np.ones(len(annot[~annot.Truth].Frame))), marker="v", color="k")
    #     plt.scatter(annot[ annot.Truth].Frame - FR_F, 1*(np.ones(len(annot[annot.Truth].Frame))), marker="v", color="r")
    plt.grid(True)
    
    DETECT = 0
    DIFF = True
    FR = 0; N = 20
    for i, ti in enumerate(ss):
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
        pl_im_pre = np.transpose(denorm(x_.squeeze()),(1,2,0)).numpy()
        
        im_ = Image.open(files[ti+1]).convert("RGB")
        x_ = model.transform_ts(im_)
        pl_im_pos = np.transpose(denorm(x_.squeeze()),(1,2,0)).numpy()

        # x_g = torch.clone(x)
        x.requires_grad = True
        with torch.set_grad_enabled(True):
            gcam = [cam(input_tensor=x, targets=[ClassifierOutputTarget(cl)]).squeeze() for cl in [0,1]]

        with torch.set_grad_enabled(False):
            p = torch.softmax(model(x), dim = 1).cpu().numpy()
            pl_im = np.transpose(denorm(x.squeeze()),(1,2,0)).numpy()

        f, a = plt.subplots(1,6,figsize=(17,4))
        for c in range(2):
            g_viz = show_cam_on_image(pl_im, gcam[c], use_rgb=True)
            a[c].imshow(g_viz)
            a[c].set_title(f"CAM for Class {c}")
            # a[].imshow(g_viz)
        a[2].imshow(pl_im_pre)
        a[2].set_title(f"Pre-image: GT {gc[ti-1]}")
        a[3].imshow(pl_im)
        a[3].set_title(f"Inference: GT {gc[ti]}")
        a[4].imshow(pl_im_pos)
        a[4].set_title(f"Post-image: GT {gc[ti+1]}")
        
        if DIFF:
            im_0 = exposure.match_histograms(pl_im_pre, pl_im, channel_axis=2)
            im_2 = exposure.match_histograms(pl_im_pos, pl_im, channel_axis=2)

            d1 = pl_im - im_0
            d2 = im_2 - pl_im
            dh = (1 + d1 - d2) / 2
            dh -= dh.min(); dh /= dh.max()
            a[5].imshow(dh)

        plt.suptitle(f"{i}, {ti}: yhat {int(yc[ti])}, gt {int(np.any((gc[ti],gc[ti-1],gc[ti+1])))} ({int(gc[ti])}, {np.sum((gc[ti],gc[ti-1],gc[ti+1]))}), p0 {pc[ti,0]:.2f}, p1 {pc[ti,1]:.2f} \n online p {p[0,1]:.5f}, diff score {score[ti]:.2f} \n{fi.name}")
        plt.show()

        DETECT += int(np.any((gc[ti],gc[ti-1],gc[ti+1])))

        if i > FR+N:
            break

print(f"got {DETECT} in {N} frames")
# %%
