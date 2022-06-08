# %% 
%load_ext autoreload
%autoreload 2

import os, sys, time, copy

os.environ['MKL_THREADING_LAYER'] = 'GNU'

import numpy as np
from pathlib import Path
from PIL import Image
# import datetime

import torch
import pytorch_lightning as pl
# from pytorch_lightning import Trainer
# from torch import nn
# from torch.nn import functional as F
from torch.utils.data import DataLoader
# from torchmetrics import Accuracy, F1Score, ConfusionMatrix
from torchvision import transforms
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
from HummingbirdLoader_v2 import HummingbirdLoader, Denormalize
from HummingbirdLitModel import HummingbirdModel

# %% 
dirs = find_checkpoints(Path(f"{prefix}lightning_logs"), version="version_8", log="last")#.glob("**/*.ckpt"))

mod_path = dirs[-1]

model = HummingbirdModel()

model = model.load_from_checkpoint(
        checkpoint_path=mod_path,
        hparams_file= str(mod_path.parents[1] / 'hparams.yaml') #same params as args
    )

model.data_dir= f"{prefix}{model.data_dir}"
# model.pretrained_network="resnet50"
# model.learning_rate=1e-7
# model.batch_size=128
# model.weight_decay=0

torch.set_grad_enabled(False)   
model.eval()

print(model.data_dir)
# %% 
dataloader = model.val_dataloader()
# dataloader = model.tst_dataloader()
# dataloader = model.train_dataloader(shuffle=False)
# dataloader = model.tst_external_dataloader(path="/data/shared/frame-diff-anomaly/data/FH502_02/")

# FH102_02  FH109_02  FH207_02  FH308_01  FH403_01  FH408_02  FH503_01  FH508_01  FH509_02  FH603_01  FH608_01  FH707_01  FH802_02 FH107_01  FH202_02  FH303_01  FH402_01  FH403_02  FH502_01  FH507_01  FH508_02  FH602_01  FH603_02  FH703_02  FH707_02  FH803_01 FH108_01  FH207_01  FH303_02  FH402_02  FH408_01  FH502_02  FH507_02  FH509_01  FH602_02  FH604_01  FH706_01  FH802_01

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

plt.plot(pc[:,1])
# %%
if 1:
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
    # this needs to be ordered
    files = dataloader.dataset.img_paths.copy()

    yy = np.argsort(-pc[:,1])

    sub_ind = (pc[yy,0] > 0.75) & (gc[yy] == 0)
    # sub_ind = (pc[yy,1] > 0.7)

    # sub_ind = np.where(sub_ind)[0]
    # files = files[sub_ind]  
    ss = yy[sub_ind]  
    # sub_ind=np.arange(pc.shape[0])
    # sub_p = pc[:,1]
    # yy = np.argsort(-sub_p)

    denorm = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    for i, ti in enumerate(ss):
        if i < 0:
            continue
        fi = files[ti] 
        x = Image.open(fi).convert("RGB")
        plt.figure(figsize=(6,6))
        plt.imshow(x)
        plt.title(f"{i}, {ti}: yhat {yc[ti]:.2f}, gt {gc[ti]:.2f}, p0 {pc[ti,0]:.2f}, p1 {pc[ti,1]:.2f}\n{fi.name}")
        plt.show()

        if i > 100:
            break

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
# pso = np.argsort(-pc[:,1])
# prob_sort = pc[pso, :]

# plt.figure()
# plt.plot(gc[pso])
# plt.plot(prob_sort[:,1] - prob_sort[:,0])

# plt.figure()
# plt.scatter(prob_sort[:,1], gc[pso])
# %%
# %%
