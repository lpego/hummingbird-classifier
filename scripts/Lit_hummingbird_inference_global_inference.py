# %% 
# %load_ext autoreload
# %autoreload 2

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

from HummingbirdLoader_v2 import Denormalize # HummingbirdLoader, 
from HummingbirdLitModel import HummingbirdModel        

# %% 
# EXPERIMENT = "DA_COMPLETE"
EXPERIMENT = "DA_ASYMM"

top_K = [25, 50, 100, 250, 500, 1000, 5000, 10000]

# %% 

out_csv_pred = Path(f"{prefix}data/pred_csv/{EXPERIMENT}/")
out_csv_pred.mkdir(exist_ok=True, parents=True)

files = sorted(list(Path(f"{prefix}data/pred_csv/{EXPERIMENT}/").glob("*")))
files = [v for v in files if "no_annotation" not in str(v)]

# %% 
# ANNOTATED VIDEOS AVAILABLE
# FH102_02 FH107_01 FH108_01 FH109_02 FH202_02 FH207_01 FH207_02 FH303_01 FH303_02 FH308_01 
# FH402_01 FH402_02 FH403_01 FH403_02 FH408_01 FH408_02 FH502_01 FH502_02 FH503_01 FH507_01 
# FH507_02 FH508_01 FH508_02 FH509_01 FH509_02 FH602_01 FH602_02 FH603_01 FH603_02 FH604_01 
# FH608_01 FH703_02 FH706_01 FH707_01 FH707_02 FH802_01 FH802_02 FH803_01 

cmat_score_sc = np.zeros((2,2,len(top_K)))
cmat_score_pr = np.zeros((2,2,len(top_K)))
cmat_score_tr = np.zeros((2,2,len(top_K)))

for fpath in files[:2]: 
    # print(f"processing {fpath}")
    
    df = pd.read_csv(fpath)
    df = df.drop(columns=["Unnamed: 0"])

    sort_score_frames = np.argsort(-df["score_pipeline"])
    grtr_sc_srt = df["groundtruth"].iloc[sort_score_frames].values

    sort_proba_frames = np.argsort(-df["probability"])
    grtr_pr_srt = df["groundtruth"].iloc[sort_proba_frames].values
    
    sort_tripl_frames = np.argsort(-df["score_t_diff"])
    grtr_tr_srt = df["groundtruth"].iloc[sort_tripl_frames].values

    for i, k in enumerate(top_K):
        
        temp_cmat_score_sc = np.zeros((2,2))
        temp_cmat_score_sc[1,1] = np.sum(grtr_sc_srt[:k] == 1) # True positives
        temp_cmat_score_sc[1,0] = np.sum(grtr_sc_srt[:k] == 2) # False positives
        temp_cmat_score_sc[0,1] = np.sum(grtr_sc_srt[k:] == 1) # False negatives
        temp_cmat_score_sc[0,0] = np.sum(grtr_sc_srt[k:] == 2) # True negatives
        
        cmat_score_sc[:,:,i] += temp_cmat_score_sc

        temp_cmat_score_pr = np.zeros((2,2))
        temp_cmat_score_pr[1,1] = np.sum(grtr_pr_srt[:k] == 1) # True positives
        temp_cmat_score_pr[1,0] = np.sum(grtr_pr_srt[:k] == 2) # False positives
        temp_cmat_score_pr[0,1] = np.sum(grtr_pr_srt[k:] == 1) # False negatives
        temp_cmat_score_pr[0,0] = np.sum(grtr_pr_srt[k:] == 2) # True negatives
        
        cmat_score_pr[:,:,i] += temp_cmat_score_pr

        temp_cmat_score_tr = np.zeros((2,2))
        temp_cmat_score_tr[1,1] = np.sum(grtr_tr_srt[:k] == 1) # True positives
        temp_cmat_score_tr[1,0] = np.sum(grtr_tr_srt[:k] == 2) # False positives
        temp_cmat_score_tr[0,1] = np.sum(grtr_tr_srt[k:] == 1) # False negatives
        temp_cmat_score_tr[0,0] = np.sum(grtr_tr_srt[k:] == 2) # True negatives
        
        cmat_score_tr[:,:,i] += temp_cmat_score_tr
    
print("Proba:")
print(cmat_score_pr)

print("Score:")
print(cmat_score_sc)

print("Triple:")
print(cmat_score_tr)

with open('v1_conf_mats.npy', 'wb') as f:
    np.save(f, cmat_score_pr)
    np.save(f, cmat_score_sc)
    np.save(f, cmat_score_tr)

F1_s = []; pr_s = []; re_s = []; tnr_s = []
F1_p = []; pr_p = []; re_p = []; tnr_p = []
F1_t = []; pr_t = []; re_t = []; tnr_t = []

for i, k in enumerate(top_K):
    precision = cmat_score_pr[1,1,i] / cmat_score_pr[1,:,i].sum()
    recall = cmat_score_pr[1,1,i] / cmat_score_pr[:,1,i].sum()
    F1 = 2 * (precision * recall) / (precision + recall)
    tnr = cmat_score_pr[1,0,i] / cmat_score_pr[:,0,i].sum()

    F1_p.append(F1)
    pr_p.append(precision)
    re_p.append(recall)
    tnr_p.append(tnr)

    print(f"PROBA top_K: {k} : precision {precision:.2f}, recall {recall:.2f}, F1 {F1:.2f}, TNR {tnr:.2f}")
    del F1, recall, precision, tnr 

    precision = cmat_score_sc[1,1,i] / cmat_score_sc[1,:,i].sum()
    recall = cmat_score_sc[1,1,i] / cmat_score_sc[:,1,i].sum()
    F1 = 2 * (precision * recall) / (precision + recall)
    tnr = cmat_score_sc[1,0,i] / cmat_score_pr[:,0,i].sum()

    F1_s.append(F1)
    pr_s.append(precision)
    re_s.append(recall)
    tnr_s.append(tnr)

    print(f"SCORE top_K: {k} : precision {precision:.2f}, recall {recall:.2f}, F1 {F1:.2f}, TNR {tnr:.2f}")
    del F1, recall, precision, tnr 
    
    precision = cmat_score_tr[1,1,i] / cmat_score_tr[1,:,i].sum()
    recall = cmat_score_tr[1,1,i] / cmat_score_tr[:,1,i].sum()
    F1 = 2 * (precision * recall) / (precision + recall)
    tnr = cmat_score_tr[1,0,i] / cmat_score_tr[:,0,i].sum()
    
    F1_t.append(F1)
    pr_t.append(precision)
    re_t.append(recall)
    tnr_t.append(tnr)
    
    print(f"TRIPL top_K: {k} : precision {precision:.2f}, recall {recall:.2f}, F1 {F1:.2f}, TNR {tnr:.2f}")
    del F1, recall, precision, tnr 

    print()

df = pd.DataFrame(data={"top_K": top_K, 
        "pr_p": pr_p, "re_p": re_p, 
        "F1_p": F1_p, "tnr_p": tnr_p, 
        "pr_s": pr_s, "re_s": re_s,
        "F1_s": F1_s, "tnr_s": tnr_s, 
        "pr_t": pr_t, "re_t": re_t,
        "F1_t": F1_t, "tnr_t": tnr_t})#,
        # index=top_K)

# print("...saving df")
# df.to_csv(f"{EXPERIMENT}_v1_global_scores_toplot.csv")

with plt.xkcd():
    plt.figure()
    plt.title("F1")
    plt.plot(np.arange(len(top_K)), F1_p, label="probabs only")
    plt.plot(np.arange(len(top_K)), F1_t, label="change detection only")
    plt.plot(np.arange(len(top_K)), F1_s, label="composite score")
    plt.xticks(np.arange(len(top_K)), [f"{a:.0f}" for a in  top_K], rotation=90);
    plt.grid("on")
    plt.legend()


    # plt.figure()
    # plt.title("ROC")
    # plt.plot(tnr_p, re_p, label="proba only")
    # plt.plot(tnr_s, re_s, label="comp. score")
    # plt.grid("on")
    # plt.legend()

    plt.figure()
    plt.title("Precision")
    plt.plot(np.arange(len(top_K)), pr_p, label="proba only")
    plt.plot(np.arange(len(top_K)), pr_t, label="change detection only")
    plt.plot(np.arange(len(top_K)), pr_s, label="comp. score")
    plt.xticks(np.arange(len(top_K)), [f"{a:.0f}" for a in  top_K], rotation=90);
    plt.grid("on")
    plt.legend()

    plt.figure()
    plt.title("Recall")
    plt.plot(np.arange(len(top_K)), re_p, label="proba only")
    plt.plot(np.arange(len(top_K)), re_t, label="change detection only")
    plt.plot(np.arange(len(top_K)), re_s, label="comp. score")
    plt.xticks(np.arange(len(top_K)), [f"{a:.0f}" for a in  top_K], rotation=90);
    plt.grid("on")
    plt.legend()
# %%
