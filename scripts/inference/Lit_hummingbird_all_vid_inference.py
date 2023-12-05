# %% 
# %load_ext autoreload
# %autoreload 2

import os, sys #, time, copy

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
EXPERIMENT = "DA_ASYMM"
COMPUTE_TRIPLET = True

# dirs = find_checkpoints(Path(f"{prefix}lightning_logs"), version="version_0", log="last")#.glob("**/*.ckpt"))
# dirs = find_checkpoints(Path(f"{prefix}hummingbirds-pil"), version="ixpfqgvo", log="best")#.glob("**/*.ckpt")) 
# dirs = find_checkpoints(Path(f"{prefix}hummingbirds-pil"), version="2cdynun2", log="best")#.glob("**/*.ckpt")) # Symm DA
# dirs = find_checkpoints(Path(f"{prefix}hummingbirds-pil"), version="3b84ht9l", log="best")#.glob("**/*.ckpt")) # Asym DA
dirs = find_checkpoints(Path(f"{prefix}hummingbirds-pil"), version="3anp5naw", log="best")#.glob("**/*.ckpt")) # Long Tr DA asym


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

videos = sorted(list(Path("/data/shared/frame-diff-anomaly/data/").glob("*")))
videos = [v for v in videos if "no_annotation" not in str(v)]
videos = [v for v in videos if "tool-explore" not in str(v)]

trainer = pl.Trainer(
    max_epochs=1,
    gpus=1, #[0,1],
    # callbacks=[pbar_cb], 
    enable_checkpointing=False,
    logger=False
)

for vname in videos[:]: 
    print(f"Infer {vname}")

    dataloader = model.tst_external_dataloader(path=f"/data/shared/frame-diff-anomaly/data/{vname.name}/")
    annot = pd.read_csv(f"{prefix}data/Weinstein2018MEE_ground_truth.csv").drop_duplicates()
    annot = annot[annot.Video == vname.name].sort_values("Frame")
    annot.Truth = annot.Truth.apply(lambda x: x == "Positive")
    annot.Frame -= 1

    fname = f"{vname.name}.csv"
    fpath = Path(f"{prefix}data/pred_csv/{EXPERIMENT}/")
    fpath.mkdir(exist_ok=True, parents=True)
    
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

    # WATCH THIS NONSENSE OUT! true positives are 1, true negatives are 2
    if vname.name: 
        gc[annot[annot.Truth].Frame] = 1
        gc[annot[~annot.Truth].Frame] = 2

    files = dataloader.dataset.img_paths.copy()
    fpath_tdiff = Path(f"/data/shared/frame-diff-anomaly/data/{vname.name}/_scores_triplet.csv")
    
    if COMPUTE_TRIPLET & (not fpath_tdiff.is_file()):
        score_t_diff = main_triplet_difference(fpath_tdiff.parent, save_csv="triplet")
    elif fpath_tdiff.is_file():
        print(f"{vname.name} triplet loss exists already, loading it.")
        score_t_diff = pd.read_csv(fpath_tdiff)

    score_t_diff = score_t_diff.mag_std.values.astype(float)
    score_t_diff = (score_t_diff - score_t_diff.min())/(score_t_diff.max() - score_t_diff.min())

    score_p_diff = np.concatenate(([0], np.abs(np.diff(pc[:,1]))))
    
    pipeline_score = (0.1*score_p_diff + 0.5*pc[:,1] + 0.4*score_t_diff)

    df_sco = pd.DataFrame(index=range(len(pipeline_score)), columns=["probability", "groundtruth", "score_pipeline", "score_t_diff", "score_p_diff"])
    df_sco["probability"] = pc[:,1]
    df_sco["groundtruth"] = gc
    df_sco["score_pipeline"] = pipeline_score
    df_sco["score_t_diff"] = score_t_diff
    df_sco["score_p_diff"] = score_p_diff
    
    df_sco.to_csv(fpath / fname)
# %%
