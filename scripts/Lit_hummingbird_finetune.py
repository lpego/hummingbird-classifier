# %%
# %load_ext autoreload
# %autoreload 2

import os, sys, time, copy

os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np

from pathlib import Path
from PIL import Image
import datetime

# from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies.ddp import DDPStrategy

import torch
import pytorch_lightning as pl

from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
from torchvision import transforms

try:
    __IPYTHON__
except:
    prefix = ""  # or "../"
else:
    prefix = "../"  # or "../"

sys.path.append(f"{prefix}src")
from utils import read_pretrained_model

# from HummingbirdLoader import HeronLoader, Denormalize
from HummingbirdLitModel import HummingbirdModel

# %%
if __name__ == "__main__":

    # Define checkpoints callbacks
    # best model on validation
    best_val_cb = pl.callbacks.ModelCheckpoint(
        filename="best-val-{epoch}-{step}-{val_loss:.1f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )

    # latest model in training
    last_mod_cb = pl.callbacks.ModelCheckpoint(
        filename="last-{step}", every_n_train_steps=500, save_top_k=1
    )

    # Define progress bar callback
    pbar_cb = pl.callbacks.progress.TQDMProgressBar(refresh_rate=5)

    # %%
    model = HummingbirdModel(
        data_dir=f"{prefix}data/balanced_classes_different_locations/",
        pretrained_network="vit16",
        learning_rate=1e-5,  # was 5 in v6
        batch_size=128,
        weight_decay=1e-8,
        num_workers_loader=16,
    )

    # %%

    cbacks = [pbar_cb, best_val_cb, last_mod_cb]
    trainer = Trainer(
        gpus=-1,  # [0,1],
        max_epochs=75,
        strategy=DDPStrategy(find_unused_parameters=False),
        precision=16,
        callbacks=cbacks,
        auto_lr_find=False,  # change it for new _v6 or _v7
        auto_scale_batch_size=True,
        # profiler="simple",
    )

    trainer.fit(model)
