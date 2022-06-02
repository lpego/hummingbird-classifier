# %% 
prefix = "../"
%load_ext autoreload
%autoreload 2

# standard ecosystem
import os, sys, time, copy
import numpy as np
from pathlib import Path
from PIL import Image
import datetime

sys.path.append(f"{prefix}src")

# torchvision
from torchvision import models, transforms

from HummingBirdLoader import HummingBirdLoader, Denormalize
from learning_loops import train_model, visualize_model
from utils import read_pretrained_model

from matplotlib import pyplot as plt

# %%  

model_dir = Path(f"{prefix}/models")
model_list = list(model_dir.glob("*"))
# print(model_list)
# %%
for model in model_list: 
	# print(model)
	for file in model.iterdir():
		if ("learning_curves.dict.npy" in str(file)) and ("different_locations" in str(model)):

			print(file)

			learning_curves = np.load(model / "learning_curves.dict.npy", allow_pickle=True).item()

			n_epochs = len(learning_curves["trn"]["loss"])
			skip = 5
			f, a = plt.subplots(1,2, figsize=(9,3))
			f.suptitle(f"model = {model} \n N_epochs = {n_epochs} \n")
			xtlo = np.arange(0,31,5)
			xtlo[0] = 1
			xtla = [(a) for a in xtlo]

			a[0].plot(learning_curves["trn"]["loss"])
			a[0].plot(learning_curves["val"]["loss"])
			# plt.plot(np.arange(1,31), df_plt["trn_loss"].values)
			# plt.plot(np.arange(1,31), df_plt["val_loss"].values)
			a[0].set_ylabel("Cross-entropy (mean)")
			a[0].set_xticks(xtlo, xtla)
			a[0].set_xlabel("Epochs")

			a[1].plot(learning_curves["trn"]["accuracy"])
			a[1].plot(learning_curves["val"]["accuracy"])
			# plt.plot(np.arange(1,31),df_plt["trn_acc"].values)
			# plt.plot(np.arange(1,31),df_plt["val_acc"].values)
			a[1].set_ylabel("Accuracy [%]")
			a[1].set_xticks(xtlo, xtla)
			a[1].set_xlabel("Epochs");
# %%

# %%
