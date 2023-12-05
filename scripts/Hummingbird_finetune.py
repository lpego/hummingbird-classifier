# %%
import os, sys

os.environ["MKL_THREADING_LAYER"] = "GNU"

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning import Trainer  # , LightningModule

try:
    __IPYTHON__
except:
    prefix = ""  # or "../"
else:
    prefix = "../"  # or "../"

sys.path.append(f"{prefix}src")
# from utils import read_pretrained_model
# from HummingbirdLoader import HeronLoader, Denormalize
from HummingbirdModel import HummingbirdModel

# %%
if __name__ == "__main__":
    # scripts/Lit_hummingbird_finetune.py --batch_size=185 --data_dir=data/bal_cla_diff_loc_all_vid/
    # --learning_rate=0.00010856749693422446
    # --num_workers_loader=20 --pretrained_network=resnet18

    # Define checkpoints callbacks
    # best model on validation
    best_val_cb = pl.callbacks.ModelCheckpoint(
        filename="best-val-{epoch}-{step}-{val_loss:.1f}",
        monitor="val_loss",
        mode="min",
        save_top_k=2,
    )

    # latest model in training
    last_mod_cb = pl.callbacks.ModelCheckpoint(
        filename="last-{step}", every_n_train_steps=500, save_top_k=1
    )

    # Define progress bar callback
    pbar_cb = pl.callbacks.progress.TQDMProgressBar(refresh_rate=5)

    # %%
    model = HummingbirdModel(
        pos_data_dir=f"{prefix}data/bal_cla_diff_loc_all_vid/",  # bal_cla_diff_loc_all_vid/", "double_negs_bal_cla_diff_loc_all_vid/"
        neg_data_dir=f"{prefix}data/plenty_negs_all_vid/",  # plenty_negs_all_vid/", # bal_cla_diff_loc_all_vid/", "double_negs_bal_cla_diff_loc_all_vid/"
        pretrained_network="densenet161",  # resnet50
        learning_rate=2.5e-7,  # 1e-6
        batch_size=64,  # 128
        weight_decay=1e-8,  # 1e-3
        num_workers_loader=16,
        step_size_decay=20,
    )

    # %%
    name_run = "asymmetric_data_augm_very_long"  # f"{model.pretrained_network}"
    cbacks = [pbar_cb, best_val_cb, last_mod_cb]
    wb_logger = WandbLogger(
        project="hummingbirds-pil", name=name_run if name_run else None
    )
    wb_logger.watch(model, log="all")
    # TensorBoardLogger("tb_logs", name="")

    trainer = Trainer(
        gpus=-1,  # [0,1],
        max_epochs=100,
        strategy=DDPStrategy(find_unused_parameters=False),
        precision=16,
        callbacks=cbacks,
        auto_lr_find=False,  #
        auto_scale_batch_size=False,
        logger=wb_logger,
        replace_sampler_ddp=False
        # profiler="simple",
    )

    trainer.fit(model)
