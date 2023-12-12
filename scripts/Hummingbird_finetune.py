# %%
import os, sys

os.environ["MKL_THREADING_LAYER"] = "GNU"

import argparse
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning import Trainer  # , LightningModule

# try:
#     __IPYTHON__
# except:
#     prefix = ""  # or "../"
# else:
#     prefix = "../"  # or "../"

sys.path.append(".")
# from utils import read_pretrained_model
# from HummingbirdLoader import HeronLoader, Denormalize
from src.HummingbirdModel import HummingbirdModel

# %%


def train_model(args, cfg):
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
        batch_size=64,
        weight_decay=1e-8,  # 1e-3
        num_workers_loader=16,
        step_size_decay=20,
    )

    # Check if there is a model to load, if there is, load it and train from there
    if args.save_model.exists() and args.save_model.is_dir():
        if args.verbose:
            print(f"Loading model from {args.save_model}")
        try:
            fmodel = list(args.save_model.glob("last-*.ckpt"))[0]
        except:
            print("No last-* model in folder, loading best model")
            fmodel = list(
                args.save_model.glob("best-val-epoch=*-step=*-val_loss=*.*.ckpt")
            )[-1]

        print(f"Loading model from {fmodel}")
        model = model.load_from_checkpoint(fmodel)

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
    return f"model {model.pretrained_network} trained"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="path to config file with per-script args",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="path with images for training",
    )
    parser.add_argument(
        "--save_model",
        type=str,
        required=True,
        help="path to where to save model checkpoints",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="print more info")
    args = parser.parse_args()

    with open(str(args.config_file), "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg = cfg_to_arguments(cfg)

    if args.verbose:
        print(f"main args: {args}")
        print(f"scripts config: {cfg}")

    args.input_dir = Path(args.input_dir)
    args.save_model = Path(args.save_model)
    args.save_model = args.save_model / "checkpoints"

    np.random.seed(cfg.glob_random_seed)  # apply this seed to img tranfsorms
    torch.manual_seed(cfg.glob_random_seed)  # needed for torchvision 0.7
    torch.cuda.manual_seed(cfg.glob_random_seed)  # needed for torchvision 0.7

    sys.exit(main(args, cfg))
