# %%
# %load_ext autoreload
# %autoreload 2

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

import torch

torch.hub.set_dir("/data/shared/hummingbird-classifier/models/")

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

sys.path.append(".")
from src.HummingbirdModel import HummingbirdModel
from src.utils import cfg_to_arguments, SaveLogCallback

# Set the thread layer used by MKL
os.environ["MKL_THREADING_LAYER"] = "GNU"


# %%
def main(args, cfg):
    """
    Function to train a model for classification of macrozoobenthos images.
    The model is trained on the dataset specified in the config file, saved to the folder specified every 50 steps and at the end of the training.

    Parameters
    ----------
    args : argparse.Namespace
        Namespace containing the arguments passed to the script. Notably:

            - input_dir: path to the directory containing the images to be classified
            - save_model: path to the directory where the model will be saved
            - config_file: path to the config file with train / inference parameters

    cfg : dict
        Dictionary containing the configuration parameters.

    Returns
    -------
    None. Saves the model in the specified folder.
    """

    # Define checkpoints callbacks
    # best model on validation
    best_val_cb = pl.callbacks.ModelCheckpoint(
        dirpath=args.save_model,
        filename="best-val-{epoch}-{step}-{val_loss:.1f}",
        monitor="val_loss",
        mode="min",
        save_top_k=cfg.trcl_save_topk,
    )

    # latest model in training
    last_mod_cb = pl.callbacks.ModelCheckpoint(
        dirpath=args.save_model,
        filename="last-{step}",
        every_n_train_steps=50,
        # save_top_k=cfg.trcl_save_topk
    )

    # Define progress bar callback
    pbar_cb = pl.callbacks.progress.TQDMProgressBar(refresh_rate=5)

    # Define logger callback to log training date
    trdatelog = SaveLogCallback(model_folder=args.save_model)

    # Define model from config
    model = HummingbirdModel(
        pos_data_dir=args.input_dir,
        pretrained_network=cfg.trcl_model_pretrarch,
        learning_rate=cfg.trcl_learning_rate,
        batch_size=cfg.trcl_batch_size,
        weight_decay=cfg.trcl_weight_decay,
        num_workers_loader=cfg.trcl_num_workers,
        step_size_decay=cfg.trcl_step_size_decay,
    )

    # Check if there is a model to load, if there is, load it and train from there
    if False:  # majke this a flag
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

    # Define logger and name of run
    name_run = (
        f"classifier-{args.save_model.parents[0].name}"  # f"{model.pretrained_network}"
    )
    cbacks = [pbar_cb, best_val_cb, last_mod_cb, trdatelog]

    # Define logger, and use either wandb or tensorboard
    if cfg.trcl_logger == "wandb":
        logger = WandbLogger(
            project=cfg.trcl_wandb_project_name, name=name_run if name_run else None
        )
        logger.watch(model, log="all")

    elif cfg.trcl_logger == "tensorboard":
        logger = TensorBoardLogger(
            save_dir=args.save_model,
            name=name_run if name_run else None,
            log_graph=True,
        )

    # instantiate trainer and train
    trainer = pl.Trainer(
        accelerator="auto",  # cfg.trcl_num_gpus outdated
        max_epochs=cfg.trcl_number_epochs,
        strategy=DDPStrategy(
            find_unused_parameters=False
        ),  # TODO: check how to use in notebook
        callbacks=cbacks,
        logger=logger,
        log_every_n_steps=1
        # profiler="simple",
    )

    trainer.fit(model)


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

# %%
