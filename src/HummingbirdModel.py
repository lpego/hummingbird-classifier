# %%
# import os, sys, time, copy
# import numpy as np

from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any

# import datetime

import torch

torch.hub.set_dir("./models/")

import pytorch_lightning as pl

# import numpy as np

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from torchmetrics import F1Score, Precision, Recall
from torchvision import transforms

# try:
#     __IPYTHON__
# except:
#     prefix = ""  # or "../"
# else:
#     prefix = "../"  # or "../"

# sys.path.append(f"{prefix}src")

from src.utils import read_pretrained_model

# from ptv_transforms import transforms as trans

from src.HummingbirdLoader import (
    HummingbirdLoader,
    BlurImagePart,
    AddLightHazePart,
    CustomCrop,
)
from src.utils import Denormalize


class HummingbirdModel(pl.LightningModule):
    """
    PyTorch Lightning module for hummingbird classification.

    This model uses a pretrained CNN backbone for binary classification of hummingbird presence
    in images. Supports data augmentation, weighted sampling, and custom transforms.

    Args:
        pos_data_dir: Path to directory containing positive samples
        pretrained_network: Name of pretrained model architecture
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        weight_decay: Weight decay for optimizer
        num_workers_loader: Number of workers for data loading
        step_size_decay: Patience for learning rate scheduler
    """

    def __init__(
        self,
        pos_data_dir: str = "data/",
        # neg_data_dir="data/",
        pretrained_network: str = "resnet50",
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        weight_decay: float = 1e-8,
        num_workers_loader: int = 4,
        step_size_decay: int = 5,
    ):
        super().__init__()

        # Set our init args as class attributes
        self.pos_data_dir = pos_data_dir
        self.neg_data_dir = pos_data_dir
        self.learning_rate = learning_rate
        self.architecture = pretrained_network
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers_loader = num_workers_loader
        self.step_size_decay = step_size_decay

        # Hardcode some dataset specific attributes
        self.num_classes = 2
        self.size_im = 224
        self.dims = (3, self.size_im, self.size_im)
        # channels, width, height = self.dims

        self.transform_tr_p = transforms.Compose(
            [
                CustomCrop((100, 1, 1180, 700), p=1.0),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=10),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
                transforms.RandomAdjustSharpness(sharpness_factor=2),
                transforms.ColorJitter(
                    brightness=[0.8, 1.2], contrast=[0.8, 1.2]
                ),  # (brightness=[0.75, 1.25], contrast=[0.75, 1.25]), # was 0.8, 1.5
                transforms.Resize(
                    (self.size_im, self.size_im),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                # BlurImagePart(size=self.size_im, box_s=((20, 150), (10, 100)), p=0.2),
                # AddLightHazePart(
                #     size=self.size_im,
                #     box_s=((20, 150), (10, 100)),
                #     box_blur_rad=(30, 70),
                #     p=0.2,
                # ),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.transform_tr_n = transforms.Compose(  # = self.transform_tr_p
            [
                CustomCrop((100, 1, 1180, 700), p=1.0),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=10),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
                transforms.RandomAdjustSharpness(sharpness_factor=2),
                transforms.ColorJitter(
                    brightness=[0.8, 1.2], contrast=[0.8, 1.2]
                ),  # (brightness=[0.75, 1.25], contrast=[0.75, 1.25]), # was 0.8, 1.5
                transforms.Resize(
                    (self.size_im, self.size_im),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                BlurImagePart(size=self.size_im, box_s=((20, 150), (10, 100)), p=0.2),
                AddLightHazePart(
                    size=self.size_im,
                    box_s=((20, 150), (10, 100)),
                    box_blur_rad=(30, 70),
                    p=0.2,
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.transform_ts = transforms.Compose(
            [
                CustomCrop((100, 1, 1180, 700), p=1.0),
                transforms.Resize(
                    (self.size_im, self.size_im),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),  # AT LEAST 224
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # Define PyTorch model
        self.model = read_pretrained_model(self.architecture, self.num_classes)
        self.f1_score = F1Score(task="multiclass", num_classes=self.num_classes)
        self.precision = Precision(
            task="multiclass", num_classes=self.num_classes, average="none"
        )
        self.recall = Recall(
            task="multiclass", num_classes=self.num_classes, average="none"
        )

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Unnormalized logits of shape (batch_size, num_classes)
        """
        return self.model(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step for one batch.

        Args:
            batch: Tuple of (images, labels, indices)
            batch_idx: Index of the current batch

        Returns:
            Training loss for this batch
        """
        x, y, _ = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)

        # precision and recall, F1 score from torchmetrics
        f1_sc = self.f1_score(preds, y)
        prec = self.precision(preds.int(), y.int())
        reca = self.recall(preds.int(), y.int())

        self.log("trn_loss", loss, prog_bar=True, sync_dist=True)
        self.log("trn_prec", prec[1], prog_bar=False, sync_dist=True)
        self.log("trn_reca", reca[1], prog_bar=False, sync_dist=True)
        self.log("trn_f1", f1_sc, prog_bar=False, sync_dist=True)
        self.log("learning_rate", self.learning_rate, prog_bar=True, sync_dist=False)
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        print_log: str = "val",
    ) -> torch.Tensor:
        """
        Validation step for one batch.

        Args:
            batch: Tuple of (images, labels, indices)
            batch_idx: Index of the current batch
            print_log: Prefix for logging metrics

        Returns:
            Validation loss for this batch
        """
        x, y, _ = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)

        f1_sc = self.f1_score(preds, y)
        prec = self.precision(preds.int(), y.int())
        reca = self.recall(preds.int(), y.int())

        self.log(f"{print_log}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{print_log}_acc", f1_sc, prog_bar=True, sync_dist=True)
        self.log(f"{print_log}_prec", prec[1], prog_bar=False, sync_dist=True)
        self.log(f"{print_log}_reca", reca[1], prog_bar=False, sync_dist=True)
        self.log(
            f"{print_log}_pred_counts",
            preds.sum().float(),
            prog_bar=False,
            sync_dist=True,
        )

        return loss

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        print_log: str = "tst",
    ) -> torch.Tensor:
        """
        Test step for one batch.

        Args:
            batch: Tuple of (images, labels, indices)
            batch_idx: Index of the current batch
            print_log: Prefix for logging metrics

        Returns:
            Test loss for this batch
        """
        # Reuse the validation_step for testing
        return self.validation_step(batch, batch_idx, print_log)

    def predict_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prediction step for one batch.

        Args:
            batch: Tuple of (images, labels, indices)
            batch_idx: Index of the current batch
            dataloader_idx: Index of the dataloader (for multiple dataloaders)

        Returns:
            Tuple of (predictions, probabilities, true_labels)
        """
        x, y, _ = batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds, probs, y

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizer and learning rate scheduler.

        Returns:
            Dictionary containing optimizer and scheduler configuration
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.step_size_decay,
            cooldown=0,
            factor=0.1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    ######################
    # DATA RELATED HOOKS #
    ######################

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        """
        Create training dataloader with weighted sampling.

        Args:
            shuffle: Whether to use weighted random sampling (True) or sequential sampling

        Returns:
            DataLoader for training data
        """
        dir_dict_trn = {
            "negatives": [
                Path(self.neg_data_dir) / "trn_set/class_0",
                Path(self.neg_data_dir) / "val_set/class_0",
            ],
            "positives": [
                Path(self.pos_data_dir) / "trn_set/class_1",
                Path(self.pos_data_dir) / "val_set/class_1",
            ],
        }

        trn_d = HummingbirdLoader(
            dir_dict_trn,
            learning_set="trn",
            ls_inds=[],
            transforms={  # self.transform_tr_n,
                "0": self.transform_tr_p,
                "1": self.transform_tr_p,
            },  # can load two sets of transforms, one for positives one for negatives
        )

        print(len(trn_d), self.neg_data_dir, self.pos_data_dir)

        # number of draws from the weighted random samples matches the 2 * (n_positive // batch_size)
        if not shuffle:
            return DataLoader(
                trn_d,
                batch_size=self.batch_size,
                shuffle=shuffle,
                drop_last=True,
                num_workers=self.num_workers_loader,
            )

        else:  # means weighted random sampling
            # make the number of draws comparable to a full sweep over a balanced set.
            # Should just train for longer, but "epochs" are now shorter
            num_samples = (trn_d.labels == 1).sum().item()

            weights = []
            for c in [0, 1]:
                weights.extend(
                    len(trn_d.labels[trn_d.labels == c])
                    * [len(trn_d.labels) / (trn_d.labels == c).sum()]
                )

            self.trn_weights = torch.Tensor(weights)
            sampler = WeightedRandomSampler(self.trn_weights, num_samples=num_samples)

            return DataLoader(
                trn_d,
                batch_size=self.batch_size,
                drop_last=True,
                num_workers=self.num_workers_loader,
                sampler=sampler,
            )

    def val_dataloader(self) -> DataLoader:
        """
        Create validation dataloader.

        Returns:
            DataLoader for validation data
        """

        dir_dict_val = {
            "negatives": Path(self.neg_data_dir) / "tst_set/class_0",
            "positives": Path(self.pos_data_dir) / "tst_set/class_1",
        }
        val_d = HummingbirdLoader(
            dir_dict_val, learning_set="val", ls_inds=[], transforms=self.transform_ts
        )

        return DataLoader(
            val_d, batch_size=self.batch_size, num_workers=self.num_workers_loader
        )

    def tst_dataloader(self) -> DataLoader:
        """
        Create test dataloader.

        Returns:
            DataLoader for test data
        """

        dir_dict_tst = {
            "negatives": Path(self.neg_data_dir) / "tst_set/class_0",
            "positives": Path(self.pos_data_dir) / "tst_set/class_1",
        }
        tst_d = HummingbirdLoader(
            dir_dict_tst, learning_set="tst", ls_inds=[], transforms=self.transform_ts
        )

        return DataLoader(
            tst_d, batch_size=self.batch_size, num_workers=self.num_workers_loader
        )

    def tst_external_dataloader(self, path: str, batch_size: int = 16) -> DataLoader:
        """
        Create test dataloader from external data directory.

        All images are loaded as 'negative' samples for convenience to maintain ordering.

        Args:
            path: Path to directory containing test images
            batch_size: Batch size for the dataloader

        Returns:
            DataLoader for external test data
        """

        dir_dict_tst_ex = {"negatives": Path(path), "positives": Path("")}
        tst_ex_d = HummingbirdLoader(
            dir_dict_tst_ex,
            learning_set="tst",
            ls_inds=[],
            transforms=self.transform_ts,
        )

        return DataLoader(
            tst_ex_d,
            batch_size=batch_size,
            num_workers=self.num_workers_loader,
            shuffle=False,
            drop_last=False,
        )


# %%

# self.transform_tr_pos = transforms.Compose(
#     [
#         transforms.RandomHorizontalFlip(p=0.5),
#         # transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
#         # transforms.RandomAdjustSharpness(sharpness_factor=2),
#         # transforms.RandomEqualize(),
#         # transforms.RandomAutocontrast(),
#         transforms.ColorJitter(brightness=[0.8, 1.2], contrast=[0.8, 1.2]),
#         # transforms.RandomGrayscale(p=0.4),
#         transforms.Resize(
#             (int(1 * self.size_im), int(1 * self.size_im)),
#             interpolation=Image.BILINEAR,
#         ),
#         # transforms.RandomCrop(self.size_im, pad_if_needed=True),
#         # transforms.Resize(
#         # (self.size_im, self.size_im), interpolation=Image.BILINEAR
#         # ),  # AT LEAST 224
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ]
# )


# self.test_time_augm = transforms.Compose(
#     [
#         transforms.Resize(
#             (int(2 * self.size_im), int(2 * self.size_im)),
#             interpolation=Image.BILINEAR,
#         ),
#         transforms.TenCrop(self.size_im),
#         transforms.Lambda(
#             lambda crops: torch.stack(
#                 [
#                     transforms.Normalize(
#                         (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
#                     )(transforms.ToTensor()(crop))
#                     for crop in crops
#                 ]
#             )
#         ),
#     ]
# )
