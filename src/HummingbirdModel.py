# %%
# import os, sys, time, copy
# import numpy as np

from pathlib import Path
from PIL import Image

# import datetime

import torch
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from torchmetrics import F1Score
from torchmetrics.functional import precision_recall
from torchvision import transforms

# try:
#     __IPYTHON__
# except:
#     prefix = ""  # or "../"
# else:
#     prefix = "../"  # or "../"

# sys.path.append(f"{prefix}src")

from utils import read_pretrained_model

# from ptv_transforms import transforms as trans

from HummingbirdLoader import (
    HummingbirdLoader,
    Denormalize,
    BlurImagePart,
    AddLightHazePart,
    CustomCrop,
)


# %%
class HummingbirdModel(pl.LightningModule):
    """
    pytorch lightning class def and model setup
    """

    def __init__(
        self,
        pos_data_dir="data/",
        neg_data_dir="data/",
        pretrained_network="resnet50",
        learning_rate=1e-4,
        batch_size=32,
        weight_decay=1e-8,
        num_workers_loader=4,
        step_size_decay=5,
    ):
        super().__init__()

        # Set our init args as class attributes
        self.pos_data_dir = pos_data_dir
        self.neg_data_dir = neg_data_dir
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
        self.accuracy = F1Score()

        self.save_hyperparameters()

    def forward(self, x):
        "forward pass return unnormalised logits, normalise when needed"
        return self.model(x)

    def training_step(self, batch, batch_idx):
        "training iteration per batch"
        x, y, _ = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)  # , weight=torch.Tensor((1,2)).to("cuda"))
        # loss = F.nll_loss(logits, y, weight=torch.Tensor((1,2)).to("cuda"))
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)
        # only take them for positive class. Maybe torchmetrics for that is not good
        (self.precision, self.recall) = precision_recall(
            preds.int(), y.int(), num_classes=2, average=None
        )

        self.log("trn_loss", loss, prog_bar=True)
        self.log("trn_acc", self.accuracy, prog_bar=False)
        self.log("trn_prec", self.precision[1], prog_bar=False)
        self.log("trn_reca", self.recall[1], prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx, print_log: str = "val"):
        "validation iteration per batch"
        x, y, _ = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)  # , weight=torch.Tensor((1,2)).to("cuda"))
        # loss = F.nll_loss(logits, y, weight=torch.Tensor((1,2)).to("cuda"))
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)
        (self.precision, self.recall) = precision_recall(
            preds.int(), y.int(), num_classes=2, average=None
        )

        self.log(f"{print_log}_loss", loss, prog_bar=True)
        self.log(f"{print_log}_acc", self.accuracy, prog_bar=True)
        self.log(f"{print_log}_prec", self.precision[1], prog_bar=False)
        self.log(f"{print_log}_reca", self.recall[1], prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx, print_log: str = "tst"):
        "test iteration per batch"
        # Reuse the validation_step for testing
        return self.validation_step(batch, batch_idx, print_log)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = None):
        x, y, _ = batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds, probs, y

    def configure_optimizers(self):
        "optimiser config plus lr scheduler callback"
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size_decay, gamma=0.5
        )
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        # optimizer, milestones=[10, 20, 50], gamma=0.5
        # )

        return [optimizer], [lr_scheduler]

    # figure out how Plateau scheduler could work when val fits are too good.
    # def configure_optimizers(self):
    #     "optimiser config plus lr scheduler callback"
    #     optimizer = torch.optim.AdamW(
    #         self.model.parameters(),
    #         lr=self.learning_rate,
    #         weight_decay=self.weight_decay,
    #     )
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=1, cooldown=1, factor=0.1),
    #             "monitor": "val_acc",
    #             "frequency": 1
    #         },
    #     }

    ######################
    # DATA RELATED HOOKS #
    ######################

    def train_dataloader(self, shuffle=True):
        "def of custom training dataloader"
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
                "0": self.transform_tr_n,
                "1": self.transform_tr_p,
            },  # can load two sets of transforms, one for positives one for negatives
        )

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

    def val_dataloader(self):
        "def of custom val dataloader"

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

    def tst_dataloader(self):
        "def of custom test dataloader"

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

    def tst_external_dataloader(self, path):
        """
        def of test dataloader from external data. All loaded as `negative` samples,
        but just for convenience to maintain ordering"
        e.g. /data/shared/frame-diff-anomaly/data/FH102_02/
        """

        dir_dict_tst_ex = {"negatives": Path(path), "positives": Path(".")}
        tst_ex_d = HummingbirdLoader(
            dir_dict_tst_ex,
            learning_set="tst",
            ls_inds=[],
            transforms=self.transform_ts,
        )

        return DataLoader(
            tst_ex_d, batch_size=self.batch_size, num_workers=self.num_workers_loader
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
