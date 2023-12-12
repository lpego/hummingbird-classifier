from torchvision import models, transforms

from pathlib import Path
import yaml
import numpy as np

import torch
import torch.nn as nn

from pytorch_lightning import Callback
from datetime import datetime
from skimage import exposure


def read_pretrained_model(architecture, n_class):
    """
    Helper script to load models compactly from pytorch model zoo and prepare them for Hummingbird finetuning

    Parameters
    ----------
    architecture: str
        name of the model to load
    n_class: int
        number of classes to finetune the model for

    Returns
    -------
    model : pytorch model
        model with last layer replaced with a linear layer with n_class outputs
    """

    architecture = architecture.lower()

    if architecture == "vgg":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        in_feat = model.classifier[-1].in_features

        model.classifier[-1] = nn.Linear(
            in_features=in_feat, out_features=n_class, bias=True
        )

        for param in model.features.parameters():
            param.requires_grad = False

        for param in model.classifier.parameters():
            if np.any([a == 2 for a in param.shape]):
                pass
            else:
                param.requires_grad = False

    elif architecture == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(
            in_features=model.fc.in_features, out_features=n_class, bias=True
        )

        # Freeze base feature extraction trunk:
        for param in model.parameters():
            param.requires_grad = True

        for param in model.fc.parameters():
            param.requires_grad = True

    elif architecture == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(
            in_features=model.fc.in_features, out_features=n_class, bias=True
        )

        # Freeze base feature extraction trunk:
        for param in model.parameters():
            param.requires_grad = True

        # for param in model.fc.parameters():
        #     param.requires_grad = True

    elif architecture == "densenet161":
        model = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(
            in_features=model.classifier.in_features, out_features=n_class, bias=True
        )

        for param in model.parameters():
            param.requires_grad = True

        for param in model.classifier.parameters():
            param.requires_grad = True

    elif architecture == "mobilenet":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(
            in_features=model.classifier[1].in_features,
            out_features=n_class,
            bias=True,
        )

        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier[1].parameters():
            param.requires_grad = True

    elif architecture == "efficientnet-b2":
        model = models.efficientnet_b2(
            weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1
        )
        model.classifier[1] = nn.Linear(
            in_features=model.classifier[1].in_features,
            out_features=n_class,
            bias=True,
        )

        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier[1].parameters():
            param.requires_grad = True

    elif architecture == "efficientnet-b1":
        model = models.efficientnet_b1(
            weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1
        )
        model.classifier[1] = nn.Linear(
            in_features=model.classifier[1].in_features,
            out_features=n_class,
            bias=True,
        )

        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier[1].parameters():
            param.requires_grad = True

    elif architecture == "vit16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(
            in_features=model.heads.head.in_features,
            out_features=n_class,
            bias=True,
        )

        for param in model.parameters():
            param.requires_grad = False

        for param in model.heads.head.parameters():
            param.requires_grad = True

    elif architecture == "convnext-small":
        model = models.convnext_small(
            weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1
        )
        model.classifier[2] = nn.Linear(
            in_features=model.classifier[2].in_features, out_features=n_class, bias=True
        )
        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier[2].parameters():
            param.requires_grad = True
    else:
        raise OSError("Model not found")

    return model


def find_checkpoints(dirs=Path("lightning_logs"), type="best"):
    """
    Find the checkpoints of the models in the given directory

    Parameters
    ----------
    dirs: Path
        directory where to look for the checkpoints
    type: str
        type of checkpoints to look for, either "best" or "last"

    Returns
    -------
    chkp: list[Path]
        Path to the model checkpoint, can be mpre than one if there are multiple models in the directory
    """

    ch_sf = sorted(list(dirs.glob(f"checkpoints/*.ckpt")))
    chkp = [a for a in ch_sf if type in str(a.name)]

    return chkp


class cfg_to_arguments(object):
    """
    This class is used to convert a dictionary to an object and extend the argparser.
    In the __init__ method, we iterate over the dictionary and add each key as an attribute to the object.
    Input is a dictionary, output is an object, that mimicks the argparse object.


    Example
    -------
        cfg = {'a': 1, 'b': 2}
        args = cfg_to_arguments(cfg)
        print(args.a) # 1
        print(args.b) # 2

    cfg can be from configs stored in YAML file, a JSON file, or a dictionary, whatever you prefer.
    """

    def __init__(self, args):
        """
        Parameters
        ----------
        args: dict
            dictionary of arguments
        """
        for key in args:
            setattr(self, key, args[key])

    def __str__(self):
        """Prints the object as a string"""
        return self.__dict__.__str__()


class SaveLogCallback(Callback):
    """
    Callback to save the log of the training

    TODO: will need to be updated to save the log of the training in more detail and in a more
    structured way
    """

    def __init__(self, model_folder):
        # super().__init__()
        self.model_folder = model_folder

    # def on_train_start(self, trainer, pl_module):
    #     self.model_folder = self.model_folder / "checkpoints"
    #     # store locally some meta info, if file exists, append to it
    #     # this in each model folder
    #     flog = self.model_folder.parents[0] / "trn_date.yaml"
    #     flag = "a" if flog.is_file() else "w"
    #     with open(self.model_folder.parents[0] / "trn_date.yaml", flag) as f:
    #         yaml.safe_dump(
    #             {"train-date-start": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, f
    #         )

    #     # this is a global file containing all the training dates for all models
    #     flog = self.model_folder.parents[1] / "all_trn_date.yaml"
    #     flag = "a" if flog.is_file() else "w"
    #     with open(self.model_folder.parents[1] / "all_trn_date.yaml", flag) as f:
    #         yaml.safe_dump(
    #             {
    #                 f"{self.model_folder.parents[1].name}": {
    #                     "start": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #                 }
    #             },
    #             f,
    #         )

    def on_train_end(self, trainer, pl_module):
        """
        Save the end date of the training
        """

        # store locally some meta info, if file exists, append to it
        # this in each model folder
        flog = self.model_folder.parents[0] / "trn_date.yaml"
        flag = "a" if flog.is_file() else "w"
        with open(self.model_folder.parents[0] / "trn_date.yaml", flag) as f:
            yaml.safe_dump(
                {"train-date-end": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, f
            )

        # this is a global file containing all the training dates for all models
        flog = self.model_folder.parents[1] / "all_trn_date.yaml"
        flag = "a" if flog.is_file() else "w"
        with open(self.model_folder.parents[1] / "all_trn_date.yaml", flag) as f:
            yaml.safe_dump(
                {
                    f"{self.model_folder.parents[0].name}": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                },
                f,
            )


class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        x_n = tensor.mul_(self.std).add_(self.mean)
        return x_n

        # for t, m, s in zip(tensor, self.mean, self.std):
        #     x_n = t.mul_(s).add_(m)
        #     # The normalize code -> t.sub_(m).div_(s)
        # return x_n


def compute_frame_change_detection(
    frame_list,
    index_central,
    index_prior,
    index_after,
    score=np.std,
    histogram_match=True,
):
    """
    Compute the frame change detection. Given a list 3 of frames, compute the difference, as:
        d1 = t - t_prior
        d2 = t_after - t
        composite_diff = (1 + d1 - d2) / 2

    Parameters
    ----------
    frame_list: list[np.array]
        list of frames, in time order, each of shape (H, W, C)
    index_central: int
        index of the central frame
    index_prior: int
        index of the frame prior to the central frame
    index_after: int
        index of the frame after the central frame
    score: callable
        function to compute the score of the frame difference
    histogram_match: bool
        whether to histogram match the frames prior to computing the difference

    Returns
    -------
    composite_diff: np.array
        composite difference of the prior and after frames
    score_frame: np.array
        score of the composite difference

    """
    if len(frame_list) != 3:
        raise ValueError("frame_list must have 3 frames")

    frame_pre = frame_list[index_prior]
    frame_post = frame_list[index_after]
    frame_central = frame_list[index_central]

    # scale in [0, 1] if not already
    if frame_pre.max() > 1:
        frame_pre = frame_pre / 255
        frame_central = frame_central / 255
        frame_post = frame_post / 255

    if histogram_match:
        # match histograms to the central feame
        frame_pre = exposure.match_histograms(frame_pre, frame_central, channel_axis=2)
        frame_post = exposure.match_histograms(
            frame_post, frame_central, channel_axis=2
        )

    return composite_diff, score_frame
