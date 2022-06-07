from torchvision import models, transforms

from pathlib import Path
import yaml

import torch.nn as nn


def read_pretrained_model(architecture, n_class):
    """
    Helper script to load models compactly from pytorch model zoo and prepare them for Hummingbird finetuning
    architecture : string of model name with
    """

    architecture = architecture.lower()

    if architecture == "vgg":
        model = models.vgg16(pretrained=True)

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

        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(
            in_features=model.fc.in_features, out_features=n_class, bias=True
        )

        # Freeze base feature extraction trunk:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.fc.parameters():
            param.requires_grad = True

    elif architecture == "resnet50":

        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(
            in_features=model.fc.in_features, out_features=n_class, bias=True
        )

        # Freeze base feature extraction trunk:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.fc.parameters():
            param.requires_grad = True

    elif architecture == "densenet161":

        model = models.densenet161(pretrained=True)
        model.classifier = nn.Linear(
            in_features=model.classifier.in_features, out_features=n_class, bias=True
        )

        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier.parameters():
            param.requires_grad = True

    elif architecture == "mobilenet":

        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(
            in_features=model.classifier[1].in_features,
            out_features=n_class,
            bias=True,
        )

        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier[1].parameters():
            param.requires_grad = True

    elif architecture == "efficientnet_b2":  # NEED LATEST VERSION OF TORCH STUFF
        model = models.efficientnet_b2(pretrained=True)
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

        model = models.vit_b_16(pretrained=True)
        model.heads.head = nn.Linear(
            in_features=model.heads.head.in_features,
            out_features=n_class,
            bias=True,
        )

        for param in model.parameters():
            param.requires_grad = False

        for param in model.heads.head.parameters():
            param.requires_grad = True

    else:
        raise OSError("Model not found")

    return model


def find_checkpoints(dirs=Path("lightning_logs"), version=None, log="val"):
    """
    dirs: *Path where the logs are.
    version: version of log to read. Default is the last one
    log: e.g. "val" or "last": whether to read best val model or just last one from trn
    """

    if version:
        ch_sf = list(dirs.glob(f"{version}/checkpoints/*.ckpt"))
    else:  # pick last
        ch_sp = [a.parents[1] for a in dirs.glob("**/*.ckpt")]
        ch_sp.sort()
        ch_sf = list(ch_sp[-1].glob("**/*.ckpt"))

    chkp = [a for a in ch_sf if log in str(a.name)]

    return chkp
