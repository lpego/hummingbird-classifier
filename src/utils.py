from torchvision import models, transforms

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

    return model
