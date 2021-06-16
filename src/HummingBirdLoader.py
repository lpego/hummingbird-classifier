import torch
import numpy as np

from torchvision import utils
from PIL import Image


from torch.utils.data import Dataset


class HummingBirdLoader(Dataset):
    def __init__(self, dir_dict, ls_inds=[], learning_set="all", transforms=None):

        self.transforms = transforms
        self.ls_inds = ls_inds
        self.learning_set = learning_set

        self.img_paths, self.labels, self.inds = self.prepare_data(
            dir_dict, ls_inds=self.ls_inds
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # print(idx)
        # print(self.img_paths[idx])
        with open(self.img_paths[idx], "rb") as f:
            img = Image.open(f).convert("RGB")

        tensor_image = self.transforms(img)
        label = self.labels[idx]
        return tensor_image, label

    @staticmethod
    def prepare_data(dir_dict, ls_inds):

        # Make paths
        positives = list(dir_dict["positives"].glob("*.jpg"))
        negatives = list(dir_dict["negatives"].glob("*.jpg"))
        img_paths = np.asarray(positives + negatives)

        # Make labels
        labels = [1] * len(positives) + [0] * len(negatives)
        labels = np.asarray(labels)
        labels = torch.LongTensor(labels)

        if len(ls_inds) < 1:
            return img_paths, labels, ls_inds

        return img_paths[ls_inds], labels[ls_inds], ls_inds


class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
