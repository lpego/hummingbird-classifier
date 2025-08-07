import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from torchvision import utils
from PIL import Image, ImageFilter, ImageDraw, ImageOps

from torch.utils.data import Dataset


class HummingbirdLoader(Dataset):
    """
    PyTorch Dataset for loading hummingbird images with positive and negative samples.

    Args:
        dir_dict: Dictionary with 'positives' and 'negatives' keys containing paths to image directories
        ls_inds: List of indices for subset selection (empty = use all data)
        learning_set: String identifier for the learning set type
        transforms: Transform functions to apply to images
    """

    def __init__(
        self,
        dir_dict: Dict,
        ls_inds: List = [],
        learning_set: str = "all",
        transforms: Optional[Any] = None,
    ):
        self.transforms = transforms
        self.imsize = 224

        self.ls_inds = ls_inds
        self.learning_set = learning_set

        self.img_paths, self.labels, self.inds = self.prepare_data(
            dir_dict, ls_inds=self.ls_inds
        )

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple of (image_tensor, label, index)
        """
        try:
            with open(self.img_paths[idx], "rb") as f:
                img = Image.open(f).convert("RGB")
            label = self.labels[idx]

            if isinstance(self.transforms, dict):
                tensor_image = self.transforms[str(label.item())](img)
            else:
                tensor_image = self.transforms(img)

            # if self.learning_set == "trn":
            #     if len(self.transforms) > 1:
            #         tensor_image = self.transforms[label](img)
            #     else:
            #         tensor_image = self.transforms[0](img)
            # else:
            #     tensor_image = self.transforms(img)

            return tensor_image, label, idx

        except OSError:
            return (
                torch.zeros((3, self.imsize, self.imsize)),
                torch.LongTensor([-1]).squeeze(),
                idx,
            )

    @staticmethod
    def prepare_data(
        dir_dict: Dict, ls_inds: List
    ) -> Tuple[np.ndarray, torch.Tensor, List]:
        """
        Prepare image paths and labels from directory structure.

        Args:
            dir_dict: Dictionary with 'positives' and 'negatives' keys
            ls_inds: List of indices for subset selection

        Returns:
            Tuple of (image_paths, labels, indices)
        """
        # Make paths
        if isinstance(dir_dict["negatives"], list):
            negatives = []
            for sub_dic in dir_dict["negatives"]:
                negatives += list(sub_dic.glob("*.jpg"))
        else:
            negatives = list(dir_dict["negatives"].glob("*.jpg"))

        if isinstance(dir_dict["negatives"], list):
            positives = []
            for sub_dic in dir_dict["positives"]:
                positives += list(sub_dic.glob("*.jpg"))
        else:
            positives = list(dir_dict["positives"].glob("*.jpg"))

        negatives.sort()
        positives.sort()

        img_paths = np.asarray(negatives + positives)

        # Make labels
        labels = [0] * len(negatives) + [1] * len(positives)
        labels = np.asarray(labels)
        labels = torch.LongTensor(labels)

        if len(ls_inds) < 1:
            return img_paths, labels, ls_inds

        return img_paths[ls_inds], labels[ls_inds], ls_inds


class BlurImagePart(object):
    """
    Blur a random rectangular region of the image for data augmentation.

    Args:
        size: Image size (assumes square images)
        box_s: Tuple defining box size ranges ((width_min, width_max), (height_min, height_max))
        gaussian_rad: Radius for Gaussian blur filter
        interpolation: PIL interpolation method
        p: Probability of applying the transform
    """

    def __init__(
        self,
        size: int,
        box_s: Tuple,
        gaussian_rad: int = 2,
        interpolation: int = Image.BILINEAR,
        p: float = 0.2,
    ):
        self.size = size
        self.interpolation = interpolation
        self.box_size_interval = box_s
        self.gaussian_rad = gaussian_rad
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply blur to a random rectangular region of the image.

        Args:
            img: PIL Image to be processed

        Returns:
            PIL Image with potentially blurred rectangular region
        """
        if torch.rand(1).item() < self.p:
            # 1 sample box size
            width = np.random.uniform(
                low=self.box_size_interval[0][0],
                high=self.box_size_interval[0][1],
                size=None,
            )

            height = np.random.uniform(
                low=self.box_size_interval[1][0],
                high=self.box_size_interval[1][1],
                size=None,
            )

            ul_corner_x = np.random.uniform(low=0, high=self.size - width, size=None)
            ul_corner_y = np.random.uniform(low=0, high=self.size - height, size=None)

            box = [
                int(a)
                for a in [
                    ul_corner_x,
                    ul_corner_y,
                    ul_corner_x + width,
                    ul_corner_y + height,
                ]
            ]
            crop_img = img.crop(box)
            blur_image = crop_img.filter(
                ImageFilter.GaussianBlur(radius=self.gaussian_rad)
            )
            img.paste(blur_image, box)
        return img

    def __repr__(self) -> str:
        """Return string representation of the transform."""
        return self.__class__.__name__ + "(im size={0}, box size={1})".format(
            self.size, self.box_size_interval
        )


class AddLightHazePart(object):
    """
    Add a light haze effect to a random rectangular region of the image.

    Args:
        size: Image size (assumes square images)
        box_s: Tuple defining box size ranges ((width_min, width_max), (height_min, height_max))
        box_blur_rad: Tuple defining blur radius range (min_radius, max_radius)
        interpolation: PIL interpolation method
        p: Probability of applying the transform
    """

    def __init__(
        self,
        size: int,
        box_s: Tuple,
        box_blur_rad: Tuple = (20, 70),
        interpolation: int = Image.BILINEAR,
        p: float = 0.25,
    ):
        self.size = size
        self.interpolation = interpolation
        self.box_size_interval = box_s
        self.box_blur_rad = box_blur_rad
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply light haze effect to a random rectangular region of the image.

        Args:
            img: PIL Image to be processed

        Returns:
            PIL Image with potentially added haze effect
        """
        if torch.rand(1).item() < self.p:
            # 1 sample box size
            width = np.random.uniform(
                low=self.box_size_interval[0][0],
                high=self.box_size_interval[0][1],
                size=None,
            )

            height = np.random.uniform(
                low=self.box_size_interval[1][0],
                high=self.box_size_interval[1][1],
                size=None,
            )

            ul_corner_x = np.random.uniform(low=0, high=self.size - width, size=None)
            ul_corner_y = np.random.uniform(low=0, high=self.size - height, size=None)

            box = [
                int(a)
                for a in [
                    ul_corner_x,
                    ul_corner_y,
                    ul_corner_x + width,
                    ul_corner_y + height,
                ]
            ]

            y = Image.new("RGB", (self.size, self.size), (0, 0, 0))
            draw = ImageDraw.Draw(y)
            draw.rectangle(box, fill=(225, 225, 225), outline=(255, 255, 255))

            y = y.filter(
                ImageFilter.GaussianBlur(
                    np.random.randint(self.box_blur_rad[0], self.box_blur_rad[1])
                )
            )
            z = torch.max(
                torch.Tensor(0.6 * np.array(y)), torch.Tensor(0.4 * np.array(img))
            ).numpy()
            # z = np.clip((255 - np.max(z)) + z, 0, 255)
            # z = z.astype(np.uint8)
            img = Image.fromarray(z.astype(np.uint8))
            img = ImageOps.autocontrast(img)

        return img

    def __repr__(self) -> str:
        """Return string representation of the transform."""
        return self.__class__.__name__ + "(im size={0}, blur box size={1})".format(
            self.size, self.box_size_interval
        )


class CustomCrop(object):
    """
    Crop a specific part of the image and discard the rest.

    Args:
        box: Tuple defining crop coordinates (left, top, right, bottom)
        p: Probability of applying the transform
    """

    def __init__(self, box: Tuple, p: float = 1.0):
        self.box = box
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Crop the image to the specified box region.

        Args:
            img: PIL Image to be cropped

        Returns:
            PIL Image cropped to the specified region
        """
        if torch.rand(1).item() < self.p:
            crop_img = img.crop(self.box)
        return crop_img

    def __repr__(self) -> str:
        """Return string representation of the transform."""
        return self.__class__.__name__ + "(box={0})".format(self.box)
