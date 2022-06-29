import torch
import numpy as np

from torchvision import utils
from PIL import Image, ImageFilter, ImageDraw, ImageOps

from torch.utils.data import Dataset


class HummingbirdLoader(Dataset):
    def __init__(self, dir_dict, ls_inds=[], learning_set="all", transforms=None):

        self.transforms = transforms
        self.imsize = 224

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
    def prepare_data(dir_dict, ls_inds):

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


class BlurImagePart(object):
    """Blur only a part of the image
    Args:
    size (sequence of ints): ((h min, h max), (w min, w max)). It defines a random crop where to apply blur
    interpolation (int, optional): Desired interpolation. Default  ``PIL.Image.BILINEAR``

    FROM:
        from PIL import Image, ImageFilter

        image = Image.open('path/to/image_file')
        box = (30, 30, 110, 110)
        crop_img = image.crop(box)
        # Use GaussianBlur directly to blur the image 10 times.
        blur_image = crop_img.filter(ImageFilter.GaussianBlur(radius=10))
        image.paste(blur_image, box)
        image.save('path/to/save_image_file')
    """

    def __init__(
        self, size, box_s, gaussian_rad=2, interpolation=Image.BILINEAR, p=0.2
    ):

        self.size = size
        self.interpolation = interpolation
        self.box_size_interval = box_s
        self.gaussian_rad = gaussian_rad
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be blurred in random box.
        Returns:
            PIL Image: image as in but with blurred out rectangle.
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

    def __repr__(self):
        return self.__class__.__name__ + "(im size={0}, box size={1})".format(
            self.size, self.box_size_interval
        )


class AddLightHazePart(object):
    """Blur only a part of the image
    Args:
       size (sequence of ints): ((h min, h max), (w min, w max)). It defines a random crop where to apply blur
       interpolation (int, optional): Desired interpolation. Default  ``PIL.Image.BILINEAR``

       FROM:
       y = Image.new("RGB", (224, 224), (0, 0, 0))
       draw = ImageDraw.Draw(y)
       draw.rectangle((100, 50, 200, 100), fill=(225, 225, 225), outline=(255, 255, 255))
       y = y.filter(ImageFilter.BoxBlur(31))
       z = Image.fromarray((50+(0.6*np.array(y) + 0.4*np.array(x))).astype(np.uint8))
       z
    """

    def __init__(
        self, size, box_s, box_blur_rad=(20, 70), interpolation=Image.BILINEAR, p=0.25
    ):
        self.size = size
        self.interpolation = interpolation
        self.box_size_interval = box_s
        self.box_blur_rad = box_blur_rad
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be blurred in random box.
        Returns:
            PIL Image: image as in but with blurred out rectangle.
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

    def __repr__(self):
        return self.__class__.__name__ + "(im size={0}, blur box size={1})".format(
            self.size, self.box_size_interval
        )



class CustomCrop(object):
    """Crop a specific part of the image, and discard the rest
    Args:
    size (sequence of ints): ((h min, h max), (w min, w max)). It defines the crop 

    FROM:
        from PIL import Image, ImageFilter

        image = Image.open('path/to/image_file')
        box = (30, 30, 110, 110)
        crop_img = image.crop(box)
        # Use GaussianBlur directly to blur the image 10 times.
        blur_image = crop_img.filter(ImageFilter.GaussianBlur(radius=10))
        image.paste(blur_image, box)
        image.save('path/to/save_image_file')
    """

    def __init__(
        self, box, p=1.0
    ):

        self.box = box
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be blurred in random box.
        Returns:
            PIL Image: image as in but with blurred out rectangle.
        """
        if torch.rand(1).item() < self.p:
            crop_img = img.crop(self.box)
        return crop_img

    def __repr__(self):
        return self.__class__.__name__ + "(im size={0}, box size={1})".format(
            self.size, self.box
        )
