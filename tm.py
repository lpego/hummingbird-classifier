# %%
import torch
import numpy as np

from torchvision import utils
from PIL import Image, ImageFilter, ImageDraw

from torch.utils.data import Dataset
from torchvision import transforms

# %%

fi = dataloader.dataset.img_paths[124]
x = Image.open(fi).convert("RGB")
# %%
size = 224
box_s = ((10, 100), (10, 100))
x = transforms.Resize(size=(size, size))(x)

# %%
from HummingbirdLoader_v2 import BlurImagePart, AddLightHazePart

# %%
for _ in range(10):
    y = AddLightHazePart(
        size=224, box_s=((20, 150), (10, 100)), box_blur_rad=(30, 70), p=0.99
    )(x)
    y = BlurImagePart(size=224, box_s=((20, 150), (10, 100)), p=0.7)(y)
    plt.figure()
    plt.imshow(y)


# %%
