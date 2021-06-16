# %% 
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms, utils
import pandas as pd
import os
from skimage import io, transform
import matplotlib.pyplot as plt
# import numpy as np

# %% read in labels data
labels = pd.read_csv('../data/Weinstein2018MEE_ground_truth.csv')
labels.iloc[:10, ]

# %% slice labels file to keep only extracted frames
labels = labels[labels["Video"] == "FH102_02"]
# just making sure that concatenation works... 
str(labels.iloc[0, 0])+str(labels.iloc[0, 1])
"frame"+str(labels.iloc[0, 1])+".jpeg"

# %%  create file name variable
i = 0
labels["framename"] = "teststring"
while i < len(labels): 
    labels.iloc[i, 3] = "frame"+str(labels.iloc[i, 1])+".jpeg"
    i = i + 1
print("All done!")

# %% check and write new csv
labels.head
labels.tail
labels.to_csv('labels_FH102_02.csv')

# %% define dataloader class
class DeepMeerkatData(Dataset): 
    """Data class for manually scored dataset for Ben's DeepMeerkat paper.""" 

    def __init__(self, csv_file, root_dir, transform=None): 
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """ 
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transforms

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx): 
        if torch.is_tensor(idx):
            idx = idx.tolist() 

        img_name = os.path.join(self.root_dir, 
                                self.labels.iloc[idx, 4]) # this grabs the frame name
        image = io.imread(img_name) 
        groundtruth = self.labels.iloc[idx, 2] # this grabs the Positive/Negative value
        sample = {'image' : image, 'groundtruth' : groundtruth}
    
        if self.transform:
            sample = self.transform(sample)

        return sample

# %% instantiate dataset
fh102_02 = DeepMeerkatData(csv_file='labels_FH102_02.csv', root_dir='images/')

### iterate through the dataloader
len(fh102_02)

# not sure how to call the dictionary "sample"...
# fh102_02[1]
# features, labels = next(iter(fh102_02))
# img = fh102_02[0].squeeze()
# lbl = fh102_02[0]
# plt.imshow(img)
# plt.show()
# print(f"Label: {lbl}")

# %%
