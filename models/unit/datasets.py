import glob
import random
import os
import numpy as np
from jittor.dataset.dataset import Dataset
from PIL import Image
import jittor.transform as transform

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transform.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))
        self.total_len = max(len(self.files_A), len(self.files_B)) 
        self.batch_size = None
        self.shuffle = False
        self.drop_last = False
        self.num_workers = None
        self.buffer_size = 512*1024*1024

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return item_A, item_B