import glob
import random
import os
import numpy as np
from jittor.dataset.dataset import Dataset
from PIL import Image
import jittor.transform as transform

class ImageDataset(Dataset):
    def __init__(self, root, input_shape, mode="train"):
        self.transform = transform.Compose(
            [
                transform.Resize(input_shape[-2:]),
                transform.ImageNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        self.total_len = len(self.files)
        self.batch_size = None
        self.shuffle = False
        self.drop_last = False
        self.num_workers = None
        self.buffer_size = 512*1024*1024

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return img_A, img_B
