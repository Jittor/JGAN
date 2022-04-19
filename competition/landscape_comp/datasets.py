import glob
import random
import os
import numpy as np

from jittor.dataset.dataset import Dataset
import jittor.transform as transform
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, root, mode="train", transforms=None):
        super().__init__()
        self.transforms = transform.Compose(transforms)
        self.mode = mode
        if self.mode == 'train':
            self.files = sorted(glob.glob(os.path.join(root, mode, "imgs") + "/*.*"))
        self.labels = sorted(glob.glob(os.path.join(root, mode, "labels") + "/*.*"))
        self.set_attrs(total_len=len(self.labels))
        print(f"from {mode} split load {self.total_len} images.")

    def __getitem__(self, index):
        label_path = self.labels[index % len(self.labels)]
        photo_id = label_path.split('/')[-1][:-4]
        img_B = Image.open(label_path)
        img_B = Image.fromarray(np.array(img_B).astype("uint8")[:, :, np.newaxis].repeat(3,2))

        if self.mode == "train":
            img_A = Image.open(self.files[index % len(self.files)])
            if np.random.random() < 0.5:
                img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
                img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
            img_A = self.transforms(img_A)
        else:
            img_A = np.empty([1])
        img_B = self.transforms(img_B)

        return img_A, img_B, photo_id
