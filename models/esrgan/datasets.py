import glob
import random
import os
import numpy as np

from jittor.dataset.dataset import Dataset
from PIL import Image
import jittor.transform as transform
import jittor as jt

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def denormalize(var):
    """ Denormalizes image tensors using mean and std """
    return jt.clamp(var * jt.array(std).broadcast(var, [0,2,3]) + jt.array(mean).broadcast(var, [0,2,3]), 0, 255)


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        super().__init__()
        hr_height, hr_width = hr_shape
        # transform for low resolution images and high resolution images
        self.lr_transform = transform.Compose(
            [
                transform.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
                transform.ImageNormalize(mean, std),
            ]
        )
        self.hr_transform = transform.Compose(
            [
                transform.Resize((hr_height, hr_height), Image.BICUBIC),
                transform.ImageNormalize(mean, std),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))
        self.total_len = len(self.files)
        self.batch_size = None
        self.shuffle = False
        self.drop_last = False
        self.num_workers = None
        self.buffer_size = 512*1024*1024

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)
        return img_lr, img_hr
        return {"lr": img_lr, "hr": img_hr}