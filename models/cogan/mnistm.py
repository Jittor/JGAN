"""Dataset setting and data loader for MNIST-M.

Modified from
https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py

CREDIT: https://github.com/corenel
"""

from __future__ import print_function

import errno
import os
import pickle
from jittor.dataset.dataset import Dataset
from PIL import Image


class MNISTM(Dataset):
    """`MNIST-M Dataset."""

    def __init__(self, mnist_root="data", train=True, transform=None, target_transform=None):
        """Init MNIST-M dataset."""
        super(MNISTM, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        mnistm_data = pickle.load(open(os.path.join(mnist_root, 'mnistm.pkl'), 'rb'))
        if self.train:
            self.train_data, self.train_labels = mnistm_data['train_imgs'], mnistm_data['train_labels']
        else:
            self.test_data, self.test_labels = mnistm_data['test_imgs'], mnistm_data['test_labels']
        
        self.total_len = len(self.train_data) if self.train else len(self.test_data)
        self.batch_size = None
        self.shuffle = False
        self.drop_last = False
        self.num_workers = None
        self.buffer_size = 512*1024*1024

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img, mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target