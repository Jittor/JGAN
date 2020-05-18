import glob
import random
import os
import numpy as np

from jittor.dataset.dataset import Dataset
import jittor.transform as transform
from PIL import Image

class CelebADataset(Dataset):
    def __init__(self, root, transform_=None, mode="train", attributes=None):
        super().__init__()
        self.transform = transform.Compose(transform_)

        self.selected_attrs = attributes
        self.files = sorted(glob.glob("%s/images/*.jpg" % root))
        self.files = self.files[:-2000] if mode == "train" else self.files[-2000:]
        self.label_path = glob.glob("%s/*.txt" % root)[0]
        self.annotations = self.get_annotations()
        self.set_attrs(total_len=len(self.files))

    def get_annotations(self):
        """Extracts annotations for CelebA"""
        annotations = {}
        lines = [line.rstrip() for line in open(self.label_path, "r")]
        self.label_names = lines[1].split()
        for _, line in enumerate(lines[2:]):
            filename, *values = line.split()
            labels = []
            for attr in self.selected_attrs:
                idx = self.label_names.index(attr)
                labels.append(1 * (values[idx] == "1"))
            annotations[filename] = labels
        return annotations

    def __getitem__(self, index):
        filepath = self.files[index % len(self.files)]
        filename = filepath.split("/")[-1]
        img = self.transform(Image.open(filepath))
        label = self.annotations[filename]
        label = np.array(label).astype(np.float32)

        return img, label
