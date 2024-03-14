import PIL
import numpy as np
from pathlib import Path
import random
import jittor as jt
import os
from typing import Union
import io


def tile(images, num_sample):
    num_img = len(images)
    w = images[0].width
    h = images[0].height
    img = PIL.Image.new('RGB', (num_img // num_sample * w, num_sample * h))
    for i in range(num_img // num_sample):
        for j in range(num_sample):
            img.paste(images[i * num_sample + j], (w * i, j * h))
    return img

def save_file(fname: str, data: Union[bytes, str]):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, 'wb') as fout:
        if isinstance(data, str):
            data = data.encode('utf8')
        fout.write(data)

def visualize_training(cfg):
    if not os.path.exists(f'visual/{cfg.name}'):
        os.makedirs(f'visual/{cfg.name}')
    images = []
    for cat_dir in os.walk(cfg.data_path).__next__()[1]:
        train_imgs = list(Path(os.path.join(cfg.data_path, cat_dir)).rglob('*'))
        img_paths = random.choices(train_imgs, k=64)
        for path in img_paths:
            images.append(PIL.Image.open(path))
    img = tile(images, 8)
    image_bits = io.BytesIO()
    img.save(image_bits, format='png', compress_level=0, optimize=False)
    save_file(f'visual/{cfg.name}/train_sample.png', image_bits.getbuffer())

def visualize_generate(samples, cfg, epoch):
    jt.save_image(samples, 
                  f'visual/{cfg.name}/epoch_{epoch}_generate_sample.png',
                  nrow=8,
                  normalize=True,
                  range=(-1, 1),
            )