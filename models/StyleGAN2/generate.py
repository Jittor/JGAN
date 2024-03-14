from train import get_config
from training.misc import define_G
from tqdm import tqdm
import jittor as jt
import io
import os
import PIL
import numpy as np


def save_file(batch, index, batch_size):
    for i in range(batch_size):
        jt.save_image(batch[i], 
                      f'/mnt/disk/yuanlu/eval/FFHQ/{index*100 + i}.png',
                      nrow=1,
                      normalize=True,
                      range=(-1, 1),
        )

def generate():
    cfg = get_config()
    G_ema = define_G(cfg)
    G_ema.load(cfg.g_ema_pretrained)
    if cfg.gpu:
        jt.flags.use_cuda = 1
    else:
        jt.flags.use_cuda = 0
    for i in tqdm(range(cfg.eval_size // cfg.batch_size + 1)):
        sample_z = jt.randn(cfg.batch_size, 512)
        eval_images, _ = G_ema([sample_z])
        save_file(eval_images, i, cfg.batch_size)

if __name__ == "__main__":
    generate()