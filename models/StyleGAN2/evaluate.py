from train import get_config
from training.misc import define_G
from tqdm import tqdm
import jittor as jt
import io
import os
import PIL
import numpy as np
from subprocess import run

def generate():
    cfg = get_config()
    G_ema = define_G(cfg)
    G_ema.load(cfg.g_ema_pretrained)
    if cfg.gpu:
        jt.flags.use_cuda = 1
    else:
        jt.flags.use_cuda = 0
    if not os.path.exists(cfg.eval_path):
        os.makedirs(cfg.eval_path)
    for i in tqdm(range(cfg.eval_size // cfg.batch_size + 1)):
        sample_z = jt.randn(cfg.batch_size, 512)
        eval_images, _ = G_ema([sample_z])
        for j in range(eval_images.shape[0]):
            if i*cfg.batch_size + j + 1 <= cfg.eval_size:
                jt.save_image(eval_images[j], 
                        os.path.join(cfg.eval_path, f'{i*cfg.batch_size + j}.png'),
                        nrow=1,
                        normalize=True,
                        range=(-1, 1),
                )
    if cfg.eval_orig is None:
        path = os.path.join(cfg.data_path, '0')
        run(f'python3.7 -m jittor_fid {path} {cfg.eval_path}'.split())
    else:
        run(f'python3.7 -m jittor_fid {cfg.eval_orig} {cfg.eval_path}'.split())
    
    
if __name__ == "__main__":
    generate()