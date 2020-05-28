"""
Inpainting using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 context_encoder.py'
"""

import argparse
import os
import numpy as np
import math
import cv2
from PIL import Image
import time

import jittor as jt
from jittor import init
from jittor import nn
import jittor.transform as transform

from datasets import *
from models import *

jt.flags.use_cuda = 1
os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--mask_size", type=int, default=64, help="size of random mask")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

def save_image(img, path, nrow=10):
    N,C,W,H = img.shape
    if (N%nrow!=0):
        print("save_image error: N%nrow!=0")
        return
    img=img.transpose((1,0,2,3))
    ncol=int(N/nrow)
    img2=img.reshape([img.shape[0],-1,H])
    img=img2[:,:W*ncol,:]
    for i in range(1,int(img2.shape[1]/W/ncol)):
        img=np.concatenate([img,img2[:,W*ncol*i:W*ncol*(i+1),:]],axis=2)
    min_=img.min()
    max_=img.max()
    img=(img-min_)/(max_-min_)*255
    img=img.transpose((1,2,0))
    if C==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path,img)

# Calculate output of image discriminator (PatchGAN)
patch_h, patch_w = int(opt.mask_size / 2 ** 3), int(opt.mask_size / 2 ** 3)
patch = (1, patch_h, patch_w)

# Loss function
adversarial_loss = nn.MSELoss()
pixelwise_loss = nn.L1Loss()

# Initialize generator and discriminator
generator = Generator(channels=opt.channels)
discriminator = Discriminator(channels=opt.channels)

# Dataset loader
transforms_ = [
    transform.Resize((opt.img_size, opt.img_size), mode=Image.BICUBIC),
    transform.ImageNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
dataloader = ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_).set_attrs(
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
test_dataloader = ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, mode="val").set_attrs(
    batch_size=12,
    shuffle=True,
    num_workers=1,
)
test_iter = iter(test_dataloader)

# Optimizers
optimizer_G = jt.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = jt.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

def save_sample(batches_done):
    global test_iter
    
    try:
        samples, masked_samples, i = next(test_iter)
    except:
        test_iter = iter(test_dataloader)
        samples, masked_samples, i = next(test_iter)
    # Upper-left coordinate of mask
    i = i.numpy()[0]
    # Generate inpainted image
    gen_mask = generator(masked_samples)
    filled_samples = masked_samples.clone()
    filled_samples[:, :, i : i + opt.mask_size, i : i + opt.mask_size] = gen_mask
    # Save sample
    sample = np.concatenate((masked_samples.numpy(), filled_samples.numpy(), samples.numpy()), 2)
    save_image(sample, "images/%d.png" % batches_done, nrow=6)

warmup_times = -1
run_times = 3000
total_time = 0.
cnt = 0

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, masked_imgs, masked_parts) in enumerate(dataloader):

        # Adversarial ground truths
        valid = jt.ones([imgs.shape[0], *patch]).stop_grad()
        fake = jt.zeros([imgs.shape[0], *patch]).stop_grad()

        # -----------------
        #  Train Generator
        # -----------------
        # Generate a batch of images
        gen_parts = generator(masked_imgs)

        # Adversarial and pixelwise loss
        g_adv = adversarial_loss(discriminator(gen_parts), valid)
        g_pixel = pixelwise_loss(gen_parts, masked_parts)
        # Total loss
        g_loss = 0.001 * g_adv + 0.999 * g_pixel
        optimizer_G.step(g_loss)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(masked_parts), valid)
        fake_loss = adversarial_loss(discriminator(gen_parts.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)
        optimizer_D.step(d_loss)

        if warmup_times==-1:
            jt.sync_all()
            batches_done = epoch * len(dataloader) + i
            if batches_done % 50 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G adv: %f, pixel: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.numpy()[0], g_adv.numpy()[0], g_pixel.numpy()[0])
                )

            # Generate sample at sample interval
            if batches_done % opt.sample_interval == 0:
                save_sample(batches_done)
        else:            
            jt.sync_all()
            cnt += 1
            print(cnt)
            if cnt == warmup_times:
                jt.sync_all(True)
                sta = time.time()
            if cnt > warmup_times + run_times:
                jt.sync_all(True)
                total_time = time.time() - sta
                print(f"run {run_times} iters cost {total_time} seconds, and avg {total_time / run_times} one iter.")
                exit(0)
