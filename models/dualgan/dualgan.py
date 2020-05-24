import argparse
import os
import numpy as np
import math
import scipy
import sys
import time
import datetime
from datasets import *
from models import *
from jittor import nn
jt.flags.use_cuda = 1

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--dataset_name", type=str, default="edges2shoes", help="name of the dataset")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--sample_interval", type=int, default=200, help="interval betwen image samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

img_shape = (opt.channels, opt.img_size, opt.img_size)

# Loss function
cycle_loss = nn.L1Loss()

# Loss weights
lambda_adv = 1
lambda_cycle = 10
lambda_gp = 10

# Initialize generator and discriminator
G_AB = Generator()
G_BA = Generator()
D_A = Discriminator()
D_B = Discriminator()

# Configure data loader
dataloader = ImageDataset("../data/%s" % opt.dataset_name, img_shape).set_attrs(batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

val_dataloader = ImageDataset("../data/%s" % opt.dataset_name, img_shape, mode="val").set_attrs(batch_size=16, shuffle=True, num_workers=1)

# Optimizers
optimizer_G = nn.Adam(
    G_AB.parameters() + G_BA.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_A = nn.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = nn.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

def std(x):
    return jt.sqrt((x-x.mean()).sqr().mean())

def norm(x, k, dim):
    assert k==2 or k==1
    if k==1:
        return x.abs().sum(dim)
    if k==2:
        return jt.sqrt((x**2).sum(dim))

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = jt.array(np.random.random((real_samples.size(0), 1, 1, 1))).float32()
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))
    validity = D(interpolates)
    fake = jt.array(np.ones(validity.shape)).float32().stop_grad()
    # Get gradient w.r.t. interpolates
    gradients = jt.grad(validity, interpolates)
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((norm(gradients, 2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

import cv2

def save_image(img, path, nrow=10, padding=5):
    N,C,W,H = img.shape
    if (N%nrow!=0):
        print("N%nrow!=0")
        return
    ncol=int(N/nrow)
    img_all = []
    for i in range(ncol):
        img_ = []
        for j in range(nrow):
            img_.append(img[i*nrow+j])
            img_.append(np.zeros((C,W,padding)))
        img_all.append(np.concatenate(img_, 2))
        img_all.append(np.zeros((C,padding,img_all[0].shape[2])))
    img = np.concatenate(img_all, 1)
    img = np.concatenate([np.zeros((C,padding,img.shape[2])), img], 1)
    img = np.concatenate([np.zeros((C,img.shape[1],padding)), img], 2)
    min_=img.min()
    max_=img.max()
    img=(img-min_)/(max_-min_)*255
    img=img.transpose((1,2,0))
    if C==3:
        img = img[:,:,::-1]
    cv2.imwrite(path,img)

def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = imgs[0]
    fake_B = G_AB(real_A)
    AB = jt.contrib.concat((real_A, fake_B), -2)
    real_B = imgs[1]
    fake_A = G_BA(real_B)
    BA = jt.contrib.concat((real_B, fake_A), -2)
    img_sample = jt.contrib.concat((AB, BA), 0)
    save_image(img_sample.numpy(), "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=8)

# ----------
#  Training
# ----------

batches_done = 0
prev_time = time.time()
for epoch in range(opt.n_epochs):
    for i, batch in enumerate(dataloader):
        jt.sync_all(True)
        # Configure input
        imgs_A = batch[0]
        imgs_B = batch[1]

        # ----------------------
        #  Train Discriminators
        # ----------------------

        # Generate a batch of images
        fake_A = G_BA(imgs_B).stop_grad()
        fake_B = G_AB(imgs_A).stop_grad()

        # ----------
        # Domain A
        # ----------

        # Compute gradient penalty for improved wasserstein training
        gp_A = compute_gradient_penalty(D_A, imgs_A, fake_A)
        # Adversarial loss
        D_A_loss = -jt.mean(D_A(imgs_A)) + jt.mean(D_A(fake_A)) + lambda_gp * gp_A

        # ----------
        # Domain B
        # ----------

        # Compute gradient penalty for improved wasserstein training
        gp_B = compute_gradient_penalty(D_B, imgs_B, fake_B)
        # Adversarial loss
        D_B_loss = -jt.mean(D_B(imgs_B)) + jt.mean(D_B(fake_B)) + lambda_gp * gp_B

        # Total loss
        D_loss = D_A_loss + D_B_loss
        D_A_loss.sync()
        D_B_loss.sync()
        optimizer_D_A.step(D_A_loss)
        optimizer_D_B.step(D_B_loss)

        if i % opt.n_critic == 0:

            # ------------------
            #  Train Generators
            # ------------------

            # Translate images to opposite domain
            fake_A = G_BA(imgs_B)
            fake_B = G_AB(imgs_A)

            # Reconstruct images
            recov_A = G_BA(fake_B)
            recov_B = G_AB(fake_A)

            # Adversarial loss
            G_adv = -jt.mean(D_A(fake_A)) - jt.mean(D_B(fake_B))
            # Cycle loss
            G_cycle = cycle_loss(recov_A, imgs_A) + cycle_loss(recov_B, imgs_B)
            # Total loss
            G_loss = lambda_adv * G_adv + lambda_cycle * G_cycle

            optimizer_G.step(G_loss)

            # --------------
            # Log Progress
            # --------------

            # Determine approximate time left
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time) / opt.n_critic)
            prev_time = time.time()

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, cycle: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    D_loss.data[0],
                    G_adv.data.data[0],
                    G_cycle.data[0],
                    time_left,
                )
            )

        # Check sample interval => save sample if there
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

        batches_done += 1