
import argparse
import os
import numpy as np
import math
import itertools
import sys
import datetime
import time
from torchvision.utils import save_image
from models import *
from datasets import *
from jittor import nn
import jittor as jt

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="edges2shoes", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=64, help="size of image height")
parser.add_argument("--img_width", type=int, default=64, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

# Losses
adversarial_loss = nn.MSELoss()
cycle_loss = nn.L1Loss()
pixelwise_loss = nn.L1Loss()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G_AB = GeneratorUNet(input_shape)
G_BA = GeneratorUNet(input_shape)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

G_AB.load_parameters(torch.load('/home/storage/zwy/workspace/GAN/PyTorch-GAN/implementations/discogan/G_AB_init.pth'))
G_BA.load_parameters(torch.load('/home/storage/zwy/workspace/GAN/PyTorch-GAN/implementations/discogan/G_BA_init.pth'))
D_A.load_parameters(torch.load('/home/storage/zwy/workspace/GAN/PyTorch-GAN/implementations/discogan/D_A_init.pth'))
D_B.load_parameters(torch.load('/home/storage/zwy/workspace/GAN/PyTorch-GAN/implementations/discogan/D_B_init.pth'))

# Optimizers
optimizer_G = nn.Adam(
    G_AB.parameters() + G_BA.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_A = nn.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = nn.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Dataset loader
dataloader = ImageDataset("../../../PyTorch-GAN/data/%s" % opt.dataset_name, input_shape).set_attrs(batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

val_dataloader = ImageDataset("../../../PyTorch-GAN/data/%s" % opt.dataset_name, input_shape, mode="val").set_attrs(batch_size=16, shuffle=False, num_workers=opt.n_cpu)

from pdb import set_trace as st
def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    G_AB.eval()
    G_BA.eval()
    real_A = imgs[0]
    fake_B = G_AB(real_A)
    real_B = imgs[1]
    fake_A = G_BA(real_B)
    img_sample = jt.contrib.concat((real_A, fake_B, real_B, fake_A), 0)
    save_image(torch.Tensor(img_sample.numpy()), "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=8, normalize=True)

# ----------
#  Training
# ----------

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        jt.sync_all(True)
        # Model inputs
        real_A = batch[0].stop_grad()
        real_B = batch[1].stop_grad()

        # Adversarial ground truths
        valid = jt.array(np.ones((real_A.size(0), *D_A.output_shape))).float32().stop_grad()
        fake = jt.array(np.zeros((real_A.size(0), *D_A.output_shape))).float32().stop_grad()

        # ------------------
        #  Train Generators
        # ------------------

        G_AB.train()
        G_BA.train()

        # GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = adversarial_loss(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = adversarial_loss(D_A(fake_A), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Pixelwise translation loss
        loss_pixelwise = (pixelwise_loss(fake_A, real_A) + pixelwise_loss(fake_B, real_B)) / 2

        # Cycle loss
        loss_cycle_A = cycle_loss(G_BA(fake_B), real_A)
        loss_cycle_B = cycle_loss(G_AB(fake_A), real_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G = loss_GAN + loss_cycle + loss_pixelwise
        loss_G.sync()
        optimizer_G.step(loss_G)

        # -----------------------
        #  Train Discriminator A
        # -----------------------


        # Real loss
        loss_real = adversarial_loss(D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        loss_fake = adversarial_loss(D_A(fake_A.stop_grad()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2
        loss_D_A.sync()
        optimizer_D_A.step(loss_D_A)

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        # Real loss
        loss_real = adversarial_loss(D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        loss_fake = adversarial_loss(D_B(fake_B.stop_grad()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2
        loss_D_B.sync()
        optimizer_D_B.step(loss_D_B)

        loss_D = 0.5 * (loss_D_A + loss_D_B)

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        
        info = jt.liveness_info()
        print(info)
        # Print log
        print(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, pixel: %f, cycle: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.data[0],
                loss_G.data[0],
                loss_GAN.data[0],
                loss_pixelwise.data[0],
                loss_cycle.data[0],
                time_left,
            )
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)