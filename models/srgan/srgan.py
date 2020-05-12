"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 srgan.py'
"""

import argparse
import os
import numpy as np
import math
import itertools
import sys

from torchvision.utils import save_image, make_grid
from models import *
from datasets import *
import torch
import jittor as jt

jt.flags.use_cuda = 1

os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = GeneratorResNet()
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
feature_extractor = FeatureExtractor()

generator.load_parameters(torch.load('/home/storage/zwy/workspace/GAN/PyTorch-GAN/implementations/srgan/generator_init.pth'))
discriminator.load_parameters(torch.load('/home/storage/zwy/workspace/GAN/PyTorch-GAN/implementations/srgan/discriminator_init.pth'))
feature_extractor.load_parameters(torch.load('/home/storage/zwy/workspace/GAN/PyTorch-GAN/implementations/srgan/feature_extractor_init.pth'))

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = nn.MSELoss()
criterion_content = nn.L1Loss()


# Optimizers
optimizer_G = nn.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = nn.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

dataloader = ImageDataset("../../data/%s" % opt.dataset_name, hr_shape=hr_shape).set_attrs(batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
# ----------
#  Training
# ----------

for epoch in range(opt.epoch, opt.n_epochs):
    for i, imgs in enumerate(dataloader):
        jt.sync_all(True)
        # Configure model input
        imgs_lr = imgs[0]
        imgs_hr = imgs[1]

        # Adversarial ground truths
        valid = jt.array(np.ones((imgs_lr.size(0), *discriminator.output_shape))).float32().stop_grad()
        fake = jt.array(np.zeros((imgs_lr.size(0), *discriminator.output_shape))).float32().stop_grad()


        # ------------------
        #  Train Generators
        # ------------------

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.stop_grad())

        # Total loss
        loss_G = loss_content + 1e-3 * loss_GAN
        optimizer_G.step(loss_G)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.stop_grad()), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        optimizer_D.step(loss_D)

        # --------------
        #  Log Progress
        # --------------

        sys.stdout.write(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), loss_D.data[0], loss_G.data[0])
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            # Save image grid with upsampled inputs and SRGAN outputs
            imgs_lr = nn.Upsample(4)(imgs_lr)
            gen_hr = make_grid(torch.Tensor(gen_hr.numpy()), nrow=1, normalize=True)
            imgs_lr = make_grid(torch.Tensor(imgs_lr.numpy()), nrow=1, normalize=True)
            img_grid = torch.cat((imgs_lr, gen_hr), -1)
            save_image(img_grid, "images/%d.png" % batches_done, normalize=False)