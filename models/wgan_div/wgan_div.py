import argparse
import os
import numpy as np
import math
import sys

import jittor as jt
from jittor import nn

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def execute(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def execute(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

import cv2
def save_image(img, path, nrow=10):
    N,C,W,H = img.shape
    img2 = img.reshape([-1,W*nrow*nrow,H])
    img = img2[:,:W*nrow,:]
    for i in range(1,nrow):
        img = np.concatenate([img,img2[:,W*nrow*i:W*nrow*(i+1),:]],axis=2)
    min_ = img.min()
    max_ = img.max()
    img = (img - min_) / (max_ - min_) * 255
    img = img.transpose((1,2,0))
    cv2.imwrite(path,img)

k = 2
p = 6

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Configure data loader
from jittor.dataset.mnist import MNIST
import jittor.transform as transform

transform = transform.Compose([
    transform.Resize(opt.img_size),
    transform.Gray(),
    transform.ImageNormalize(mean=[0.5], std=[0.5]),
])
dataloader = MNIST(train=True, transform=transform).set_attrs(batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = nn.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = nn.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = jt.array(imgs).stop_grad()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Sample noise as generator input
        z = jt.array(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))).float32()

        # Generate a batch of images
        fake_imgs = generator(z)

        # Real images
        real_validity = discriminator(real_imgs)
        # Fake images
        fake_validity = discriminator(fake_imgs)
        from pdb import set_trace as st
        # st()
        # Compute W-div gradient penalty
        real_grad = jt.grad(real_validity, real_imgs)
        real_grad_norm = real_grad.view(real_grad.size(0), -1).sqr().sum(1) ** (p / 2)

        fake_grad = jt.grad(fake_validity, fake_imgs)
        fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).sqr().sum(1) ** (p / 2)

        div_gp = jt.mean(real_grad_norm + fake_grad_norm) * k / 2

        # Adversarial loss
        d_loss = -jt.mean(real_validity) + jt.mean(fake_validity) + div_gp

        optimizer_D.step(d_loss)


        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -jt.mean(fake_validity)

            optimizer_G.step(g_loss)

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.data[0], g_loss.data[0])
            )

            if batches_done % opt.sample_interval == 0:
                save_image(fake_imgs.numpy()[:25], "images/%d.png" % batches_done, nrow=5)

            batches_done += opt.n_critic
