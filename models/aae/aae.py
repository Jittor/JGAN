import argparse
import os
import numpy as np
import time
import itertools
import cv2

import jittor as jt
from jittor import init
from jittor import nn
jt.flags.use_cuda = 1

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

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the latent code")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=3000, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

def reparameterization(mu, logvar):
    std = jt.exp(logvar / 2)
    sampled_z = jt.array(np.random.normal(0, 1, (mu.shape[0], opt.latent_dim))).float32()
    z = sampled_z * std + mu
    return z

img_shape = (opt.channels, opt.img_size, opt.img_size)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(nn.Linear(int(np.prod(img_shape)), 512), nn.Leaky_relu(0.2), nn.Linear(512, 512), nn.BatchNorm1d(512), nn.Leaky_relu(0.2))
        self.mu = nn.Linear(512, opt.latent_dim)
        self.logvar = nn.Linear(512, opt.latent_dim)

    def execute(self, img):
        img_flat = jt.reshape(img, [img.shape[0], (- 1)])
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(nn.Linear(opt.latent_dim, 512), nn.Leaky_relu(0.2), nn.Linear(512, 512), nn.BatchNorm1d(512), nn.Leaky_relu(0.2), nn.Linear(512, int(np.prod(img_shape))), nn.Tanh())

    def execute(self, z):
        img_flat = self.model(z)
        img = jt.reshape(img_flat, [img_flat.shape[0], *img_shape])
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(nn.Linear(opt.latent_dim, 512), nn.Leaky_relu(0.2), nn.Linear(512, 256), nn.Leaky_relu(0.2), nn.Linear(256, 1), nn.Sigmoid())

    def execute(self, z):
        validity = self.model(z)
        return validity

# Use binary cross-entropy loss
adversarial_loss = nn.BCELoss()
pixelwise_loss = nn.L1Loss()

# Initialize generator and discriminator
encoder = Encoder()
decoder = Decoder()
discriminator = Discriminator()

# Configure data loader
from jittor.dataset.mnist import MNIST
import jittor.transform as transform

transform = transform.Compose([
    transform.Resize(opt.img_size),
    transform.Gray(),
    transform.ImageNormalize(mean=[0.5], std=[0.5]),
])
train_loader = MNIST(train=True, transform=transform).set_attrs(batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = nn.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D = nn.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

def sample_image(n_row, batches_done):
    """Saves a grid of generated digits"""
    # Sample noise
    z = jt.array(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))).float32().stop_grad()
    gen_imgs = decoder(z)
    save_image(gen_imgs.numpy(), "images/%d.png" % batches_done, nrow=n_row)

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(train_loader):
        sta = time.time()
        # Adversarial ground truths
        valid = jt.ones([imgs.shape[0], 1]).stop_grad()
        fake = jt.zeros([imgs.shape[0], 1]).stop_grad()

        # Configure input
        real_imgs = jt.array(imgs).stop_grad()

        # -----------------
        #  Train Generator
        # -----------------
        encoded_imgs = encoder(real_imgs)
        decoded_imgs = decoder(encoded_imgs)

        # Loss measures generator's ability to fool the discriminator
        g_loss = (0.001 * adversarial_loss(discriminator(encoded_imgs), valid) + 0.999 * pixelwise_loss(
            decoded_imgs, real_imgs
        ))

        optimizer_G.step(g_loss)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Sample noise as discriminator ground truth
        z = jt.array(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))).float32().stop_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(z), valid).float32()
        fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake).float32()
        d_loss = 0.5 * (real_loss + fake_loss)
        
        optimizer_D.step(d_loss)
        
        jt.sync_all(True)
        if i % 50 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Time: %f]"
                % (epoch, opt.n_epochs, i, len(train_loader), d_loss.data[0], g_loss.data[0], time.time() - sta)
            )
        
        batches_done = epoch * len(train_loader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)