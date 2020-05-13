import argparse
import os
import numpy as np
import math
import cv2

import jittor as jt
from jittor import init
from jittor import nn

jt.flags.use_cuda = 1

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

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="number of image channels")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.gauss_(m.weight, 0.0, 0.02)
        init.gauss_(m.bias, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        init.gauss_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0.0)

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = (opt.img_size // 4)
        self.l1 = nn.Sequential(
            nn.Linear(opt.latent_dim, (128 * (self.init_size ** 2))))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm(128), 
            nn.Upsample(scale_factor=2), 
            nn.Conv(128, 128, 3, stride=1, padding=1), 
            nn.BatchNorm(128, 0.8), nn.Leaky_relu(0.2), 
            nn.Upsample(scale_factor=2), 
            nn.Conv(128, 64, 3, stride=1, padding=1), 
            nn.BatchNorm(64, 0.8),  
            nn.Leaky_relu(0.2), 
            nn.Conv(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, noise):
        out = self.l1(noise)
        out = jt.reshape(out, [out.shape[0], 128, self.init_size, self.init_size])
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.down = nn.Sequential(
            nn.Conv(opt.channels, 64, 3, 2, 1), 
            nn.Relu()
        )
        self.down_size = (opt.img_size // 2)
        down_dim = (64 * ((opt.img_size // 2) ** 2))
        self.fc = nn.Sequential(
            nn.Linear(down_dim, 32), 
            nn.BatchNorm1d(32, 0.8), 
            nn.Relu(), 
            nn.Linear(32, down_dim), 
            nn.BatchNorm1d(down_dim), 
            nn.Relu()
        )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2), 
            nn.Conv(64, opt.channels, 3, 1, 1)
        )

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, img):
        out = self.down(img)
        out = self.fc(jt.reshape(out, [out.shape[0], (- 1)]))
        out = self.up(jt.reshape(out, [out.shape[0], 64, self.down_size, self.down_size]))
        return out

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
train_loader = MNIST(train=True, transform=transform).set_attrs(batch_size=opt.batch_size, shuffle=False)

# Optimizers
optimizer_G = nn.Adam(
    generator.parameters(), 
    lr=opt.lr, 
    betas=(opt.b1, opt.b2)
)
optimizer_D = nn.Adam(
    discriminator.parameters(), 
    lr=opt.lr, 
    betas=(opt.b1, opt.b2)
)

# ----------
#  Training
# ----------

# BEGAN hyper parameters
gamma = 0.75
lambda_k = 0.001
k = 0.0

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(train_loader):
        # Configure input
        real_imgs = jt.array(imgs)

        # -----------------
        #  Train Generator
        # -----------------

        # Sample noise as generator input
        z = jt.array(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))).float32()

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = jt.mean(jt.abs(discriminator(gen_imgs) - gen_imgs))
        optimizer_G.step(g_loss)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Measure discriminator's ability to classify real from generated samples
        d_real = discriminator(real_imgs)
        d_fake = discriminator(gen_imgs.stop_grad())

        d_loss_real = jt.mean(jt.abs(d_real - real_imgs))
        d_loss_fake = jt.mean(jt.abs(d_fake - gen_imgs.stop_grad()))
        d_loss = d_loss_real - k * d_loss_fake

        optimizer_D.step(d_loss)

        # ----------------
        # Update weights
        # ----------------

        diff = jt.mean(gamma * d_loss_real - d_loss_fake)

        # Update weight term for fake samples
        k = k + lambda_k * diff.data[0]
        k = min(max(k, 0), 1)  # Constraint to interval [0, 1]

        # Update convergence metric
        M = (d_loss_real + jt.abs(diff)).data[0]

        # --------------
        # Log Progress
        # --------------
        if i % 50 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] -- M: %f, k: %f"
                % (epoch, opt.n_epochs, i, len(train_loader), d_loss.data[0], g_loss.data[0], M, k)
            )

        batches_done = epoch * len(train_loader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.numpy()[:25], "images/%d.png" % batches_done, nrow=5)