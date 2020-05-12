import jittor as jt
from jittor import init
from jittor import nn

import argparse
import os
import numpy as np
import math
import scipy
import itertools
import mnistm
from torchvision.utils import save_image
import torch
from jittor.dataset.mnist import MNIST
import jittor.transform as transform

jt.flags.use_cuda = 1

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if (classname.find('Linear') != (- 1)):
        init.gauss_(m.weight, mean=0.0, std=0.02)
    elif (classname.find('BatchNorm') != (- 1)):
        init.gauss_(m.weight, mean=1.0, std=0.02)
        init.constant_(m.bias, value=0.0)


class CoupledGenerators(nn.Module):
    def __init__(self):
        super(CoupledGenerators, self).__init__()
        self.init_size = (opt.img_size // 4)
        self.fc = nn.Sequential(nn.Linear(opt.latent_dim, (128 * (self.init_size ** 2))))
        self.shared_conv = nn.Sequential(nn.BatchNorm(128), nn.Upsample(scale_factor=2), nn.Conv(128, 128, 3, stride=1, padding=1), nn.BatchNorm(128, eps=0.8), nn.LeakyReLU(0.2), nn.Upsample(scale_factor=2))
        self.G1 = nn.Sequential(nn.Conv(128, 64, 3, stride=1, padding=1), nn.BatchNorm(64, eps=0.8), nn.LeakyReLU(0.2), nn.Conv(64, opt.channels, 3, stride=1, padding=1), nn.Tanh())
        self.G2 = nn.Sequential(nn.Conv(128, 64, 3, stride=1, padding=1), nn.BatchNorm(64, eps=0.8), nn.LeakyReLU(0.2), nn.Conv(64, opt.channels, 3, stride=1, padding=1), nn.Tanh())

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, noise):
        out = self.fc(noise)
        out = out.view((out.shape[0], 128, self.init_size, self.init_size))
        img_emb = self.shared_conv(out)
        img1 = self.G1(img_emb)
        img2 = self.G2(img_emb)
        return (img1, img2)

class CoupledDiscriminators(nn.Module):
    def __init__(self):
        super(CoupledDiscriminators, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv(in_filters, out_filters, 3, stride=2, padding=1)]
            if bn:
                block.append(nn.BatchNorm(out_filters, eps=0.8))
            block.extend([nn.LeakyReLU(0.2), nn.Dropout(p=0.25)])
            return block
        self.shared_conv = nn.Sequential(*discriminator_block(opt.channels, 16, bn=False), *discriminator_block(16, 32), *discriminator_block(32, 64), *discriminator_block(64, 128))
        ds_size = (opt.img_size // (2 ** 4))
        self.D1 = nn.Linear((128 * (ds_size ** 2)), 1)
        self.D2 = nn.Linear((128 * (ds_size ** 2)), 1)

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, img1, img2):
        out = self.shared_conv(img1)
        out = out.view((out.shape[0], (- 1)))
        validity1 = self.D1(out)
        out = self.shared_conv(img2)
        out = out.view((out.shape[0], (- 1)))
        validity2 = self.D2(out)
        return (validity1, validity2)

# Loss function
adversarial_loss = nn.MSELoss()

# Initialize models
coupled_generators = CoupledGenerators()
coupled_discriminators = CoupledDiscriminators()

print(coupled_generators)
print(coupled_discriminators)

transform = transform.Compose([
    transform.Resize(opt.img_size),
    transform.ImageNormalize(mean=[0.5], std=[0.5]),
])
dataloader1 = MNIST(train=True, transform=transform).set_attrs(batch_size=opt.batch_size, shuffle=True)

dataloader2 = mnistm.MNISTM(mnist_root = "../../data/mnistm", train=True, transform = transform).set_attrs(batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = nn.Adam(coupled_generators.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = nn.Adam(coupled_discriminators.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# ----------
#  Training
# ----------
from pdb import set_trace as st

for epoch in range(opt.n_epochs):
    for i, ((imgs1, _), (imgs2, _)) in enumerate(zip(dataloader1, dataloader2)):
        jt.sync_all(True)
        batch_size = imgs1.shape[0]

        # Adversarial ground truths
        valid = jt.ones([batch_size, 1]).float32().stop_grad()
        fake = jt.zeros([batch_size, 1]).float32().stop_grad()

         # ------------------
        #  Train Generators
        # ------------------

        # Sample noise as generator input
        z = jt.array(np.random.normal(0, 1, (batch_size, opt.latent_dim))).float32()

        # Generate a batch of images
        gen_imgs1, gen_imgs2 = coupled_generators(z)
        # Determine validity of generated images
        validity1, validity2 = coupled_discriminators(gen_imgs1, gen_imgs2)

        g_loss = (adversarial_loss(validity1, valid) + adversarial_loss(validity2, valid)) / 2

        optimizer_G.step(g_loss)

        # ----------------------
        #  Train Discriminators
        # ----------------------

        # Determine validity of real and generated images
        validity1_real, validity2_real = coupled_discriminators(imgs1, imgs2)
        validity1_fake, validity2_fake = coupled_discriminators(gen_imgs1.stop_grad(), gen_imgs2.stop_grad())

        d_loss = (
            adversarial_loss(validity1_real, valid)
            + adversarial_loss(validity1_fake, fake)
            + adversarial_loss(validity2_real, valid)
            + adversarial_loss(validity2_fake, fake)
        ) / 4

        optimizer_D.step(d_loss)
        if i % 50 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader1), d_loss.data[0], g_loss.data[0])
            )

        batches_done = epoch * len(dataloader1) + i
        if batches_done % opt.sample_interval == 0:
            gen_imgs = torch.cat((torch.Tensor(gen_imgs1.numpy()), torch.Tensor(gen_imgs2.numpy())), 0)
            save_image(gen_imgs, "images/%d.png" % batches_done, nrow=8, normalize=True)
