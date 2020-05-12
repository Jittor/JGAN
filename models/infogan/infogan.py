import argparse
import os
import numpy as np
import math
import itertools

os.makedirs("images/static/", exist_ok=True)
os.makedirs("images/varying_c1/", exist_ok=True)
os.makedirs("images/varying_c2/", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
parser.add_argument("--code_dim", type=int, default=2, help="latent code")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

import jittor as jt
from jittor import init
from jittor import nn

def weights_init_normal(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') != (- 1)):
        init.gauss_(m.weight, mean=0.0, std=0.02)
    elif (classname.find('BatchNorm') != (- 1)):
        init.gauss_(m.weight, mean=1.0, std=0.02)
        init.constant_(m.bias, value=0.0)


def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    return jt.array(y_cat).float32()

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        input_dim = ((opt.latent_dim + opt.n_classes) + opt.code_dim)
        self.init_size = (opt.img_size // 4)
        self.l1 = nn.Sequential(nn.Linear(input_dim, (128 * (self.init_size ** 2))))
        self.conv_blocks = nn.Sequential(nn.BatchNorm(128), nn.Upsample(scale_factor=2), nn.Conv(128, 128, 3, stride=1, padding=1), nn.BatchNorm(128, eps=0.8), nn.LeakyReLU(scale=0.2), nn.Upsample(scale_factor=2), nn.Conv(128, 64, 3, stride=1, padding=1), nn.BatchNorm(64, eps=0.8), nn.LeakyReLU(scale=0.2), nn.Conv(64, opt.channels, 3, stride=1, padding=1), nn.Tanh())

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, noise, labels, code):
        gen_input = jt.contrib.concat((noise, labels, code), dim=1)
        out = self.l1(gen_input)
        out = out.view((out.shape[0], 128, self.init_size, self.init_size))
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            'Returns layers of each discriminator block'
            block = [nn.Conv(in_filters, out_filters, 3, stride=2, padding=1), nn.LeakyReLU(scale=0.2), nn.Dropout(p=0.25)]
            if bn:
                block.append(nn.BatchNorm(out_filters, eps=0.8))
            return block
        self.conv_blocks = nn.Sequential(*discriminator_block(opt.channels, 16, bn=False), *discriminator_block(16, 32), *discriminator_block(32, 64), *discriminator_block(64, 128))
        ds_size = (opt.img_size // (2 ** 4))
        self.adv_layer = nn.Sequential(nn.Linear((128 * (ds_size ** 2)), 1))
        self.aux_layer = nn.Sequential(nn.Linear((128 * (ds_size ** 2)), opt.n_classes), nn.Softmax())
        self.latent_layer = nn.Sequential(nn.Linear((128 * (ds_size ** 2)), opt.code_dim))

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, img):
        out = self.conv_blocks(img)
        out = out.view((out.shape[0], (- 1)))
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)
        return (validity, label, latent_code)


# Loss functions
adversarial_loss = nn.MSELoss()
categorical_loss = nn.CrossEntropyLoss()
continuous_loss = nn.MSELoss()

# Loss weights
lambda_cat = 1
lambda_con = 0.1

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
optimizer_info = nn.Adam(
    itertools.chain(generator.parameters(), discriminator.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)

# Static generator inputs for sampling
static_z = jt.array(np.zeros((opt.n_classes ** 2, opt.latent_dim))).float32()
static_label = to_categorical(
    np.array([num for _ in range(opt.n_classes) for num in range(opt.n_classes)]), num_columns=opt.n_classes
)
static_code = jt.array(np.zeros((opt.n_classes ** 2, opt.code_dim))).float32()

import cv2
def save_image(img, path, nrow=10):
    N,C,W,H = img.shape
    if N > nrow * nrow:
        img = img[:nrow*nrow,:]
    elif N < nrow * nrow:
        img = np.concatenate([img, np.zeros(nrow*nrow-N,C,W,H)],axis=0)
    img2 = img.reshape([-1,W*nrow*nrow,H])
    img = img2[:,:W*nrow,:]
    for i in range(1,nrow):
        img = np.concatenate([img,img2[:,W*nrow*i:W*nrow*(i+1),:]],axis=2)
    min_ = img.min()
    max_ = img.max()
    img = (img - min_) / (max_ - min_) * 255
    img = img.transpose((1,2,0))
    cv2.imwrite(path,img)

def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Static sample
    z = jt.array(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))).float32()
    static_sample = generator(z, static_label, static_code)
    save_image(static_sample.numpy(), "images/static/%d.png" % batches_done, nrow=n_row)

    # Get varied c1 and c2
    zeros = np.zeros((n_row ** 2, 1))
    c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
    c1 = jt.array(np.concatenate((c_varied, zeros), -1)).float32()
    c2 = jt.array(np.concatenate((zeros, c_varied), -1)).float32()
    sample1 = generator(static_z, static_label, c1)
    sample2 = generator(static_z, static_label, c2)
    save_image(sample1.numpy(), "images/varying_c1/%d.png" % batches_done, nrow=n_row)
    save_image(sample2.numpy(), "images/varying_c2/%d.png" % batches_done, nrow=n_row)


# ----------
#  Training
# ----------
from pdb import set_trace as st
for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = jt.ones((batch_size, 1)).float32().stop_grad()
        fake = jt.zeros((batch_size, 1)).float32().stop_grad()

        # Configure input
        real_imgs = jt.array(imgs).float32()
        labels = to_categorical(labels.numpy(), num_columns=opt.n_classes)

        # -----------------
        #  Train Generator
        # -----------------

        # Sample noise and labels as generator input
        z = jt.array(np.random.normal(0, 1, (batch_size, opt.latent_dim))).float32()
        label_input = to_categorical(np.random.randint(0, opt.n_classes, batch_size), num_columns=opt.n_classes)
        code_input = jt.array(np.random.uniform(-1, 1, (batch_size, opt.code_dim))).float32()

        # Generate a batch of images
        gen_imgs = generator(z, label_input, code_input)

        # Loss measures generator's ability to fool the discriminator
        validity, _, _ = discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, valid)
        optimizer_G.step(g_loss)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Loss for real images
        real_pred, _, _ = discriminator(real_imgs)
        d_real_loss = adversarial_loss(real_pred, valid)

        # Loss for fake images
        fake_pred, _, _ = discriminator(gen_imgs.detach())
        d_fake_loss = adversarial_loss(fake_pred, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        optimizer_D.step(d_loss)

        # ------------------
        # Information Loss
        # ------------------

        # Sample labels
        sampled_labels = np.random.randint(0, opt.n_classes, batch_size)

        # Ground truth labels
        gt_labels = jt.array(sampled_labels).float32().stop_grad()

        # Sample noise, labels and code as generator input
        z = jt.array(np.random.normal(0, 1, (batch_size, opt.latent_dim))).float32().stop_grad()
        label_input = to_categorical(sampled_labels, num_columns=opt.n_classes)
        code_input = jt.array(np.random.uniform(-1, 1, (batch_size, opt.code_dim))).float32().stop_grad()

        gen_imgs = generator(z, label_input, code_input)
        _, pred_label, pred_code = discriminator(gen_imgs)

        info_loss = lambda_cat * categorical_loss(pred_label, gt_labels) + lambda_con * continuous_loss(
            pred_code, code_input
        )

        optimizer_info.step(info_loss)

        # --------------
        # Log Progress
        # --------------

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.data[0], g_loss.data[0], info_loss.data[0])
        )
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)
