import argparse
import os
import numpy as np
import datetime
import time
import sys
from models import *
from datasets import *
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="edges2shoes", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--latent_dim", type=int, default=8, help="number of latent codes")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
parser.add_argument("--lambda_pixel", type=float, default=10, help="pixelwise loss weight")
parser.add_argument("--lambda_latent", type=float, default=0.5, help="latent loss weight")
parser.add_argument("--lambda_kl", type=float, default=0.01, help="kullback-leibler loss weight")
opt = parser.parse_args()
print(opt)

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Loss functions
mae_loss = nn.L1Loss()

# Initialize generator, encoder and discriminators
generator = Generator(opt.latent_dim, input_shape)
encoder = Encoder(opt.latent_dim, input_shape)
D_VAE = MultiDiscriminator(input_shape)
D_LR = MultiDiscriminator(input_shape)

# Optimizers
optimizer_G = nn.Adam(encoder.parameters() + generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_VAE = nn.Adam(D_VAE.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_LR = nn.Adam(D_LR.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

dataloader = ImageDataset("../../../PyTorch-GAN/data/%s" % opt.dataset_name,   input_shape).set_attrs(batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

valdataloader = ImageDataset("../../../PyTorch-GAN/data/%s" % opt.dataset_name,   input_shape, mode="val").set_attrs(batch_size=8, shuffle=False, num_workers=1)


def reparameterization(mu, logvar):
    std = jt.exp(logvar / 2)
    sampled_z = jt.array(np.random.normal(0, 1, (mu.shape[0], opt.latent_dim))).float32()
    z = sampled_z * std + mu
    return z

def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    generator.eval()
    img_samples = None
    imgs = next(iter(dataloader))
    for idx in range(8):
        img_A = jt.array(imgs[0][idx])
        real_A = img_A.reindex([opt.latent_dim, *img_A.shape], ["i1", "i2", "i3"])
        # Sample latent representations
        sampled_z = jt.array(np.random.normal(0, 1, (opt.latent_dim, opt.latent_dim))).float32()
        # Generate samples
        fake_B = generator(real_A, sampled_z)
        # Concatenate samples horisontally
        fake_B_ = []
        for i in range(fake_B.size(0)): fake_B_.append(fake_B.numpy()[i])
        fake_B = np.concatenate(fake_B_, -1)
        img_sample = np.concatenate((img_A.numpy(), fake_B), -1)[np.newaxis,:]
        # Concatenate with previous samples vertically
        img_samples = img_sample if img_samples is None else np.concatenate((img_samples, img_sample), -2)
    min_, max_ = img_samples.min(), img_samples.max()
    img_samples = (img_samples[0] - min_) / (max_ - min_) * 255.
    img_samples = img_samples.transpose((1,2,0))
    cv2.imwrite("images/%s/%s.png" % (opt.dataset_name, batches_done), img_samples)
    generator.train()

# ----------
#  Training
# ----------

# Adversarial loss
valid = 1
fake = 0

prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = batch[0].stop_grad()
        real_B = batch[1].stop_grad()
        # -------------------------------
        #  Train Generator and Encoder
        # -------------------------------

        # ----------
        # cVAE-GAN
        # ----------

        # Produce output using encoding of B (cVAE-GAN)
        mu, logvar = encoder(real_B)
        encoded_z = reparameterization(mu, logvar)
        fake_B = generator(real_A, encoded_z)

        # Pixelwise loss of translated image by VAE
        loss_pixel = mae_loss(fake_B, real_B)
        # Kullback-Leibler divergence of encoded B
        loss_kl = 0.5 * jt.sum(jt.exp(logvar) + mu.sqr() - logvar - 1)
        # Adversarial loss
        loss_VAE_GAN = D_VAE.compute_loss(fake_B, valid)

        # ---------
        # cLR-GAN
        # ---------

        # Produce output using sampled z (cLR-GAN)
        sampled_z = jt.array(np.random.normal(0, 1, (real_A.shape[0], opt.latent_dim))).float32()
        _fake_B = generator(real_A, sampled_z)
        # cLR Loss: Adversarial loss
        loss_LR_GAN = D_LR.compute_loss(_fake_B, valid)

        # ----------------------------------
        # Total Loss (Generator + Encoder)
        # ----------------------------------
        loss_GE = loss_VAE_GAN + loss_LR_GAN + opt.lambda_pixel * loss_pixel + opt.lambda_kl * loss_kl
        # loss_GE.sync()
        # optimizer_E.step(loss_GE)


        # ---------------------
        # Generator Only Loss
        # ---------------------

        # Latent L1 loss
        _mu, _ = encoder(_fake_B)
        loss_latent = opt.lambda_latent * mae_loss(_mu, sampled_z) + loss_GE
        loss_latent.sync()
        optimizer_G.step(loss_latent)

        # ----------------------------------
        #  Train Discriminator (cVAE-GAN)
        # ----------------------------------
        loss_D_VAE = D_VAE.compute_loss(real_B, valid) + D_VAE.compute_loss(fake_B.stop_grad(), fake)
        loss_D_VAE.sync()
        optimizer_D_VAE.step(loss_D_VAE)

        # ---------------------------------
        #  Train Discriminator (cLR-GAN)
        # ---------------------------------

        loss_D_LR = D_LR.compute_loss(real_B, valid) + D_LR.compute_loss(_fake_B.stop_grad(), fake)
        loss_D_LR.sync()
        optimizer_D_LR.step(loss_D_LR)

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        if i % 10 == 0:
            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [D VAE_loss: %f, LR_loss: %f] [G loss: %f, pixel: %f, kl: %f, latent: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D_VAE.data[0],
                    loss_D_LR.data[0],
                    loss_GE.data[0],
                    loss_pixel.data[0],
                    loss_kl.data[0],
                    loss_latent.data[0],
                    time_left,
                )
            )

        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        saved_name = "saved_models"
        generator.save(os.path.join(f"{saved_name}/{opt.dataset_name}/generator_last.pkl"))
        encoder.save(os.path.join(f"{saved_name}/{opt.dataset_name}/encoder_last.pkl"))
        D_VAE.save(os.path.join(f"{saved_name}/{opt.dataset_name}/D_VAE_last.pkl"))
        D_LR.save(os.path.join(f"{saved_name}/{opt.dataset_name}/D_LR_last.pkl"))