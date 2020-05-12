import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys
from torchvision.utils import save_image
from models import *
from datasets import *
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="edges2shoes", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--n_residual", type=int, default=3, help="number of residual blocks in encoder / decoder")
parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")
parser.add_argument("--style_dim", type=int, default=8, help="dimensionality of the style code")
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

criterion_recon = nn.L1Loss()

# Initialize encoders, generators and discriminators
Enc1 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
Dec1 = Decoder(dim=opt.dim, n_upsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
Enc2 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
Dec2 = Decoder(dim=opt.dim, n_upsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
D1 = MultiDiscriminator()
D2 = MultiDiscriminator()

# Loss weights
lambda_gan = 1
lambda_id = 10
lambda_style = 1
lambda_cont = 1
lambda_cyc = 0

# Optimizers
optimizer_G = nn.Adam(
    itertools.chain(Enc1.parameters(), Dec1.parameters(), Enc2.parameters(), Dec2.parameters()),
    lr=opt.lr,
    betas=(opt.b1, opt.b2),
)
optimizer_D1 = nn.Adam(D1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D2 = nn.Adam(D2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders
transform_ = [
    transform.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = ImageDataset("../../../PyTorch-GAN/data/%s" % opt.dataset_name, transform_=transform_).set_attrs(batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

val_dataloader = ImageDataset("../../../PyTorch-GAN/data/%s" % opt.dataset_name, transform_=transform_, mode="val").set_attrs(batch_size=5, shuffle=True, num_workers=1)

def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    img_samples = None
    for img1, img2 in zip(imgs[0], imgs[1]):
        # Create copies of image
        X1 = img1.unsqueeze(0).repeat(opt.style_dim, 1, 1, 1)
        X1 = Variable(X1.type(Tensor))
        # Get random style codes
        s_code = np.random.uniform(-1, 1, (opt.style_dim, opt.style_dim))
        s_code = Variable(Tensor(s_code))
        # Generate samples
        c_code_1, _ = Enc1(X1)
        X12 = Dec2(c_code_1, s_code)
        # Concatenate samples horisontally
        X12 = torch.cat([x for x in X12.data.cpu()], -1)
        img_sample = torch.cat((img1, X12), -1).unsqueeze(0)
        # Concatenate with previous samples vertically
        img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2)
    save_image(img_samples, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True)


# ----------
#  Training
# ----------

# Adversarial ground truths
valid = 1
fake = 0

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        X1 = batch[0]
        X2 = batch[1]

        # Sampled style codes
        style_1 = jt.array(np.random.randn(X1.size(0), opt.style_dim, 1, 1)).float32()
        style_2 = jt.array(np.random.randn(X1.size(0), opt.style_dim, 1, 1)).float32()

        # -------------------------------
        #  Train Encoders and Generators
        # -------------------------------

        # Get shared latent representation
        c_code_1, s_code_1 = Enc1(X1)
        c_code_2, s_code_2 = Enc2(X2)

        # Reconstruct images
        X11 = Dec1(c_code_1, s_code_1)
        X22 = Dec2(c_code_2, s_code_2)

        # Translate images
        X21 = Dec1(c_code_2, style_1)
        X12 = Dec2(c_code_1, style_2)

        # Cycle translation
        c_code_21, s_code_21 = Enc1(X21)
        c_code_12, s_code_12 = Enc2(X12)
        X121 = Dec1(c_code_12, s_code_1) if lambda_cyc > 0 else 0
        X212 = Dec2(c_code_21, s_code_2) if lambda_cyc > 0 else 0

        # Losses
        loss_GAN_1 = lambda_gan * D1.compute_loss(X21, valid)
        loss_GAN_2 = lambda_gan * D2.compute_loss(X12, valid)
        loss_ID_1 = lambda_id * criterion_recon(X11, X1)
        loss_ID_2 = lambda_id * criterion_recon(X22, X2)
        loss_s_1 = lambda_style * criterion_recon(s_code_21, style_1)
        loss_s_2 = lambda_style * criterion_recon(s_code_12, style_2)
        loss_c_1 = lambda_cont * criterion_recon(c_code_12, c_code_1.stop_grad())
        loss_c_2 = lambda_cont * criterion_recon(c_code_21, c_code_2.stop_grad())
        loss_cyc_1 = lambda_cyc * criterion_recon(X121, X1) if lambda_cyc > 0 else 0
        loss_cyc_2 = lambda_cyc * criterion_recon(X212, X2) if lambda_cyc > 0 else 0

        # Total loss
        loss_G = (
            loss_GAN_1
            + loss_GAN_2
            + loss_ID_1
            + loss_ID_2
            + loss_s_1
            + loss_s_2
            + loss_c_1
            + loss_c_2
            + loss_cyc_1
            + loss_cyc_2
        )

        optimizer_G.step(loss_G)

        # -----------------------
        #  Train Discriminator 1
        # -----------------------


        loss_D1 = D1.compute_loss(X1, valid) + D1.compute_loss(X21.stop_grad(), fake)

        optimizer_D1.step(loss_D1)

        # -----------------------
        #  Train Discriminator 2
        # -----------------------


        loss_D2 = D2.compute_loss(X2, valid) + D2.compute_loss(X12.stop_grad(), fake)

        optimizer_D2.step(loss_D2)

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s"
            % (epoch, opt.n_epochs, i, len(dataloader), (loss_D1 + loss_D2).data[0], loss_G.data[0], time_left)
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    if epoch >= opt.decay_epoch:
        optimizer_G.lr = opt.lr * (opt.n_epochs - epoch - 1) / (opt.n_epochs - opt.decay_epoch)
        optimizer_D1.lr = opt.lr * (opt.n_epochs - epoch - 1) / (opt.n_epochs - opt.decay_epoch)
        optimizer_D2.lr = opt.lr * (opt.n_epochs - epoch - 1) / (opt.n_epochs - opt.decay_epoch)