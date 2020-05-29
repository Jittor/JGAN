import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

from models import *
from datasets import *
from utils import *

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="monet2photo", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

# Losses
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

# Optimizers
optimizer_G = nn.Adam(
    G_AB.parameters() + G_BA.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_A = nn.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = nn.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
transform_ = [
    transform.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
    transform.RandomCrop((opt.img_height, opt.img_width)),
    transform.RandomHorizontalFlip(),
    transform.ImageNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
]

# Training data loader
dataloader = ImageDataset("../data/%s" % opt.dataset_name, transform_=transform_, unaligned=True).set_attrs(batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

val_dataloader = ImageDataset("../data/%s" % opt.dataset_name, transform_=transform_, unaligned=True, mode="test").set_attrs(batch_size=5, shuffle=True, num_workers=1)
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
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    G_AB.eval()
    G_BA.eval()


    real_A = imgs[0].stop_grad()
    fake_B = G_AB(real_A)
    real_B = imgs[1].stop_grad()
    fake_A = G_BA(real_B)
    # Arange images along x-axis
    real_A_ = []
    for i in range(5): real_A_.append(real_A.numpy()[i])
    real_A = np.concatenate(real_A_, -1)[np.newaxis,:,:,:]
    real_B_ = []
    for i in range(5): real_B_.append(real_B.numpy()[i])
    real_B = np.concatenate(real_B_, -1)[np.newaxis,:,:,:]
    fake_A_ = []
    for i in range(5): fake_A_.append(fake_A.numpy()[i])
    fake_A = np.concatenate(fake_A_, -1)[np.newaxis,:,:,:]
    fake_B_ = []
    for i in range(5): fake_B_.append(fake_B.numpy()[i])
    fake_B = np.concatenate(fake_B_, -1)[np.newaxis,:,:,:]
    # Arange images along y-axis
    image_grid = np.concatenate((real_A, fake_B, real_B, fake_A), 0)
    save_image(image_grid, "images/%s/%s.png" % (opt.dataset_name, batches_done), 1)


# ----------
#  Training
# ----------

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        real_A = batch[0]
        real_B = batch[1]

        # Adversarial ground truths
        valid = jt.array(np.ones((real_A.size(0), *D_A.output_shape))).float32().stop_grad()
        fake = jt.array(np.zeros((real_A.size(0), *D_A.output_shape))).float32().stop_grad()

        # ------------------
        #  Train Generators
        # ------------------

        G_AB.train()
        G_BA.train()

        # Identity loss
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)

        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity

        optimizer_G.step(loss_G)

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        # Real loss
        loss_real = criterion_GAN(D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_GAN(D_A(fake_A_.stop_grad()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        optimizer_D_A.step(loss_D_A)

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        # Real loss
        loss_real = criterion_GAN(D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        fake_B_.sync()
        loss_fake = criterion_GAN(D_B(fake_B_.stop_grad()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        optimizer_D_B.step(loss_D_B)

        loss_D = (loss_D_A + loss_D_B) / 2

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        
        if i % 50 == 0:
            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.data[0],
                    loss_G.data[0],
                    loss_GAN.data[0],
                    loss_cycle.data[0],
                    loss_identity.data[0],
                    time_left,
                )
            )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)