import jittor as jt
from jittor import init
from jittor import nn
import jittor.transform as transform
import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import cv2
import time

from models import *
from datasets import *

from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--data_path", type=str, default="./jittor_landscape_200k")
parser.add_argument("--output_path", type=str, default="./results/flickr")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=384, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

def save_image(img, path, nrow=10):
    N,C,W,H = img.shape
    if (N%nrow!=0):
        print("save_image error: N%nrow!=0")
        return
    img=img.transpose((1,0,2,3))
    ncol=int(N/nrow)
    img2=img.reshape([img.shape[0],-1,H])
    img=img2[:,:W*ncol,:]
    for i in range(1,int(img2.shape[1]/W/ncol)):
        img=np.concatenate([img,img2[:,W*ncol*i:W*ncol*(i+1),:]],axis=2)
    min_=img.min()
    max_=img.max()
    img=(img-min_)/(max_-min_)*255
    img=img.transpose((1,2,0))
    if C==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path,img)
    return img

os.makedirs(f"{opt.output_path}/images/", exist_ok=True)
os.makedirs(f"{opt.output_path}/saved_models/", exist_ok=True)

writer = SummaryWriter(opt.output_path)

# Loss functions
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_pixelwise = nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

# Initialize generator and discriminator
generator = UnetGenerator(3, 3, 7, 64, norm_layer=nn.BatchNorm2d, use_dropout=True)
discriminator = Discriminator()

if opt.epoch != 0:
    # Load pretrained models
    generator.load(f"{opt.output_path}/saved_models/generator_{opt.epoch}.pkl")
    discriminator.load(f"{opt.output_path}/saved_models/discriminator_{opt.epoch}.pkl")

# Optimizers
optimizer_G = jt.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = jt.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders
transforms = [
    transform.Resize(size=(opt.img_height, opt.img_width), mode=Image.BICUBIC),
    transform.ToTensor(),
    transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]

dataloader = ImageDataset(opt.data_path, mode="train", transforms=transforms).set_attrs(
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
# val_dataloader = ImageDataset(opt.data_path, mode="val", transforms=transforms).set_attrs(
#     batch_size=10,
#     shuffle=False,
#     num_workers=1,
# )

# @jt.single_process_scope()
# def eval(epoch, writer):
#     cnt = 1
#     os.makedirs(f"{opt.output_path}/images/test_fake_imgs/epoch_{epoch}", exist_ok=True)
#     for i, (_, real_A, photo_id) in enumerate(val_dataloader):
#         fake_B = generator(real_A)
        
#         if i == 0:
#             # visual image result
#             img_sample = np.concatenate([real_A.data, fake_B.data], -2)
#             img = save_image(img_sample, f"{opt.output_path}/images/epoch_{epoch}_sample.png", nrow=5)
#             writer.add_image('val/image', img.transpose(2,0,1), epoch)

#         fake_B = ((fake_B + 1) / 2 * 255).numpy().astype('uint8')
#         for idx in range(fake_B.shape[0]):
#             cv2.imwrite(f"{opt.output_path}/images/test_fake_imgs/epoch_{epoch}/{photo_id[idx]}.jpg", fake_B[idx].transpose(1,2,0)[:,:,::-1])
#             cnt += 1

warmup_times = -1
run_times = 3000
total_time = 0.
cnt = 0

# ----------
#  Training
# ----------

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, (real_B, real_A, _) in enumerate(dataloader):
        # Adversarial ground truths
        valid = jt.ones([real_A.shape[0], 1]).stop_grad()
        fake = jt.zeros([real_A.shape[0], 1]).stop_grad()
        fake_B = generator(real_A)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        start_grad(discriminator)
        fake_AB = jt.contrib.concat((real_A, fake_B), 1) 
        pred_fake = discriminator(fake_AB.detach())
        loss_D_fake = criterion_GAN(pred_fake, False)
        real_AB = jt.contrib.concat((real_A, real_B), 1)
        pred_real = discriminator(real_AB)
        loss_D_real = criterion_GAN(pred_real, True)
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        optimizer_D.step(loss_D)
        writer.add_scalar('train/loss_D', loss_D.item(), epoch * len(dataloader) + i)

        # ------------------
        #  Train Generators
        # ------------------
        stop_grad(discriminator)        
        fake_AB = jt.contrib.concat((real_A, fake_B), 1) 
        pred_fake = discriminator(fake_AB)
        loss_G_GAN = criterion_GAN(pred_fake, True)
        loss_G_L1 = criterion_pixelwise(fake_B, real_B)
        loss_G = loss_G_GAN + lambda_pixel * loss_G_L1
        optimizer_G.step(loss_G)
        writer.add_scalar('train/loss_G', loss_G.item(), epoch * len(dataloader) + i)

        jt.sync_all(True)

        if jt.rank == 0:
            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            jt.sync_all()
            if batches_done % 5 == 0:
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
                    % (
                        epoch,
                        opt.n_epochs,
                        i,
                        len(dataloader),
                        loss_D.numpy()[0],
                        loss_G.numpy()[0],
                        loss_G_L1.numpy()[0],
                        loss_G_GAN.numpy()[0],
                        time_left,
                    )   
                )

    if jt.rank == 0 and opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # eval(epoch, writer)
        # Save model checkpoints
        generator.save(os.path.join(f"{opt.output_path}/saved_models/generator_{epoch}.pkl"))
        discriminator.save(os.path.join(f"{opt.output_path}/saved_models/discriminator_{epoch}.pkl"))
