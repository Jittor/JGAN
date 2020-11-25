"""
StarGAN (CelebA)
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
And the annotations: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAA8YmAHNNU6BEfWMPMfM6r9a/Anno?dl=0&preview=list_attr_celeba.txt
Instructions on running the script:
1. Download the dataset and annotations from the provided link
2. Put images to folder 'img_align_celeba_attr/images'
3. Copy 'list_attr_celeba.txt' to folder 'img_align_celeba_attr'
4. Save the folder 'img_align_celeba_attr' to '../../data/'
5. Run the script by 'python3 stargan.py'
"""

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

from models import *
from datasets import *

jt.flags.use_cuda = 1

os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba_attr", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
parser.add_argument("--residual_blocks", type=int, default=6, help="number of residual blocks in generator")
parser.add_argument(
    "--selected_attrs",
    "--list",
    nargs="+",
    help="selected attributes for the CelebA dataset",
    default=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"],
)
parser.add_argument("--n_critic", type=int, default=5, help="number of training iterations for WGAN discriminator")
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
        # img = img[:,:,::-1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path,img)

c_dim = len(opt.selected_attrs)
img_shape = (opt.channels, opt.img_height, opt.img_width)

# Loss functions
criterion_cycle = nn.L1Loss()
bce_with_logits_loss = nn.BCEWithLogitsLoss(size_average=False)

def criterion_cls(logit, target):
    return bce_with_logits_loss(logit, target) / logit.size(0)

# Loss weights
lambda_cls = 1
lambda_rec = 10
lambda_gp = 10

# Initialize generator and discriminator
generator = GeneratorResNet(img_shape=img_shape, res_blocks=opt.residual_blocks, c_dim=c_dim)
discriminator = Discriminator(img_shape=img_shape, c_dim=c_dim)

# Optimizers
optimizer_G = jt.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = jt.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders
train_transforms = [
    transform.Resize(int(1.12 * opt.img_height), Image.BICUBIC),
    transform.RandomCrop(opt.img_height),
    transform.RandomHorizontalFlip(),
    transform.ImageNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
dataloader = CelebADataset(
        "../../data/%s" % opt.dataset_name, transform_=train_transforms, mode="train", attributes=opt.selected_attrs).set_attrs(
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
val_transforms = [
    transform.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transform.ImageNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
val_dataloader = CelebADataset(
        "../../data/%s" % opt.dataset_name, transform_=train_transforms, mode="val", attributes=opt.selected_attrs).set_attrs(
    batch_size=10,
    shuffle=True,
    num_workers=opt.n_cpu,
)

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = jt.array(np.random.random((real_samples.size(0), 1, 1, 1)).astype(np.float32))
    # Get random interpolation between real and fake samples
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    d_interpolates, _ = D(interpolates)
    # Get gradient w.r.t. interpolates
    gradients = jt.grad(d_interpolates, interpolates)
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


label_changes = [
    ((0, 1), (1, 0), (2, 0)),  # Set to black hair
    ((0, 0), (1, 1), (2, 0)),  # Set to blonde hair
    ((0, 0), (1, 0), (2, 1)),  # Set to brown hair
    ((3, -1),),  # Flip gender
    ((4, -1),),  # Age flip
]


def sample_images(batches_done):
    """Saves a generated sample of domain translations"""
    val_imgs, val_labels = next(iter(val_dataloader))
    val_imgs = jt.array(val_imgs)
    val_labels = jt.array(val_labels)
    img_samples = None
    for i in range(10):
        img, label = val_imgs[i], val_labels[i]
        # Repeat for number of label changes
        imgs = img.broadcast([c_dim, img.shape[0], img.shape[1], img.shape[2]],[0])
        labels = label.broadcast([c_dim, label.shape[0]],[0])
        # Make changes to labels
        for sample_i, changes in enumerate(label_changes):
            for col, val in changes:
                labels[sample_i, col] = 1 - labels[sample_i, col] if val == -1 else val

        # Generate translations
        gen_imgs = generator(imgs, labels).numpy()
        # Concatenate images by width
        gen_imgs = np.concatenate([x for x in gen_imgs], axis=-1)
        img_sample = jt.array(np.concatenate([img.numpy(), gen_imgs], axis=-1))
        # Add as row to generated samples
        img_samples = img_sample if img_samples is None else jt.contrib.concat((img_samples, img_sample), -2)
    save_image(img_samples.view(1, *img_samples.shape).numpy(), "images/%s.png" % batches_done, nrow=1)

warmup_times = -1
run_times = 3000
total_time = 0.
cnt = 0

# ----------
#  Training
# ----------

saved_samples = []
start_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        # Sample labels as generator inputs
        sampled_c = jt.array(np.random.randint(0, 2, (imgs.size(0), c_dim)).astype(np.float32)).stop_grad()
        # Generate fake batch of images
        fake_imgs = generator(imgs, sampled_c)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Real images
        real_validity, pred_cls = discriminator(imgs)
        # Fake images
        fake_validity, _ = discriminator(fake_imgs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, imgs, fake_imgs)
        # Adversarial loss
        loss_D_adv = -jt.mean(real_validity) + jt.mean(fake_validity) + lambda_gp * gradient_penalty
        # Classification loss
        loss_D_cls = criterion_cls(pred_cls, labels)
        # Total loss
        loss_D = loss_D_adv + lambda_cls * loss_D_cls

        optimizer_D.step(loss_D)

        # Every n_critic times update generator
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Translate and reconstruct image
            gen_imgs = generator(imgs, sampled_c)
            recov_imgs = generator(gen_imgs, labels)
            # Discriminator evaluates translated image
            fake_validity, pred_cls = discriminator(gen_imgs)
            # Adversarial loss
            loss_G_adv = -jt.mean(fake_validity)
            # Classification loss
            loss_G_cls = criterion_cls(pred_cls, sampled_c)
            # Reconstruction loss
            loss_G_rec = criterion_cycle(recov_imgs, imgs)
            # Total loss
            loss_G = loss_G_adv + lambda_cls * loss_G_cls + lambda_rec * loss_G_rec

            optimizer_G.step(loss_G)

            if warmup_times==-1:
                # --------------
                #  Log Progress
                # --------------

                # Determine approximate time left
                batches_done = epoch * len(dataloader) + i
                batches_left = opt.n_epochs * len(dataloader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - start_time) / (batches_done + 1))

                # Print log
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D adv: %f, aux: %f] [G loss: %f, adv: %f, aux: %f, cycle: %f] ETA: %s"
                    % (
                        epoch,
                        opt.n_epochs,
                        i,
                        len(dataloader),
                        loss_D_adv.numpy()[0],
                        loss_D_cls.numpy()[0],
                        loss_G.numpy()[0],
                        loss_G_adv.numpy()[0],
                        loss_G_cls.numpy()[0],
                        loss_G_rec.numpy()[0],
                        time_left,
                    )
                )

                # If at sample interval sample and save image
                if batches_done % opt.sample_interval == 0:
                    sample_images(batches_done)
        if warmup_times!=-1:
            jt.sync_all()
            cnt += 1
            print(cnt)
            if cnt == warmup_times:
                jt.sync_all(True)
                sta = time.time()
            if cnt > warmup_times + run_times:
                jt.sync_all(True)
                total_time = time.time() - sta
                print(f"run {run_times} iters cost {total_time} seconds, and avg {total_time / run_times} one iter.")
                exit(0)
