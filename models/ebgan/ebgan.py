import jittor as jt
from jittor import init
import argparse
import os
import numpy as np
import math
from jittor.dataset.mnist import MNIST
import jittor.transform as transform
from jittor import nn
import cv2
import time

jt.flags.use_cuda = 1
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
parser.add_argument("--sample_interval", type=int, default=400, help="number of image channels")
opt = parser.parse_args()

# Configure data loader
transforms = transform.Compose([
    transform.Resize(opt.img_size),
    transform.Gray(),
    transform.ImageNormalize(mean=[0.5],std=[0.5]),
])
dataloader = MNIST(train=True,transform=transforms).set_attrs(batch_size=opt.batch_size,shuffle=True)

def save_image(img, path, nrow=10):
    N,C,W,H = img.shape
    img2=img.reshape([-1,W*nrow*nrow,H])
    img=img2[:,:W*nrow,:]
    for i in range(1,nrow):
        img=np.concatenate([img,img2[:,W*nrow*i:W*nrow*(i+1),:]],axis=2)
    min_=img.min()
    max_=img.max()
    img=(img-min_)/(max_-min_)*255
    img=img.transpose((1,2,0))
    cv2.imwrite(path,img)

class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=None, is_train=True, sync=True):
        assert affine == None

        self.sync = sync
        self.num_features = num_features
        self.is_train = is_train
        self.eps = eps
        self.momentum = momentum
        self.weight = init.constant((num_features,), "float32", 1.0)
        self.bias = init.constant((num_features,), "float32", 0.0)
        self.running_mean = init.constant((num_features,), "float32", 0.0).stop_grad()
        self.running_var = init.constant((num_features,), "float32", 1.0).stop_grad()

    def execute(self, x):
        if self.is_train:
            xmean = jt.mean(x, dims=[0], keepdims=1)
            x2mean = jt.mean(x*x, dims=[0], keepdims=1)
            if self.sync and jt.mpi:
                xmean = xmean.mpi_all_reduce("mean")
                x2mean = x2mean.mpi_all_reduce("mean")

            xvar = x2mean-xmean*xmean
            norm_x = (x-xmean)/jt.sqrt(xvar+self.eps)
            self.running_mean += (xmean.sum([0])-self.running_mean)*self.momentum
            self.running_var += (xvar.sum([0])-self.running_var)*self.momentum
        else:
            running_mean = self.running_mean.broadcast(x, [0])
            running_var = self.running_var.broadcast(x, [0])
            norm_x = (x-running_mean)/jt.sqrt(running_var+self.eps)
        w = self.weight.broadcast(x, [0])
        b = self.bias.broadcast(x, [0])
        return norm_x * w + b

def weights_init_normal(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') != (- 1)):
        init.gauss_(m.weight.data, mean=0.0, std=0.02)
    elif (classname.find('BatchNorm') != (- 1)):
        init.gauss_(m.weight.data, mean=1.0, std=0.02)
        init.constant_(m.bias.data, value=0.0)

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = (opt.img_size // 4)
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, (128 * (self.init_size ** 2))))
        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm(128,0.8),
            nn.Leaky_relu(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm(64, 0.8),
            nn.Leaky_relu(0.2),
            nn.Conv(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def execute(self, noise):
        out = self.l1(noise)
        #print(out)
        out = out.reshape((out.shape[0], 128, self.init_size, self.init_size))
        img = self.conv_blocks(out)
        #print(img)
        return img

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.down = nn.Sequential(nn.Conv(opt.channels, 64, 3, stride=2, padding=1), nn.ReLU())
        self.down_size = (opt.img_size // 2)
        down_dim = (64 * ((opt.img_size // 2) ** 2))
        self.embedding = nn.Linear(down_dim, 32)
        self.fc = nn.Sequential(
            BatchNorm1d(32, 0.8),
            nn.ReLU(),
            nn.Linear(32, down_dim),
            BatchNorm1d(down_dim),
            nn.ReLU()
        )
        self.up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv(64, opt.channels, 3, stride=1, padding=1))

    def execute(self, img):
        out = self.down(img)
        embedding = self.embedding(out.reshape((out.shape[0], (- 1))))
        #print(embedding.shape)
        out = self.fc(embedding)
        out = self.up(out.reshape((out.shape[0], 64, self.down_size, self.down_size)))
        return (out, embedding)

# Reconstruction loss of AE
pixelwise_loss = nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Optimizers
optimizer_G = nn.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = nn.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

def pullaway_loss(embeddings):
    norm = jt.sqrt((embeddings ** 2).sum(1,keepdims=True))
    normalized_emb = embeddings / norm
    similarity = jt.matmul(normalized_emb, normalized_emb.transpose(1, 0))
    batch_size = embeddings.size(0)
    loss_pt = (jt.sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
    return loss_pt

warmup_times = 300
run_times = 3000
total_time = 0.
cnt = 0

# ----------
#  Training
# ----------

# BEGAN hyper parameters
lambda_pt = 0.1
margin = max(1, opt.batch_size / 64.0)
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        # Configure input
        real_imgs = jt.array(imgs).float32()

        # -----------------
        #  Train Generator
        # -----------------

        # Sample noise as generator input
        z = jt.array((np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))).astype('float32'))

        # Generate a batch of images
        gen_imgs = generator(z)
        recon_imgs, img_embeddings = discriminator(gen_imgs)

        # Loss measures generator's ability to fool the discriminator
        g_loss = pixelwise_loss(recon_imgs, gen_imgs.detach()) + lambda_pt * pullaway_loss(img_embeddings)
        g_loss.sync()
        optimizer_G.step(g_loss)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        # Measure discriminator's ability to classify real from generated samples
        real_recon, _ = discriminator(real_imgs)
        fake_recon, _ = discriminator(gen_imgs.detach())

        d_loss_real = pixelwise_loss(real_recon, real_imgs)
        d_loss_fake = pixelwise_loss(fake_recon, gen_imgs.detach())

        d_loss = d_loss_real
        # TODO: remove .data
        if (margin - d_loss_fake.data).item() > 0:
            d_loss += margin - d_loss_fake
        d_loss.sync()
        optimizer_D.step(d_loss)

        if warmup_times==-1:
            # --------------
            # Log Progress
            # --------------

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.data, g_loss.data)
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5)
        else:
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
