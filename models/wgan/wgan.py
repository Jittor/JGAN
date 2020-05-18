
import jittor as jt
from jittor import init
from jittor import nn
from jittor.dataset.mnist import MNIST
import jittor.transform as transform
import argparse
import os
import numpy as np
import math
import sys
import cv2

jt.flags.use_cuda = 1

os.makedirs('jittor_images', exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=5e-05, help='learning rate')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--n_critic', type=int, default=5, help='number of training steps for discriminator per iter')
parser.add_argument('--clip_value', type=float, default=0.01, help='lower and upper clip value for disc. weights')
parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
opt = parser.parse_args()
print(opt)
img_shape = (opt.channels, opt.img_size, opt.img_size)

def save_image(img, path, nrow=10):
    N,C,W,H = img.shape
    img2=img.reshape([-1,W*nrow*nrow,H])
    img=img2[:,:W*nrow,:]
    for i in range(1,nrow):
        img=np.concatenate([img,img2[:,W*nrow*i:W*nrow*(i+1),:]],axis=2)
    img=(img+1.0)/2.0*255
    img=img.transpose((1,2,0))
    cv2.imwrite(path,img)

def clamp_(var, l, r):
    var.assign(var.maximum(l).minimum(r))

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
        
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(scale=0.2))
            return layers
        self.model = nn.Sequential(*block(opt.latent_dim, 128, normalize=False), *block(128, 256), *block(256, 512), *block(512, 1024), nn.Linear(1024, int(np.prod(img_shape))), nn.Tanh())

    def execute(self, z):
        img = self.model(z)
        img = img.view((img.shape[0], *img_shape))
        return img

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(nn.Linear(int(np.prod(img_shape)), 512), nn.LeakyReLU(scale=0.2), nn.Linear(512, 256), nn.LeakyReLU(scale=0.2), nn.Linear(256, 1))

    def execute(self, img):
        img_flat = img.view((img.shape[0], (- 1)))
        validity = self.model(img_flat)
        return validity

generator = Generator()
discriminator = Discriminator()
transform = transform.Compose([
    transform.Resize(size=opt.img_size),
    transform.Gray(),
    transform.ImageNormalize(mean=[0.5], std=[0.5]),
])
dataloader = MNIST(train=True, transform=transform).set_attrs(batch_size=opt.batch_size, shuffle=True)
#TODO:RMSprop
optimizer_G = jt.nn.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = jt.nn.RMSprop(discriminator.parameters(), lr=opt.lr)
batches_done = 0

for epoch in range(opt.n_epochs):
    for (i, (real_imgs, _)) in enumerate(dataloader):
        z = jt.array(np.random.normal(0, 1, (real_imgs.shape[0], opt.latent_dim)).astype(np.float32))
        fake_imgs = generator(z).detach()
        loss_D = ((- jt.mean(discriminator(real_imgs))) + jt.mean(discriminator(fake_imgs)))
        optimizer_D.step(loss_D)
        for p in discriminator.parameters():
            clamp_(p, - opt.clip_value, opt.clip_value)

        if ((i % opt.n_critic) == 0):
            gen_imgs = generator(z)
            loss_G = (- jt.mean(discriminator(gen_imgs)))
            optimizer_G.step(loss_G)
            print(('[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]' % (epoch, opt.n_epochs, (batches_done % len(dataloader)), len(dataloader), loss_D.numpy()[0], loss_G.numpy()[0])))
        if ((batches_done % opt.sample_interval) == 0):
            save_image(gen_imgs.data[:25], ('jittor_images/%d.png' % batches_done), nrow=5)
        batches_done += 1
