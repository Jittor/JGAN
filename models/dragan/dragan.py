
import jittor as jt
from jittor import init
from jittor import nn
from jittor.dataset.mnist import MNIST
import jittor.transform as transform
import argparse
import os
import numpy as np
import math
import cv2
import time

jt.flags.use_cuda = 1
os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=1000, help='interval between image sampling')
opt = parser.parse_args()
print(opt)

def save_image(img, path, nrow=10):
    N,C,W,H = img.shape
    img2=img.reshape([img.shape[1],-1,H])
    img=img2[:,:W*nrow,:]
    for i in range(1,int(img2.shape[1]/W/nrow)):
        img=np.concatenate([img,img2[:,W*nrow*i:W*nrow*(i+1),:]],axis=2)
    img=(img+1.0)/2.0*255
    img=img.transpose((1,2,0))
    cv2.imwrite(path,img)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        jt.init.gauss_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        jt.init.gauss_(m.weight, 1.0, 0.02)
        jt.init.constant_(m.bias, 0.0)

class BCELoss(nn.Module):
    def __init__(self):
        pass
    def execute(self, output, target):
        return - (target * jt.log(jt.maximum(output, 1e-20)) + (1 - target) * jt.log(jt.maximum(1 - output, 1e-20))).mean()

class Upsample(nn.Module):
    def __init__(self, scale_factor=None, mode='nearest'):
        self.scale_factor = scale_factor if isinstance(scale_factor, tuple) else (scale_factor, scale_factor)
        self.mode = mode
    
    def execute(self, x):
        return nn.resize(x, size=(x.shape[2]*self.scale_factor[0], x.shape[3]*self.scale_factor[1]), mode=self.mode)

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = (opt.img_size // 4)
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, (128 * (self.init_size ** 2))))
        self.conv_blocks = nn.Sequential(nn.BatchNorm(128), Upsample(scale_factor=2), nn.Conv(128, 128, 3, stride=1, padding=1), nn.BatchNorm(128, eps=0.8), nn.LeakyReLU(scale=0.2), Upsample(scale_factor=2), nn.Conv(128, 64, 3, stride=1, padding=1), nn.BatchNorm(64, eps=0.8), nn.LeakyReLU(scale=0.2), nn.Conv(64, opt.channels, 3, stride=1, padding=1), nn.Tanh())
        for m in self.conv_blocks:
            weights_init_normal(m)

    def execute(self, noise):
        out = self.l1(noise)
        out = out.view((out.shape[0], 128, self.init_size, self.init_size))
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv(in_filters, out_filters, 3, stride=2, padding=1), nn.LeakyReLU(scale=0.2), nn.Dropout(p=0.25)]
            if bn:
                block.append(nn.BatchNorm(out_filters, eps=0.8))
            for m in block:
                weights_init_normal(m)
            return block
        self.model = nn.Sequential(*discriminator_block(opt.channels, 16, bn=False), *discriminator_block(16, 32), *discriminator_block(32, 64), *discriminator_block(64, 128))
        ds_size = (opt.img_size // (2 ** 4))
        self.adv_layer = nn.Sequential(nn.Linear((128 * (ds_size ** 2)), 1), nn.Sigmoid())

    def execute(self, img):
        out = self.model(img)
        out = out.view((out.shape[0], (- 1)))
        validity = self.adv_layer(out)
        return validity

adversarial_loss = BCELoss()
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Configure data loader
transform = transform.Compose([
    transform.Resize(size=opt.img_size),
    transform.Gray(),
    transform.ImageNormalize(mean=[0.5], std=[0.5]),
])
dataloader = MNIST(train=True, transform=transform).set_attrs(batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = jt.nn.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = jt.nn.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

def std(x):
    return jt.sqrt((x-x.mean()).sqr().mean().maximum(1e-6))

def norm(x, k, dim):
    assert k==2 or k==1
    if k==1:
        return x.abs().sum(dim)
    if k==2:
        return jt.sqrt((x.sqr()).sum(dim).maximum(1e-6))

def compute_gradient_penalty(D, X):
    'Calculates the gradient penalty loss for DRAGAN'
    alpha = jt.array(np.random.random(size=X.shape).astype(np.float32)).stop_grad()
    interpolates = ((alpha * X) + ((1 - alpha) * (X + ((0.5 * std(X)) * jt.random(X.shape)))))
    d_interpolates = D(interpolates)
    gradients = jt.grad(d_interpolates, interpolates)
    gradient_penalty = (lambda_gp * ((norm(gradients, 2, dim=1) - 1).sqr()).mean())
    return gradient_penalty

warmup_times = 300
run_times = 3000
total_time = 0.
cnt = 0

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for (i, (real_imgs, _)) in enumerate(dataloader):
        valid = jt.ones([real_imgs.shape[0], 1]).stop_grad()
        fake = jt.zeros([real_imgs.shape[0], 1]).stop_grad()

        # -----------------
        #  Train Generator
        # -----------------
        
        z = jt.array(np.random.normal(0, 1, (real_imgs.shape[0], opt.latent_dim)).astype(np.float32))
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        optimizer_G.step(g_loss)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs)
        d_loss = (real_loss + fake_loss) / 2 + gradient_penalty
        optimizer_D.step(d_loss)
        
        if warmup_times==-1:
            print(('[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]' % (epoch, opt.n_epochs, i, len(dataloader), d_loss.numpy()[0], g_loss.numpy()[0])))
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
    if warmup_times==-1:
        save_image(gen_imgs.data, ('images/%d.png' % epoch), nrow=int(math.sqrt(opt.batch_size)))
