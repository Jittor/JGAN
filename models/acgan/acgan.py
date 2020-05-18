
import jittor as jt
from jittor import init
from jittor import nn
from jittor.dataset.mnist import MNIST
import jittor.transform as transform
import argparse
import os, sys
import numpy as np
import math
import cv2
import time
import random

jt.flags.use_cuda = 1
os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
opt = parser.parse_args()
print(opt)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        jt.init.gauss_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        jt.init.gauss_(m.weight, 1.0, 0.02)
        jt.init.constant_(m.bias, 0.0)

def save_image(img, path):
    img2=img.reshape([-1,3200,32])
    img=img2[:,:320,:]
    for i in range(1,10):
        img=np.concatenate([img,img2[:,320*i:320*(i+1),:]],axis=2)
    print(img.shape)
    img=(img+1.0)/2.0*255
    img=img.transpose((1,2,0))
    cv2.imwrite(path,img)

def mul(a, b):
    return a*b

class Upsample(nn.Module):
    def __init__(self, scale_factor=None, mode='nearest'):
        self.scale_factor = scale_factor if isinstance(scale_factor, tuple) else (scale_factor, scale_factor)
        self.mode = mode
    
    def execute(self, x):
        return nn.resize(x, size=(x.shape[2]*self.scale_factor[0], x.shape[3]*self.scale_factor[1]), mode=self.mode)

class Embedding(nn.Module):
    def __init__(self, num, dim):
        self.num = num
        self.dim = dim
        self.weight = jt.init.gauss([num,dim],'float32').stop_grad()
    
    def execute(self, x):
        res = self.weight[x].reshape([x.shape[0],self.dim])
        return res

class BCELoss(nn.Module):
    def __init__(self):
        pass
    def execute(self, output, target):
        return - (target * jt.log(jt.maximum(output, 1e-20)) + (1 - target) * jt.log(jt.maximum(1 - output, 1e-20))).mean()

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        pass
    def execute(self, output, target):
        return nn.cross_entropy_loss(output, target)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = Embedding(opt.n_classes, opt.latent_dim)
        self.init_size = (opt.img_size // 4)
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, (128 * (self.init_size ** 2))))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm(128), 
            Upsample(scale_factor=2), 
            nn.Conv(128, 128, 3, stride=1, padding=1), 
            nn.BatchNorm(128, 0.8), 
            nn.Leaky_relu(0.2),
            Upsample(scale_factor=2), 
            nn.Conv(128, 64, 3, stride=1, padding=1), 
            nn.BatchNorm(64, 0.8), 
            nn.Leaky_relu(0.2),
            nn.Conv(64, opt.channels, 3, stride=1, padding=1)
        )
        self.tanh = nn.Tanh()
        for m in self.conv_blocks:
            weights_init_normal(m)

    def execute(self, noise, labels):
        ebd = self.label_emb(labels)
        gen_input = mul(ebd, noise)
        out = self.l1(gen_input)
        out = jt.reshape(out, [out.shape[0], 128, self.init_size, self.init_size])
        img = self.conv_blocks(out)
        img = self.tanh(img)
        return img

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            'Returns layers of each discriminator block'
            block = [nn.Conv(in_filters, out_filters, 3, 2, 1), nn.Leaky_relu(0.2), nn.Dropout(0.25)]
            print("Conv shape",block[0].weight.shape)
            if bn:
                block.append(nn.BatchNorm(out_filters, 0.8))
            for m in block:
                weights_init_normal(m)
            return block
        self.conv_blocks = nn.Sequential(*discriminator_block(opt.channels, 16, bn=False), *discriminator_block(16, 32), *discriminator_block(32, 64), *discriminator_block(64, 128))
        ds_size = (opt.img_size // (2 ** 4))
        self.adv_layer = nn.Sequential(nn.Linear((128 * (ds_size ** 2)), 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear((128 * (ds_size ** 2)), opt.n_classes), nn.Softmax(dim=1))

    def execute(self, img):
        out = self.conv_blocks(img)
        out = jt.reshape(out, [out.shape[0], (- 1)])
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        return (validity, label)

adversarial_loss = BCELoss()
auxiliary_loss = CrossEntropyLoss()

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
optimizer_G = jt.nn.Adam(generator.parameters(), opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = jt.nn.Adam(discriminator.parameters(), opt.lr, betas=(opt.b1, opt.b2))

def sample_image(n_row, batches_done):
    'Saves a grid of generated digits ranging from 0 to n_classes'
    z = jt.array(np.random.normal(0, 1, ((n_row ** 2), opt.latent_dim)).astype(np.float32))
    labels = np.array([num for _ in range(n_row) for num in range(n_row)]).astype(np.float32)
    labels = jt.array(labels)
    gen_imgs = generator(z, labels)
    gen_imgs = gen_imgs.tanh()
    save_image(gen_imgs.numpy(), ('images/%d.png' % batches_done))

warmup_times = 300
run_times = 3000
total_time = 0.
cnt = 0

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for (i, (real_imgs, labels)) in enumerate(dataloader):
        batch_size = real_imgs.shape[0]
        valid = jt.ones([batch_size, 1]).stop_grad()
        fake = jt.zeros([batch_size, 1]).stop_grad()

        # -----------------
        #  Train Generator
        # -----------------

        z = jt.array(np.random.normal(0, 1, (batch_size, opt.latent_dim)).astype(np.float32))
        gen_labels = jt.array(np.random.randint(0, opt.n_classes, batch_size).astype(np.float32))
        gen_imgs = generator(z, gen_labels)
        (validity, pred_label) = discriminator(gen_imgs)
        g_loss = (0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels)))
        optimizer_G.step(g_loss)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        (real_pred, real_aux) = discriminator(real_imgs)
        d_real_loss = ((adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2)
        (fake_pred, fake_aux) = discriminator(gen_imgs.detach())
        d_fake_loss = ((adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2)
        d_loss = ((d_real_loss + d_fake_loss) / 2)
        optimizer_D.step(d_loss)

        if warmup_times==-1:
            pred = np.concatenate([real_aux.numpy(), fake_aux.numpy()], axis=0)
            gt = np.concatenate([labels.numpy(), gen_labels.numpy()], axis=0)
            d_acc = np.mean((np.argmax(pred, axis=1) == gt))
            print(('[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]' % (epoch, opt.n_epochs, i, len(dataloader), d_loss.mean().data, (100 * d_acc), g_loss.mean().data)))
            batches_done = ((epoch * len(dataloader)) + i)
            if ((batches_done % opt.sample_interval) == 0):
                sample_image(n_row=10, batches_done=batches_done)
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
