import argparse
import os
import numpy as np
import math
import mnistm
from jittor.dataset.mnist import MNIST
import jittor.transform as transform
import jittor as jt
from jittor import init
from jittor import nn
jt.flags.use_cuda = 1

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_residual_blocks", type=int, default=6, help="number of residual blocks in generator")
parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the noise input")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes in the dataset")
parser.add_argument("--sample_interval", type=int, default=300, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

# Calculate output of image discriminator (PatchGAN)
patch = int(opt.img_size / 2 ** 4)
patch = (1, patch, patch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') != (- 1)):
        init.gauss_(m.weight, mean=0.0, std=0.02)
    elif (classname.find('BatchNorm') != (- 1)):
        init.gauss_(m.weight, mean=1.0, std=0.02)
        init.constant_(m.bias, value=0.0)

class ResidualBlock(nn.Module):

    def __init__(self, in_features=64, out_features=64):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv(in_features, in_features, 3, stride=1, padding=1), nn.BatchNorm(in_features), nn.ReLU(), nn.Conv(in_features, in_features, 3, stride=1, padding=1), nn.BatchNorm(in_features))

    def execute(self, x):
        return (x + self.block(x))

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(opt.latent_dim, (opt.channels * (opt.img_size ** 2)))
        self.l1 = nn.Sequential(nn.Conv((opt.channels * 2), 64, 3, stride=1, padding=1), nn.ReLU())
        resblocks = []
        for _ in range(opt.n_residual_blocks):
            resblocks.append(ResidualBlock())
        self.resblocks = nn.Sequential(*resblocks)
        self.l2 = nn.Sequential(nn.Conv(64, opt.channels, 3, stride=1, padding=1), nn.Tanh())

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, img, z):
        gen_input = jt.contrib.concat((img, self.fc(z).view(*img.shape)), dim=1)
        out = self.l1(gen_input)
        out = self.resblocks(out)
        img_ = self.l2(out)
        return img_

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        def block(in_features, out_features, normalization=True):
            'Discriminator block'
            layers = [nn.Conv(in_features, out_features, 3, stride=2, padding=1), nn.LeakyReLU(scale=0.2)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_features, affine=None))
            return layers
        self.model = nn.Sequential(*block(opt.channels, 64, normalization=False), *block(64, 128), *block(128, 256), *block(256, 512), nn.Conv(512, 1, 3, stride=1, padding=1))

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, img):
        validity = self.model(img)
        return validity

class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

        def block(in_features, out_features, normalization=True):
            'Classifier block'
            layers = [nn.Conv(in_features, out_features, 3, stride=2, padding=1), nn.LeakyReLU(scale=0.2)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_features, affine=None))
            return layers
        self.model = nn.Sequential(*block(opt.channels, 64, normalization=False), *block(64, 128), *block(128, 256), *block(256, 512))
        input_size = (opt.img_size // (2 ** 4))
        self.output_layer = nn.Sequential(nn.Linear((512 * (input_size ** 2)), opt.n_classes), nn.Softmax())

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, img):
        feature_repr = self.model(img)
        feature_repr = feature_repr.view((feature_repr.shape[0], (- 1)))
        label = self.output_layer(feature_repr)
        return label

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

# Loss function
adversarial_loss = nn.MSELoss()
task_loss = nn.CrossEntropyLoss()

# Loss weights
lambda_adv = 1
lambda_task = 0.1

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
classifier = Classifier()

# Configure data loader
transform = transform.Compose([
    transform.Resize(opt.img_size),
    transform.ImageNormalize(mean=[0.5], std=[0.5]),
])
dataloader_A = MNIST(train=True, transform=transform).set_attrs(batch_size=opt.batch_size, shuffle=True)

dataloader_B = mnistm.MNISTM(mnist_root = "../../data/mnistm", train=True, transform = transform).set_attrs(batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = nn.Adam(
    generator.parameters() + classifier.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D = nn.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# ----------
#  Training
# ----------

# Keeps 100 accuracy measurements
task_performance = []
target_performance = []

for epoch in range(opt.n_epochs):
    for i, ((imgs_A, labels_A), (imgs_B, labels_B)) in enumerate(zip(dataloader_A, dataloader_B)):
        jt.sync_all(True)
        batch_size = imgs_A.size(0)
        # imgs_B = jt.array(imgs_B.numpy()).float32()
        # labels_B = jt.array(labels_B.numpy()).float32()

        # Adversarial ground truths
        valid = jt.ones([batch_size, *patch]).float32().stop_grad()
        fake = jt.zeros([batch_size, *patch]).float32().stop_grad()

        # -----------------
        #  Train Generator
        # -----------------

        # Sample noise
        z = jt.array(np.random.uniform(-1, 1, (batch_size, opt.latent_dim))).float32()

        # Generate a batch of images
        fake_B = generator(imgs_A, z)

        # Perform task on translated source image
        label_pred = classifier(fake_B)

        # Calculate the task loss
        task_loss_ = (task_loss(label_pred, labels_A) + task_loss(classifier(imgs_A), labels_A)) / 2

        # Loss measures generator's ability to fool the discriminator
        g_loss = lambda_adv * adversarial_loss(discriminator(fake_B), valid) + lambda_task * task_loss_

        optimizer_G.step(g_loss)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(imgs_B), valid)
        fake_loss = adversarial_loss(discriminator(fake_B.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        optimizer_D.step(d_loss)

        # ---------------------------------------
        #  Evaluate Performance on target domain
        # ---------------------------------------

        # Evaluate performance on translated Domain A
        acc = np.mean(np.argmax(label_pred.numpy(), axis=1) == labels_A.numpy())
        task_performance.append(acc)
        if len(task_performance) > 100:
            task_performance.pop(0)

        # Evaluate performance on Domain B
        pred_B = classifier(imgs_B)
        target_acc = np.mean(np.argmax(pred_B.numpy(), axis=1) == labels_B.numpy())
        target_performance.append(target_acc)
        if len(target_performance) > 100:
            target_performance.pop(0)

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [CLF acc: %3d%% (%3d%%), target_acc: %3d%% (%3d%%)]"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader_A),
                d_loss.data[0],
                g_loss.data[0],
                100 * acc,
                100 * np.mean(task_performance),
                100 * target_acc,
                100 * np.mean(target_performance),
            )
        )

        batches_done = len(dataloader_A) * epoch + i
        if batches_done % opt.sample_interval == 0:
            sample = jt.contrib.concat((imgs_A[:5], fake_B[:5], imgs_B[:5]), 2)
            save_image(sample.numpy(), "images/%d.png" % batches_done, nrow=5)