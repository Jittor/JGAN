
import jittor as jt
from jittor import init
from jittor import nn
from jittor.dataset.mnist import MNIST
import jittor.transform as transform
import cv2
import argparse
import os, sys
import numpy as np
from itertools import chain as ichain
import time

jt.flags.use_cuda = 1
os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser(description='ClusterGAN Training Script')
parser.add_argument('-n', '--n_epochs', dest='n_epochs', default=200, type=int, help='Number of epochs')
parser.add_argument('-b', '--batch_size', dest='batch_size', default=32, type=int, help='Batch size')
parser.add_argument('-i', '--img_size', dest='img_size', type=int, default=28, help='Size of image dimension')
parser.add_argument('-d', '--latent_dim', dest='latent_dim', default=30, type=int, help='Dimension of latent space')
parser.add_argument('-l', '--lr', dest='learning_rate', type=float, default=0.0001, help='Learning rate')
parser.add_argument('-c', '--n_critic', dest='n_critic', type=int, default=5, help='Number of training steps for discriminator per iter')
parser.add_argument('-w', '--wass_flag', dest='wass_flag', action='store_true', help='Flag for Wasserstein metric')
args = parser.parse_args()

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

class BCELoss(nn.Module):
    def __init__(self):
        pass
    def execute(self, output, target):
        return - (target * jt.log(jt.maximum(output, 1e-20)) + (1 - target) * jt.log(jt.maximum(1 - output, 1e-20))).mean()

class MSELoss(nn.Module):
    def __init__(self):
        pass
    def execute(self, output, target):
        return (output-target).sqr().mean()

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        pass
    def execute(self, output, target):
        return nn.cross_entropy_loss(output, target)

def sample_z(shape=64, latent_dim=10, n_c=10, fix_class=(- 1)):
    assert ((fix_class == (- 1)) or ((fix_class >= 0) and (fix_class < n_c))), ('Requested class %i outside bounds.' % fix_class)
    zn = jt.array(0.75 * np.random.normal(0, 1, (shape, latent_dim)).astype(np.float32)).stop_grad()
    zc_FT = np.zeros([shape, n_c])
    zc_idx = np.zeros(n_c)
    if (fix_class == (- 1)):
        zc_idx = np.random.randint(n_c, size=shape)
        zc_FT[range(shape),zc_idx]=1
    else:
        zc_idx[:] = fix_class
        zc_FT[range(shape),fix_class]=1
    zc = jt.array(zc_FT.astype(np.float32)).stop_grad()
    zc_idx = jt.array(zc_idx.astype(np.float32)).stop_grad()
    return (zn, zc, zc_idx)

def calc_gradient_penalty(netD, real_data, generated_data):
    LAMBDA = 10
    b_size = real_data.shape[0]
    alpha = jt.random([b_size, 1, 1, 1])
    alpha = alpha.broadcast(real_data)
    interpolated = ((alpha * real_data.data) + ((1 - alpha) * generated_data.data))
    prob_interpolated = netD(interpolated)
    gradients = jt.grad(prob_interpolated, interpolated)
    gradients = jt.reshape(gradients, [b_size, -1])
    gradients_norm = jt.sqrt((jt.sum((gradients ** 2), dim=1) + 1e-12))
    return (LAMBDA * ((gradients_norm - 1) ** 2).mean())

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv):
            print("Conv")
            init.gauss_(m.weight, 0, 0.02)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose):
            print("ConvTranspose")
            init.gauss_(m.weight, 0, 0.02)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            print("Linear")
            init.gauss_(m.weight, 0, 0.02)
            init.constant_(m.bias, 0)

def softmax(x):
    return nn.softmax(x, dim=1)

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

class Reshape(nn.Module):
    '\n    Class for performing a reshape as a layer in a sequential model.\n    '

    def __init__(self, shape=[]):
        super(Reshape, self).__init__()
        self.shape = shape

    def execute(self, x):
        return jt.reshape(x, [x.shape[0], *self.shape])

    def extra_repr(self):
        return 'shape={}'.format(self.shape)

class Generator_CNN(nn.Module):
    '\n    CNN to model the generator of a ClusterGAN\n    Input is a vector from representation space of dimension z_dim\n    output is a vector from image space of dimension X_dim\n    '

    def __init__(self, latent_dim, n_c, x_shape, verbose=False):
        super(Generator_CNN, self).__init__()
        self.name = 'generator'
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.x_shape = x_shape
        self.ishape = (128, 7, 7)
        self.iels = int(np.prod(self.ishape))
        self.verbose = verbose
        self.model0 = nn.Sequential(nn.Linear((self.latent_dim + self.n_c), 1024))
        self.model1 = nn.Sequential(BatchNorm1d(1024), nn.Leaky_relu(0.2))
        self.model2 = nn.Sequential(nn.Linear(1024, self.iels), BatchNorm1d(self.iels), nn.Leaky_relu(0.2))
        self.model3 = nn.Sequential(Reshape(self.ishape), nn.ConvTranspose(128, 64, 4, stride=2, padding=1, bias=True), nn.BatchNorm(64), nn.Leaky_relu(0.2))
        self.model4 = nn.Sequential(nn.ConvTranspose(64, 1, 4, stride=2, padding=1, bias=True))
        self.sigmoid = nn.Sigmoid()
        initialize_weights(self)
        if self.verbose:
            print('Setting up {}...\n'.format(self.name))
            print(self.model)

    def execute(self, zn, zc):
        z = jt.contrib.concat([zn, zc], dim=1)
        x_gen = self.model0(z)
        x_gen = self.model1(x_gen)
        x_gen = self.model2(x_gen)
        x_gen = self.model3(x_gen)
        x_gen = self.model4(x_gen)
        x_gen = self.sigmoid(x_gen)
        x_gen = jt.reshape(x_gen, [x_gen.shape[0], *self.x_shape])
        return x_gen

class Encoder_CNN(nn.Module):
    '\n    CNN to model the encoder of a ClusterGAN\n    Input is vector X from image space if dimension X_dim\n    Output is vector z from representation space of dimension z_dim\n    '

    def __init__(self, latent_dim, n_c, verbose=False):
        super(Encoder_CNN, self).__init__()
        self.name = 'encoder'
        self.channels = 1
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.cshape = (128, 5, 5)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.verbose = verbose
        self.model = nn.Sequential(nn.Conv(self.channels, 64, 4, stride=2, bias=True), nn.Leaky_relu(0.2), nn.Conv(64, 128, 4, stride=2, bias=True), nn.Leaky_relu(0.2), Reshape(self.lshape), nn.Linear(self.iels, 1024), nn.Leaky_relu(0.2), nn.Linear(1024, (latent_dim + n_c)))
        initialize_weights(self)
        if self.verbose:
            print('Setting up {}...\n'.format(self.name))
            print(self.model)

    def execute(self, in_feat):
        z_img = self.model(in_feat)
        z = jt.reshape(z_img, [z_img.shape[0], (- 1)])
        zn = z[:, 0:self.latent_dim]
        zc_logits = z[:, self.latent_dim:]
        zc = softmax(zc_logits)
        return (zn, zc, zc_logits)

class Discriminator_CNN(nn.Module):
    '\n    CNN to model the discriminator of a ClusterGAN\n    Input is tuple (X,z) of an image vector and its corresponding\n    representation z vector. For example, if X comes from the dataset, corresponding\n    z is Encoder(X), and if z is sampled from representation space, X is Generator(z)\n    Output is a 1-dimensional value\n    '

    def __init__(self, wass_metric=False, verbose=False):
        super(Discriminator_CNN, self).__init__()
        self.name = 'discriminator'
        self.channels = 1
        self.cshape = (128, 5, 5)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.wass = wass_metric
        self.verbose = verbose
        self.model = nn.Sequential(nn.Conv(self.channels, 64, 4, stride=2, bias=True), nn.Leaky_relu(0.2), nn.Conv(64, 128, 4, stride=2, bias=True), nn.Leaky_relu(0.2), Reshape(self.lshape), nn.Linear(self.iels, 1024), nn.Leaky_relu(0.2), nn.Linear(1024, 1))
        if (not self.wass):
            self.model = nn.Sequential(self.model, nn.Sigmoid())
        initialize_weights(self)
        if self.verbose:
            print('Setting up {}...\n'.format(self.name))
            print(self.model)

    def execute(self, img):
        validity = self.model(img)
        return validity

n_epochs = args.n_epochs
batch_size = args.batch_size
test_batch_size = 5000
lr = args.learning_rate
b1 = 0.5
b2 = 0.9
decay = (2.5 * 1e-05)
n_skip_iter = args.n_critic
img_size = args.img_size
channels = 1
latent_dim = args.latent_dim
n_c = 10
betan = 10
betac = 10
wass_metric = args.wass_flag
print(wass_metric)
x_shape = (channels, img_size, img_size)

bce_loss = BCELoss()
xe_loss = CrossEntropyLoss()
mse_loss = MSELoss()

# Initialize generator and discriminator
generator = Generator_CNN(latent_dim, n_c, x_shape)
encoder = Encoder_CNN(latent_dim, n_c)
discriminator = Discriminator_CNN(wass_metric=wass_metric)

# Configure data loader
transform = transform.Compose([
    transform.Resize(size=img_size),
    transform.Gray(),
])
dataloader = MNIST(train=True, transform=transform).set_attrs(batch_size=batch_size, shuffle=True)
testdata = MNIST(train=False, transform=transform).set_attrs(batch_size=batch_size, shuffle=True)
(test_imgs, test_labels) = next(iter(testdata))

ge_chain = generator.parameters()
for p in encoder.parameters():
    ge_chain.append(p)
#TODO: weight_decay=decay
optimizer_GE = jt.nn.Adam(ge_chain, lr=lr, betas=(b1, b2), weight_decay=0.0)
optimizer_D = jt.nn.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

ge_l = []
d_l = []
c_zn = []
c_zc = []
c_i = []

warmup_times = 300
run_times = 3000
total_time = 0.
cnt = 0

print(('\nBegin training session with %i epochs...\n' % n_epochs))

# ----------
#  Training
# ----------

for epoch in range(n_epochs):
    for (i, (real_imgs, itruth_label)) in enumerate(dataloader):
        generator.train()
        encoder.train()
        (zn, zc, zc_idx) = sample_z(shape=real_imgs.shape[0], latent_dim=latent_dim, n_c=n_c)
        gen_imgs = generator(zn, zc)
        D_gen = discriminator(gen_imgs)
        D_real = discriminator(real_imgs)
        
        # -----------------
        #  Train Generator
        # -----------------

        if ((i % n_skip_iter) == 0):
            (enc_gen_zn, enc_gen_zc, enc_gen_zc_logits) = encoder(gen_imgs)
            zn_loss = mse_loss(enc_gen_zn, zn)
            zc_loss = xe_loss(enc_gen_zc_logits, zc_idx)
            if wass_metric:
                ge_loss = ((jt.mean(D_gen) + (betan * zn_loss)) + (betac * zc_loss))
            else:
                valid = jt.ones([gen_imgs.shape[0], 1]).stop_grad()
                v_loss = bce_loss(D_gen, valid)
                ge_loss = ((v_loss + (betan * zn_loss)) + (betac * zc_loss))
            optimizer_GE.step(ge_loss)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        if wass_metric:
            grad_penalty = calc_gradient_penalty(discriminator, real_imgs, gen_imgs)
            d_loss = ((jt.mean(D_real) - jt.mean(D_gen)) + grad_penalty)
        else:
            fake = jt.zeros([gen_imgs.shape[0], 1]).stop_grad()
            real_loss = bce_loss(D_real, valid)
            fake_loss = bce_loss(D_gen, fake)
            d_loss = ((real_loss + fake_loss) / 2)
        optimizer_D.step(d_loss)
        
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
        d_l.append(d_loss.numpy()[0])
        ge_l.append(ge_loss.numpy()[0])
        generator.eval()
        encoder.eval()
        n_sqrt_samp = 5
        n_samp = (n_sqrt_samp * n_sqrt_samp)
        (t_imgs, t_label) = (test_imgs, test_labels)
        (e_tzn, e_tzc, e_tzc_logits) = encoder(t_imgs)
        teg_imgs = generator(e_tzn, e_tzc)
        img_mse_loss = mse_loss(t_imgs, teg_imgs)
        c_i.append(img_mse_loss.numpy()[0])
        (zn_samp, zc_samp, zc_samp_idx) = sample_z(shape=n_samp, latent_dim=latent_dim, n_c=n_c)
        gen_imgs_samp = generator(zn_samp, zc_samp)
        (zn_e, zc_e, zc_e_logits) = encoder(gen_imgs_samp)
        lat_mse_loss = mse_loss(zn_e, zn_samp)
        lat_xe_loss = xe_loss(zc_e_logits, zc_samp_idx)
        c_zn.append(lat_mse_loss.numpy()[0])
        c_zc.append(lat_xe_loss.numpy()[0])
        (r_imgs, i_label) = (real_imgs[:n_samp], itruth_label[:n_samp])
        (e_zn, e_zc, e_zc_logits) = encoder(r_imgs)
        reg_imgs = generator(e_zn, e_zc)
        save_image(reg_imgs.data[:n_samp], ('images/cycle_reg_%06i.png' % epoch), nrow=n_sqrt_samp)
        save_image(gen_imgs_samp.data[:n_samp], ('images/gen_%06i.png' % epoch), nrow=n_sqrt_samp)
        stack_imgs = None
        for idx in range(n_c):
            (zn_samp, zc_samp, zc_samp_idx) = sample_z(shape=n_c, latent_dim=latent_dim, n_c=n_c, fix_class=idx)
            gen_imgs_samp = generator(zn_samp, zc_samp)
            if (idx == 0):
                stack_imgs = gen_imgs_samp
            else:
                stack_imgs = jt.contrib.concat([stack_imgs, gen_imgs_samp], dim=0)
        save_image(stack_imgs.numpy(), ('images/gen_classes_%06i.png' % epoch), nrow=n_c)
        print(('[Epoch %d/%d] \n\tModel Losses: [D: %f] [GE: %f]' % (epoch, n_epochs, d_loss.numpy()[0], ge_loss.numpy()[0])))
        print(('\tCycle Losses: [x: %f] [z_n: %f] [z_c: %f]' % (img_mse_loss.numpy()[0], lat_mse_loss.numpy()[0], lat_xe_loss.numpy()[0])))
