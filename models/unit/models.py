
import jittor as jt
from jittor import init
from jittor import nn
import numpy as np

def weights_init_normal(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') != (- 1)):
        init.gauss_(m.weight, mean=0.0, std=0.02)
    elif (classname.find('BatchNorm') != (- 1)):
        init.gauss_(m.weight, mean=1.0, std=0.02)
        init.constant_(m.bias, value=0.0)

class ResidualBlock(nn.Module):

    def __init__(self, features):
        super(ResidualBlock, self).__init__()
        conv_block = [nn.ReflectionPad2d(1), nn.Conv(features, features, 3), nn.InstanceNorm2d(features, affine=None), nn.ReLU(), nn.ReflectionPad2d(1), nn.Conv(features, features, 3), nn.InstanceNorm2d(features, affine=None)]
        self.conv_block = nn.Sequential(*conv_block)

    def execute(self, x):
        return (x + self.conv_block(x))

class Encoder(nn.Module):

    def __init__(self, in_channels=3, dim=64, n_downsample=2, shared_block=None):
        super(Encoder, self).__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv(in_channels, dim, 7), nn.InstanceNorm2d(64, affine=None), nn.LeakyReLU(scale=0.2)]
        for _ in range(n_downsample):
            layers += [nn.Conv(dim, (dim * 2), 4, stride=2, padding=1), nn.InstanceNorm2d((dim * 2), affine=None), nn.ReLU()]
            dim *= 2
        for _ in range(3):
            layers += [ResidualBlock(dim)]
        self.model_blocks = nn.Sequential(*layers)
        self.shared_block = shared_block

        for m in self.modules():
            weights_init_normal(m)

    def reparameterization(self, mu):
        z = jt.array(np.random.normal(0, 1, mu.shape)).float32()
        return (z + mu)

    def execute(self, x):
        x = self.model_blocks(x)
        mu = self.shared_block(x)
        z = self.reparameterization(mu)
        return (mu, z)

class Generator(nn.Module):

    def __init__(self, out_channels=3, dim=64, n_upsample=2, shared_block=None):
        super(Generator, self).__init__()
        self.shared_block = shared_block
        layers = []
        dim = (dim * (2 ** n_upsample))
        for _ in range(3):
            layers += [ResidualBlock(dim)]
        for _ in range(n_upsample):
            layers += [nn.ConvTranspose(dim, (dim // 2), 4, stride=2, padding=1), nn.InstanceNorm2d((dim // 2), affine=None), nn.LeakyReLU(scale=0.2)]
            dim = (dim // 2)
        layers += [nn.ReflectionPad2d(3), nn.Conv(dim, out_channels, 7), nn.Tanh()]
        self.model_blocks = nn.Sequential(*layers)

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, x):
        x = self.shared_block(x)
        x = self.model_blocks(x)
        return x

class Discriminator(nn.Module):

    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        (channels, height, width) = input_shape
        self.output_shape = (1, (height // (2 ** 4)), (width // (2 ** 4)))

        def discriminator_block(in_filters, out_filters, normalize=True):
            'Returns downsampling layers of each discriminator block'
            layers = [nn.Conv(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters, affine=None))
            layers.append(nn.LeakyReLU(scale=0.2))
            return layers
        self.model = nn.Sequential(*discriminator_block(channels, 64, normalize=False), *discriminator_block(64, 128), *discriminator_block(128, 256), *discriminator_block(256, 512), nn.Conv(512, 1, 3, padding=1))

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, img):
        return self.model(img)