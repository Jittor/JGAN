import jittor as jt
from jittor import init
from jittor import nn
from jittor.models import vgg19
import math

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.gauss_(m.weight, 0.0, 0.02)
        init.gauss_(m.bias, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        init.gauss_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0.0)

class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19()
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])
        for m in self.modules():
            weights_init_normal(m)

    def execute(self, img):
        return self.feature_extractor(img)

class ResidualBlock(nn.Module):

    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(nn.Conv(in_features, in_features, 3, stride=1, padding=1), nn.BatchNorm(in_features, eps=0.8), nn.PReLU(), nn.Conv(in_features, in_features, 3, stride=1, padding=1), nn.BatchNorm(in_features, eps=0.8))

    def execute(self, x):
        return (x + self.conv_block(x))

class GeneratorResNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(GeneratorResNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv(in_channels, 64, 9, stride=1, padding=4), nn.PReLU())
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)
        self.conv2 = nn.Sequential(nn.Conv(64, 64, 3, stride=1, padding=1), nn.BatchNorm(64, eps=0.8))
        upsampling = []
        for out_features in range(2):
            upsampling += [nn.Conv(64, 256, 3, stride=1, padding=1), nn.BatchNorm(256), nn.PixelShuffle(upscale_factor=2), nn.PReLU()]
        self.upsampling = nn.Sequential(*upsampling)
        self.conv3 = nn.Sequential(nn.Conv(64, out_channels, 9, stride=1, padding=4), nn.Tanh())
        for m in self.modules():
            weights_init_normal(m)

    def execute(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = out1 + out2
        out = self.upsampling(out)
        out = self.conv3(out)
        return out

class Discriminator(nn.Module):

    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape
        (in_channels, in_height, in_width) = self.input_shape
        (patch_h, patch_w) = (int((in_height / (2 ** 4))), int((in_width / (2 ** 4))))
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv(in_filters, out_filters, 3, stride=1, padding=1))
            if (not first_block):
                layers.append(nn.BatchNorm(out_filters))
            layers.append(nn.LeakyReLU(scale=0.2))
            layers.append(nn.Conv(out_filters, out_filters, 3, stride=2, padding=1))
            layers.append(nn.BatchNorm(out_filters))
            layers.append(nn.LeakyReLU(scale=0.2))
            return layers
        layers = []
        in_filters = in_channels
        for (i, out_filters) in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters
        layers.append(nn.Conv(out_filters, 1, 3, stride=1, padding=1))
        self.model = nn.Sequential(*layers)
        for m in self.modules():
            weights_init_normal(m)

    def execute(self, img):
        return self.model(img)
