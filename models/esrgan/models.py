
import jittor as jt
from jittor import init
from jittor import nn
from jittor.models import vgg19
import math

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19()
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def execute(self, img):
        return self.vgg19_54(img)

class DenseResidualBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv(in_features, filters, 3, stride=1, padding=1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)
        self.b1 = block(in_features=(1 * filters))
        self.b2 = block(in_features=(2 * filters))
        self.b3 = block(in_features=(3 * filters))
        self.b4 = block(in_features=(4 * filters))
        self.b5 = block(in_features=(5 * filters), non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def execute(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = jt.contrib.concat([inputs, out], dim=1)
        return (out * self.res_scale + x)

class ResidualInResidualDenseBlock(nn.Module):

    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters))

    def execute(self, x):
        return (self.dense_blocks(x) * self.res_scale + x)

class GeneratorRRDB(nn.Module):

    def __init__(self, channels, filters=64, num_res_blocks=16, num_upsample=2):
        super(GeneratorRRDB, self).__init__()
        self.conv1 = nn.Conv(channels, filters, 3, stride=1, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        self.conv2 = nn.Conv(filters, filters, 3, stride=1, padding=1)
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [nn.Conv(filters, (filters * 4), 3, stride=1, padding=1), nn.LeakyReLU(), nn.PixelShuffle(upscale_factor=2)]
        self.upsampling = nn.Sequential(*upsample_layers)
        self.conv3 = nn.Sequential(nn.Conv(filters, filters, 3, stride=1, padding=1), nn.LeakyReLU(), nn.Conv(filters, channels, 3, stride=1, padding=1))

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

    def execute(self, img):
        return self.model(img)
