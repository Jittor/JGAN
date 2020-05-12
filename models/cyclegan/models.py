
import jittor as jt
from jittor import init
from jittor import nn

def weights_init_normal(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') != (- 1)):
        init.gauss_(m.weight, mean=0.0, std=0.02)
        if (hasattr(m, 'bias') and (m.bias is not None)):
            init.constant_(m.bias, value=0.0)
    elif (classname.find('BatchNorm') != (- 1)):
        init.gauss_(m.weight, mean=1.0, std=0.02)
        init.constant_(m.bias, value=0.0)

class ResidualBlock(nn.Module):

    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv(in_features, in_features, 3), nn.InstanceNorm2d(in_features, affine=None), nn.ReLU(), nn.ReflectionPad2d(1), nn.Conv(in_features, in_features, 3), nn.InstanceNorm2d(in_features, affine=None))

    def execute(self, x):
        return (x + self.block(x))

class GeneratorResNet(nn.Module):

    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()
        channels = input_shape[0]
        out_features = 64
        model = [nn.ReflectionPad2d(channels), nn.Conv(channels, out_features, 7), nn.InstanceNorm2d(out_features, affine=None), nn.ReLU()]
        in_features = out_features
        for _ in range(2):
            out_features *= 2
            model += [nn.Conv(in_features, out_features, 3, stride=2, padding=1), nn.InstanceNorm2d(out_features, affine=None), nn.ReLU()]
            in_features = out_features
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]
        for _ in range(2):
            out_features //= 2
            model += [nn.Upsample(scale_factor=2), nn.Conv(in_features, out_features, 3, stride=1, padding=1), nn.InstanceNorm2d(out_features, affine=None), nn.ReLU()]
            in_features = out_features
        model += [nn.ReflectionPad2d(channels), nn.Conv(out_features, channels, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, x):
        return self.model(x)

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
        self.model = nn.Sequential(*discriminator_block(channels, 64, normalize=False), *discriminator_block(64, 128), *discriminator_block(128, 256), *discriminator_block(256, 512), nn.ZeroPad2d((1, 0, 1, 0)), nn.Conv(512, 1, 4, padding=1))

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, img):
        return self.model(img)
