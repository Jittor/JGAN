import jittor as jt
from jittor import init
from jittor import nn

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.gauss_(m.weight, 0.0, 0.02)
        init.gauss_(m.bias, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        init.gauss_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0.0)

class UNetDown(nn.Module):

    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv(in_size, out_size, 4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def execute(self, x):
        return self.model(x)

class UNetUp(nn.Module):

    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [nn.ConvTranspose(in_size, out_size, 4, stride=2, padding=1), nn.InstanceNorm2d(out_size), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def execute(self, x, skip_input):
        x = self.model(x)
        x = jt.contrib.concat((x, skip_input), dim=1)
        return x

class GeneratorUNet(nn.Module):

    def __init__(self, input_shape):
        super(GeneratorUNet, self).__init__()
        (channels, _, _) = input_shape
        self.down1 = UNetDown(channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256, dropout=0.5)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5, normalize=False)
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 256, dropout=0.5)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256, 64)
        self.final = nn.Sequential(nn.Upsample(scale_factor=2), nn.ZeroPad2d((1, 0, 1, 0)), nn.Conv(128, channels, 4, padding=1), nn.Tanh())

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        return self.final(u5)

class Discriminator(nn.Module):

    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        (channels, height, width) = input_shape
        self.output_shape = (1, (height // (2 ** 3)), (width // (2 ** 3)))

        def discriminator_block(in_filters, out_filters, normalization=True):
            'Returns downsampling layers of each discriminator block'
            layers = [nn.Conv(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2))
            return layers
        self.model = nn.Sequential(*discriminator_block(channels, 64, normalization=False), *discriminator_block(64, 128), *discriminator_block(128, 256), nn.ZeroPad2d((1, 0, 1, 0)), nn.Conv(256, 1, 4, padding=1))

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, img):
        return self.model(img)
