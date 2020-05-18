
import jittor as jt
from jittor import init
from jittor import nn

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        jt.init.gauss_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        jt.init.gauss_(m.weight, 1.0, 0.02)
        jt.init.constant_(m.bias, 0.0)

class MSELoss(nn.Module):
    def __init__(self):
        pass

    def execute(self, output, target):
        return (output-target).sqr().mean()

class L1Loss(nn.Module):
    def __init__(self):
        pass
    def execute(self, output, target):
        return (output-target).abs().mean()
        
class Upsample(nn.Module):
    def __init__(self, scale_factor=None, mode='nearest'):
        self.scale_factor = scale_factor if isinstance(scale_factor, tuple) else (scale_factor, scale_factor)
        self.mode = mode
    
    def execute(self, x):
        return nn.resize(x, size=(x.shape[2]*self.scale_factor[0], x.shape[3]*self.scale_factor[1]), mode=self.mode)

class InstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=None, is_train=True, sync=True):
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
            xmean = jt.mean(x, dims=[2,3], keepdims=1)
            x2mean = jt.mean(x*x, dims=[2,3], keepdims=1)
            if self.sync and jt.mpi:
                xmean = xmean.mpi_all_reduce("mean")
                x2mean = x2mean.mpi_all_reduce("mean")

            xvar = x2mean-xmean*xmean
            norm_x = (x-xmean)/jt.sqrt(xvar+self.eps)
            self.running_mean += (xmean.sum([0,2,3])-self.running_mean)*self.momentum
            self.running_var += (xvar.sum([0,2,3])-self.running_var)*self.momentum
        else:
            running_mean = self.running_mean.broadcast(x, [0,2,3])
            running_var = self.running_var.broadcast(x, [0,2,3])
            norm_x = (x-running_mean)/jt.sqrt(running_var+self.eps)
        w = self.weight.broadcast(x, [0,2,3])
        b = self.bias.broadcast(x, [0,2,3])
        return norm_x * w + b

class UNetDown(nn.Module):

    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv(in_size, out_size, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(InstanceNorm2d(out_size, affine=None))
        layers.append(nn.LeakyReLU(scale=0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
        for m in self.model:
            weights_init_normal(m)

    def execute(self, x):
        return self.model(x)

class UNetUp(nn.Module):

    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [nn.ConvTranspose(in_size, out_size, 4, stride=2, padding=1, bias=False), InstanceNorm2d(out_size, affine=None), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
        for m in self.model:
            weights_init_normal(m)

    def execute(self, x, skip_input):
        x = self.model(x)
        x = jt.contrib.concat((x, skip_input), dim=1)
        return x

class GeneratorUNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)
        self.final = nn.Sequential(Upsample(scale_factor=2), nn.ZeroPad2d((1, 0, 1, 0)), nn.Conv(128, out_channels, 4, padding=1), nn.Tanh())
        for m in self.final:
            weights_init_normal(m)

    def execute(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)

class Discriminator(nn.Module):

    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            'Returns downsampling layers of each discriminator block'
            layers = [nn.Conv(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(InstanceNorm2d(out_filters, affine=None))
            layers.append(nn.LeakyReLU(scale=0.2))
            for m in layers:
                weights_init_normal(m)
            return layers
        self.model = nn.Sequential(*discriminator_block((in_channels * 2), 64, normalization=False), *discriminator_block(64, 128), *discriminator_block(128, 256), *discriminator_block(256, 512), nn.ZeroPad2d((1, 0, 1, 0)), nn.Conv(512, 1, 4, padding=1, bias=False))
        weights_init_normal(self.model[-1])

    def execute(self, img_A, img_B):
        img_input = jt.contrib.concat((img_A, img_B), dim=1)
        return self.model(img_input)
