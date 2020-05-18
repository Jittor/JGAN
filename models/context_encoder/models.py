
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

class Generator(nn.Module):

    def __init__(self, channels=3):
        super(Generator, self).__init__()

        def downsample(in_feat, out_feat, normalize=True):
            layers = [nn.Conv(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm(out_feat, eps=0.8))
            layers.append(nn.Leaky_relu(scale=0.2))
            return layers

        def upsample(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm(out_feat, eps=0.8))
            layers.append(nn.ReLU())
            return layers
        self.model = nn.Sequential(*downsample(channels, 64, normalize=False), *downsample(64, 64), *downsample(64, 128), *downsample(128, 256), *downsample(256, 512), nn.Conv(512, 4000, 1), *upsample(4000, 512), *upsample(512, 256), *upsample(256, 128), *upsample(128, 64), nn.Conv(64, channels, 3, stride=1, padding=1), nn.Tanh())
        for m in self.model:
            weights_init_normal(m)

    def execute(self, x):
        return self.model(x)

class Discriminator(nn.Module):

    def __init__(self, channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            'Returns layers of each discriminator block'
            layers = [nn.Conv(in_filters, out_filters, 3, stride=stride, padding=1)]
            if normalize:
                layers.append(InstanceNorm2d(out_filters))
            layers.append(nn.Leaky_relu(scale=0.2))
            return layers
        layers = []
        in_filters = channels
        for (out_filters, stride, normalize) in [(64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters
        layers.append(nn.Conv(out_filters, 1, 3, stride=1, padding=1))
        self.model = nn.Sequential(*layers)
        for m in self.model:
            weights_init_normal(m)

    def execute(self, img):
        return self.model(img)
