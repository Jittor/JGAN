
import jittor as jt
from jittor import init
from jittor import nn

def weights_init_normal(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') != (- 1)):
        init.gauss_(m.weight, mean=0.0, std=0.02)

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
        
class ResidualBlock(nn.Module):

    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [nn.Conv(in_features, in_features, 3, stride=1, padding=1, bias=False), InstanceNorm2d(in_features, affine=None), nn.ReLU(), nn.Conv(in_features, in_features, 3, stride=1, padding=1, bias=False), InstanceNorm2d(in_features, affine=None)]
        self.conv_block = nn.Sequential(*conv_block)
        for m in self.conv_block:
            weights_init_normal(m)

    def execute(self, x):
        return (x + self.conv_block(x))

class GeneratorResNet(nn.Module):

    def __init__(self, img_shape=(3, 128, 128), res_blocks=9, c_dim=5):
        super(GeneratorResNet, self).__init__()
        (channels, img_size, _) = img_shape
        model = [nn.Conv((channels + c_dim), 64, 7, stride=1, padding=3, bias=False), InstanceNorm2d(64, affine=None), nn.ReLU()]
        curr_dim = 64
        for _ in range(2):
            model += [nn.Conv(curr_dim, (curr_dim * 2), 4, stride=2, padding=1, bias=False), InstanceNorm2d((curr_dim * 2), affine=None), nn.ReLU()]
            curr_dim *= 2
        for _ in range(res_blocks):
            model += [ResidualBlock(curr_dim)]
        for _ in range(2):
            model += [nn.ConvTranspose(curr_dim, (curr_dim // 2), 4, stride=2, padding=1, bias=False), InstanceNorm2d((curr_dim // 2), affine=None), nn.ReLU()]
            curr_dim = (curr_dim // 2)
        model += [nn.Conv(curr_dim, channels, 7, stride=1, padding=3), nn.Tanh()]
        self.model = nn.Sequential(*model)
        for m in self.model:
            weights_init_normal(m)

    def execute(self, x, c):
        c = c.view((c.shape[0], c.shape[1], 1, 1))
        c = c.broadcast([c.shape[0], c.shape[1], x.shape[2], x.shape[3]])
        x = jt.contrib.concat((x, c), dim=1)
        return self.model(x)

class Discriminator(nn.Module):

    def __init__(self, img_shape=(3, 128, 128), c_dim=5, n_strided=6):
        super(Discriminator, self).__init__()
        (channels, img_size, _) = img_shape

        def discriminator_block(in_filters, out_filters):
            'Returns downsampling layers of each discriminator block'
            layers = [nn.Conv(in_filters, out_filters, 4, stride=2, padding=1), nn.LeakyReLU(scale=0.01)]
            for m in layers:
                weights_init_normal(m)
            return layers
        layers = discriminator_block(channels, 64)
        curr_dim = 64
        for _ in range((n_strided - 1)):
            layers.extend(discriminator_block(curr_dim, (curr_dim * 2)))
            curr_dim *= 2
        self.model = nn.Sequential(*layers)
        self.out1 = nn.Conv(curr_dim, 1, 3, padding=1, bias=False)
        kernel_size = (img_size // (2 ** n_strided))
        self.out2 = nn.Conv(curr_dim, c_dim, kernel_size, bias=False)
        weights_init_normal(self.out1)
        weights_init_normal(self.out2)

    def execute(self, img):
        feature_repr = self.model(img)
        out_adv = self.out1(feature_repr)
        out_cls = self.out2(feature_repr)
        return (out_adv, out_cls.view((out_cls.shape[0], (- 1))))
