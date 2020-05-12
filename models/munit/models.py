
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

class LambdaLR():

    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), 'Decay must start before the training session ends!'
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return (1.0 - (max(0, ((epoch + self.offset) - self.decay_start_epoch)) / (self.n_epochs - self.decay_start_epoch)))

class Encoder(nn.Module):

    def __init__(self, in_channels=3, dim=64, n_residual=3, n_downsample=2, style_dim=8):
        super(Encoder, self).__init__()
        self.content_encoder = ContentEncoder(in_channels, dim, n_residual, n_downsample)
        self.style_encoder = StyleEncoder(in_channels, dim, n_downsample, style_dim)

    def execute(self, x):
        content_code = self.content_encoder(x)
        style_code = self.style_encoder(x)
        return (content_code, style_code)

class Decoder(nn.Module):

    def __init__(self, out_channels=3, dim=64, n_residual=3, n_upsample=2, style_dim=8):
        super(Decoder, self).__init__()
        layers = []
        dim = (dim * (2 ** n_upsample))
        for _ in range(n_residual):
            layers += [ResidualBlock(dim, norm='adain')]
        for _ in range(n_upsample):
            layers += [nn.Upsample(scale_factor=2), nn.Conv(dim, (dim // 2), 5, stride=1, padding=2), LayerNorm((dim // 2)), nn.ReLU()]
            dim = (dim // 2)
        layers += [nn.ReflectionPad2d(3), nn.Conv(dim, out_channels, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)
        num_adain_params = self.get_num_adain_params()
        self.mlp = MLP(style_dim, num_adain_params)

    def get_num_adain_params(self):
        'Return the number of AdaIN parameters needed by the model'
        num_adain_params = 0
        for m in self.modules():
            if (m.__class__.__name__ == 'AdaptiveInstanceNorm2d'):
                num_adain_params += (2 * m.num_features)
        return num_adain_params

    def assign_adain_params(self, adain_params):
        'Assign the adain_params to the AdaIN layers in model'
        for m in self.modules():
            if (m.__class__.__name__ == 'AdaptiveInstanceNorm2d'):
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:(2 * m.num_features)]
                m.bias = mean.contiguous().view((- 1))
                m.weight = std.contiguous().view((- 1))
                if (adain_params.shape[1] > (2 * m.num_features)):
                    adain_params = adain_params[:, (2 * m.num_features):]

    def execute(self, content_code, style_code):
        self.assign_adain_params(self.mlp(style_code))
        img = self.model(content_code)
        return img

class ContentEncoder(nn.Module):

    def __init__(self, in_channels=3, dim=64, n_residual=3, n_downsample=2):
        super(ContentEncoder, self).__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv(in_channels, dim, 7), nn.InstanceNorm2d(dim, affine=None), nn.ReLU()]
        for _ in range(n_downsample):
            layers += [nn.Conv(dim, (dim * 2), 4, stride=2, padding=1), nn.InstanceNorm2d((dim * 2), affine=None), nn.ReLU()]
            dim *= 2
        for _ in range(n_residual):
            layers += [ResidualBlock(dim, norm='in')]
        self.model = nn.Sequential(*layers)

    def execute(self, x):
        return self.model(x)

class StyleEncoder(nn.Module):

    def __init__(self, in_channels=3, dim=64, n_downsample=2, style_dim=8):
        super(StyleEncoder, self).__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv(in_channels, dim, 7), nn.ReLU()]
        for _ in range(2):
            layers += [nn.Conv(dim, (dim * 2), 4, stride=2, padding=1), nn.ReLU()]
            dim *= 2
        for _ in range((n_downsample - 2)):
            layers += [nn.Conv(dim, dim, 4, stride=2, padding=1), nn.ReLU()]
        layers += [nn.AdaptiveAvgPool2d(1), nn.Conv(dim, style_dim, 1, stride=1, padding=0)]
        self.model = nn.Sequential(*layers)

    def execute(self, x):
        return self.model(x)

class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, dim=256, n_blk=3, activ='relu'):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, dim), nn.ReLU()]
        for _ in range((n_blk - 2)):
            layers += [nn.Linear(dim, dim), nn.ReLU()]
        layers += [nn.Linear(dim, output_dim)]
        self.model = nn.Sequential(*layers)

    def execute(self, x):
        return self.model(x.view((x.shape[0], (- 1))))

class MultiDiscriminator(nn.Module):

    def __init__(self, in_channels=3):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            'Returns downsampling layers of each discriminator block'
            layers = [nn.Conv(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters, affine=None))
            layers.append(nn.LeakyReLU(scale=0.2))
            return layers
        self.disc_0 = nn.Sequential(*discriminator_block(in_channels, 64, normalize=False), *discriminator_block(64, 128), *discriminator_block(128, 256), *discriminator_block(256, 512), nn.Conv(512, 1, 3, padding=1))
        self.disc_1 = nn.Sequential(*discriminator_block(in_channels, 64, normalize=False), *discriminator_block(64, 128), *discriminator_block(128, 256), *discriminator_block(256, 512), nn.Conv(512, 1, 3, padding=1))
        self.disc_2 = nn.Sequential(*discriminator_block(in_channels, 64, normalize=False), *discriminator_block(64, 128), *discriminator_block(128, 256), *discriminator_block(256, 512), nn.Conv(512, 1, 3, padding=1))

    def compute_loss(self, x, gt):
        'Computes the MSE between model output and scalar gt'
        loss = sum([torch.mean(((out - gt) ** 2)) for out in self.forward(x)])
        return loss

    def execute(self, x):
        outputs = []
        outputs.append(self.disc_0(x))
        outputs.append(self.disc_1(x))
        outputs.append(self.disc_2(x))
        return outputs

class ResidualBlock(nn.Module):

    def __init__(self, features, norm='in'):
        super(ResidualBlock, self).__init__()
        norm_layer = (AdaptiveInstanceNorm2d if (norm == 'adain') else nn.InstanceNorm2d)
        self.block = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv(features, features, 3), norm_layer(features), nn.ReLU(), nn.ReflectionPad2d(1), nn.Conv(features, features, 3), norm_layer(features))

    def execute(self, x):
        return (x + self.block(x))

# class AdaptiveInstanceNorm2d(nn.Module):
#     'Reference: https://github.com/NVlabs/MUNIT/blob/master/networks.py'

#     def __init__(self, num_features, eps=1e-05, momentum=0.1):
#         super(AdaptiveInstanceNorm2d, self).__init__()
#         self.num_features = num_features
#         self.eps = eps
#         self.momentum = momentum
#         self.weight = None
#         self.bias = None
#         self.register_buffer('running_mean', torch.zeros(num_features))
#         self.register_buffer('running_var', torch.ones(num_features))

#     def execute(self, x):
#         assert ((self.weight is not None) and (self.bias is not None)), 'Please assign weight and bias before calling AdaIN!'
#         (b, c, h, w) = x.shape
#         running_mean = self.running_mean.repeat(b)
#         running_var = self.running_var.repeat(b)
#         x_reshaped = x.contiguous().view((1, (b * c), h, w))
#         out = F.batch_norm(x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps)
#         return out.view((b, c, h, w))

#     def __repr__(self):
#         return (((self.__class__.__name__ + '(') + str(self.num_features)) + ')')

class LayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-05, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = init.uniform((num_features,), 'float32', 0, 1)
            self.beta = init.constant((num_features,), 'float32', 0)

    def execute(self, x):
        shape = ([(- 1)] + ([1] * (x.dim() - 1)))
        mean = x.view(x.size(0), (- 1)).mean(1).view(*shape)
        std = x.view(x.size(0), (- 1)).std(1).view(*shape)
        x = ((x - mean) / (std + self.eps))
        if self.affine:
            shape = ([1, (- 1)] + ([1] * (x.dim() - 2)))
            x = ((x * self.gamma.view(*shape)) + self.beta.view(*shape))
        return x
