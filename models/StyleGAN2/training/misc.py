from . import stylegan2


def define_G(opt):
    # w_shift = opt.optim_param_g == 'w_shift'
    generator = stylegan2.Generator(
        opt.size, opt.z_dim, opt.n_mlp, lr_mlp=opt.lr_mlp, channel_multiplier=opt.channel_multiplier)
    return generator

def define_GNet(opt):
    # w_shift = opt.optim_param_g == 'w_shift'
    generator = networks.Generator(
        opt.z_dim, 0, opt.z_dim, opt.size, 3, 
        mapping_kwargs={"num_layers": 2}, 
        synthesis_kwargs={"channel_base": 16384,
                          "channel_max": 512,
                          "num_fp16_res": 4,
                          "conv_clamp": 256})
    return generator


def define_D(opt):
    discriminator = stylegan2.Discriminator(opt.size, channel_multiplier=opt.channel_multiplier)
    return discriminator

def define_DNet(opt):
    discriminator = networks.Discriminator(
        0, opt.size, 3, channel_base=16384, channel_max=512, num_fp16_res=4, conv_clamp=256
    )
    return discriminator


def set_requires_grad(params, flag):
    for p in params:
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def accumulate_by_keys(keys, g_ema, g_train, decay=0.999):
    """Only accumulate weights in the given list of keys."""
    dict_trn = dict(g_train.named_parameters())
    dict_ema = dict(g_ema.named_parameters())

    for k in keys:
        assert k in dict_ema, "key %s is not in the param dict of G_ema." % k
        dict_ema[k].data.mul_(decay).add_(dict_trn[k].data, alpha=1 - decay)
