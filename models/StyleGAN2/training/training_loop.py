import jittor as jt
from jittor import transform
import os
import time
from tqdm import tqdm
import numpy as np
import PIL
from training.misc import *
from training.utils.visualize import visualize_training, visualize_generate
from .loss import *
import io
from typing import Union
import wandb
import copy
from .augment import *
from training.utils import training_stats
import random

def make_noise(batch, latent_dim, n_noise):
    if n_noise == 1:
        return [jt.randn(batch, latent_dim)]
    noises = jt.randn(n_noise, batch, latent_dim).unbind(0)
    return list(noises)


def mixing_noise(batch, latent_dim, prob):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2)
    else:
        return make_noise(batch, latent_dim, 1)

def init_weights(m):
    if hasattr(m, "weight") and m.weight is not None:
        m.weight.data.fill(0.01)
    if hasattr(m, "bias") and m.bias is not None:
        m.bias.data.fill(0.01)

def accumulate(G_ema, G, rate):
    G_vars = G.state_dict()
    G_ema_vars = G_ema.state_dict()
    for key in G_ema_vars:
        G_ema_vars[key] = rate * G_ema_vars[key] + G_vars[key] * (1 - rate)
    G_ema.load_state_dict(G_ema_vars)

def training_loop(cfg):
    start_time = time.time()

    if jt.rank == 0:
        print('Loading training set...')
    transformer = transform.Compose([
        transform.RandomHorizontalFlip(),
        transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    loader = jt.dataset.ImageFolder(cfg.data_path, transform=transformer).set_attrs(batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    if jt.rank == 0:
        print('Loading networks...')
    G = define_G(cfg)
    D = define_D(cfg)
    G_ema = copy.deepcopy(G)
    G_ema.eval()

    sample_z = jt.randn((cfg.sample_size, cfg.z_dim))
    d_mbr = cfg.D_reg_interval / (cfg.D_reg_interval + 1)
    g_mbr = cfg.G_reg_interval / (cfg.G_reg_interval + 1)
    d_opt = jt.optim.Adam(D.parameters(), lr=cfg.dlr*d_mbr, betas=(cfg.dbeta1**d_mbr, cfg.dbeta2**d_mbr), eps=cfg.deps)
    g_opt = jt.optim.Adam(G.parameters(), lr=cfg.glr*g_mbr, betas=(cfg.gbeta1**g_mbr, cfg.gbeta2**g_mbr), eps=cfg.geps)

    if cfg.resume:
        if jt.rank == 0:
            print('Resuming training from saved generator and discirminator...')
        if os.path.exists(cfg.g_pretrained) and os.path.exists(cfg.d_pretrained):
            G.load(cfg.g_pretrained)
            D.load(cfg.d_pretrained)
            G_ema.load(cfg.g_ema_pretrained)
        else:
            print('There is no trained generator or discriminator at this location, starting training from scratch.')

    visualize_training(cfg)

    if jt.rank == 0:
        print()
        print(f'Start training...')
        print()

    pl_mean = 0.
    sample_z = jt.randn((cfg.sample_size, cfg.z_dim))

    # def D_step(input, r1=False):
    #     noise = mixing_noise(cfg.batch_size, cfg.z_dim, cfg.style_mixing_prob)
    #     set_requires_grad(D.parameters(), True)
    #     set_requires_grad(G.parameters(), False)
    #     D.train()
    #     fake_samples, _ = G(noise)        
    #     pred_fake = D(fake_samples)
    #     pred_real = D(input)
    #     D_loss_fake = jt.nn.softplus(pred_fake)
    #     D_loss_real = jt.nn.softplus(-pred_real)
    #     D_loss = D_loss_fake.mean() + D_loss_real.mean()
    #     d_opt.zero_grad()
    #     d_opt.backward(D_loss)
    #     d_opt.step()

    #     if r1:
    #         input.requires_grad = True
    #         pred_real = D(input)
    #         r1_grads = jt.grad(jt.sum(pred_real), input)
    #         r1_penalty = jt.sum(jt.pow(r1_grads, 2), dims=(1, 2, 3))
    #         d_opt.zero_grad()
    #         d_opt.backward(cfg.r1 / 2 * r1_penalty * cfg.D_reg_interval + 0 * pred_real[0])
    #         d_opt.step()
       
    #     return D_loss.item()

    # def G_step(pl_mean, pl=False):
    #     set_requires_grad(D.parameters(), False)
    #     set_requires_grad(G.parameters(), True)
    #     G.train()
    #     noise = mixing_noise(cfg.batch_size, cfg.z_dim, cfg.style_mixing_prob)
    #     fake_samples, _ = G(noise)
    #     pred_fake = D(fake_samples)
    #     G_loss = jt.mean(jt.nn.softplus(-pred_fake))
    #     g_opt.zero_grad()
    #     g_opt.backward(G_loss)
    #     g_opt.step()

    #     if pl: 
    #         batch_size = max(1, cfg.batch_size // cfg.path_batch_shrink)
    #         noise = mixing_noise(batch_size, cfg.z_dim, cfg.style_mixing_prob)
    #         fake_samples, latents = G(noise)
    #         latents.requires_grad = True
    #         pl_noise = jt.randn_like(fake_samples) / np.sqrt(fake_samples.shape[2] * fake_samples.shape[3])
    #         pl_grads = jt.grad(jt.sum(fake_samples * pl_noise), latents)
    #         pl_lengths = jt.sqrt(jt.mean(jt.sum(jt.pow(pl_grads, 2), dim=2), dim=1))
    #         pl_mean += (jt.mean(pl_lengths) - pl_mean) * 0.01
    #         pl_penalty = jt.pow((pl_lengths - pl_mean), 2)
    #         loss_Gpl = pl_penalty * (cfg.path_regularize)
    #         G_loss_pl = jt.mean(fake_samples[:, 0, 0, 0] * 0 + loss_Gpl) * (cfg.G_reg_interval)
    #         g_opt.zero_grad()
    #         g_opt.backward(G_loss_pl)
    #         g_opt.step()
    #     accum = 0.5 ** (32 / (10 * 1000))
    #     accumulate(G_ema, G, accum)
    #     return pl_mean, G_loss.item()

    num_iter = 0
    for epoch in range(cfg.num_epoch):
        for idx, batch in enumerate(tqdm(loader)):
            real_img, labels = batch
            if real_img.shape[0] != cfg.batch_size // cfg.num_gpus:
                break

            noise = mixing_noise(cfg.batch_size, cfg.z_dim, cfg.style_mixing_prob)
            set_requires_grad(D.parameters(), True)
            set_requires_grad(G.parameters(), False)
            D.train()
            fake_samples, _ = G(noise)        
            pred_fake = D(fake_samples)
            pred_real = D(real_img)
            D_loss_fake = jt.nn.softplus(pred_fake)
            D_loss_real = jt.nn.softplus(-pred_real)
            D_loss = D_loss_fake.mean() + D_loss_real.mean()
            d_opt.zero_grad()
            d_opt.backward(D_loss)
            d_opt.step()

            if num_iter % cfg.D_reg_interval == 0:
                real_img.requires_grad = True
                pred_real = D(real_img)
                r1_grads = jt.grad(jt.sum(pred_real), real_img)
                r1_penalty = jt.sum(jt.pow(r1_grads, 2), dims=(1, 2, 3))
                d_opt.zero_grad()
                d_opt.backward(cfg.r1 / 2 * r1_penalty * cfg.D_reg_interval + 0 * pred_real[0])
                d_opt.step()

            if jt.rank == 0:
                if cfg.use_wandb:
                    wandb.log({'D_loss': D_loss.item()})

            set_requires_grad(D.parameters(), False)
            set_requires_grad(G.parameters(), True)
            G.train()
            noise = mixing_noise(cfg.batch_size, cfg.z_dim, cfg.style_mixing_prob)
            fake_samples, _ = G(noise)
            pred_fake = D(fake_samples)
            G_loss = jt.mean(jt.nn.softplus(-pred_fake))
            g_opt.zero_grad()
            g_opt.backward(G_loss)
            g_opt.step()

            if num_iter % cfg.G_reg_interval == 0:
                batch_size = max(1, cfg.batch_size // cfg.path_batch_shrink)
                noise = mixing_noise(batch_size, cfg.z_dim, cfg.style_mixing_prob)
                fake_samples, latents = G(noise)
                latents.requires_grad = True
                pl_noise = jt.randn_like(fake_samples) / np.sqrt(fake_samples.shape[2] * fake_samples.shape[3])
                pl_grads = jt.grad(jt.sum(fake_samples * pl_noise), latents)
                pl_lengths = jt.sqrt(jt.mean(jt.sum(jt.pow(pl_grads, 2), dim=2), dim=1))
                pl_mean += (jt.mean(pl_lengths) - pl_mean) * 0.01
                pl_penalty = jt.pow((pl_lengths - pl_mean), 2)
                loss_Gpl = pl_penalty * (cfg.path_regularize)
                G_loss_pl = jt.mean(fake_samples[:, 0, 0, 0] * 0 + loss_Gpl) * (cfg.G_reg_interval)
                g_opt.zero_grad()
                g_opt.backward(G_loss_pl)
                g_opt.step()
            accum = 0.5 ** (32 / (10 * 1000))
            accumulate(G_ema, G, accum)

            if jt.rank == 0:
                if cfg.use_wandb:
                    wandb.log({'G_loss': G_loss.item()})

            # jt.sync_all()
            num_iter += 1
        
        jt.sync_all()
        if epoch % cfg.save_freq == 0 and jt.rank == 0:
            print("Saving trained model...")
            os.makedirs(os.path.join(cfg.ckpt_path, cfg.name), exist_ok=True)
            G.save(os.path.join(cfg.ckpt_path, cfg.name, 'G.pkl'))
            G_ema.save(os.path.join(cfg.ckpt_path, cfg.name, 'G_ema.pkl'))
            D.save(os.path.join(cfg.ckpt_path, cfg.name, 'D.pkl'))

        if epoch % cfg.display_freq == 0 and jt.rank == 0:
            os.makedirs(os.path.join(cfg.vis_path, cfg.name), exist_ok=True)
            print("Visualizing generated images...")
            with jt.no_grad():
                images, _ = G_ema([sample_z])
            visualize_generate(images, cfg, epoch + 1)

    if jt.rank == 0:
        print(f'Training finished in {time.time() - start_time} seconds. Exiting...')
