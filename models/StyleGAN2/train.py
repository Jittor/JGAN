import jittor as jt
from munch import Munch
from training.training_loop import training_loop
import os
from training.misc import *
import math
import wandb
import argparse

def train():
    cfg = get_config()
    if cfg.gpu:
        jt.flags.use_cuda = 1
    else:
        jt.flags.use_cuda = 0
    if cfg.use_wandb:
        wandb.init(project=cfg.name)
    training_loop(cfg)

def get_config():
    with open("config.yaml", "r") as f:
        config = Munch.fromYAML(f)

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("--data_path", type=str, default=None, help="path to the dataset")
    parser.add_argument("--eval_path", type=str, default=None, help="path to the evaluation folder")
    parser.add_argument(
        "--epoch", type=int, default=None, help="total training epochs"
    )
    parser.add_argument(
        "--batch", type=int, default=None, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=None,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=None,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=None,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="number of gpus to use",
    )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    args = parser.parse_args()

    if args.batch is not None:
        config.batch_size = args.batch
    if args.lr is not None:
        config.glr = args.lr
        config.dlr = args.lr
    if args.data_path is not None:
        config.name = os.path.basename(args.data_path)
        config.data_path = args.data_path
    if args.epoch is not None:
        config.num_epoch = args.epoch
    if args.d_reg_every is not None:
        config.D_reg_interval = args.d_reg_every
    if args.g_reg_every is not None:
        config.G_reg_interval = args.g_reg_every
    if args.wandb:
        config.use_wandb = True
    if args.n_sample is not None:
        config.sample_size = args.n_sample
    if args.num_gpus is not None:
        config.num_gpus = args.num_gpus
    if args.ckpt_path is not None:
        config.ckpt_path = args.ckpt_path
        config.g_pretrained = os.path.join(config.ckpt_path, config.name, 'G.pkl')
        config.d_pretrained = os.path.join(config.ckpt_path, config.name, 'D.pkl')
        config.g_ema_pretrained = os.path.join(config.ckpt_path, config.name, 'G_ema.pkl')

    return config

if __name__ == "__main__":
    train()
