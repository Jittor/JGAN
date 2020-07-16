
import jittor as jt
import argparse
import os
import numpy as np
import math
from jittor import nn
import pickle
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
opt = parser.parse_args()
print(opt)

# Define Model
generator = Generator(opt)
discriminator = Discriminator(opt)

# Load Parameters
generator.load_parameters(pickle.load(open('saved_models/generator_last.pkl', 'rb')))
discriminator.load_parameters(pickle.load(open('saved_models/discriminator_last.pkl', 'rb')))

import cv2
def save_image(img, path, nrow=10, padding=5):
    N,C,W,H = img.shape
    if (N%nrow!=0):
        print("N%nrow!=0")
        return
    ncol=int(N/nrow)
    img_all = []
    for i in range(ncol):
        img_ = []
        for j in range(nrow):
            img_.append(img[i*nrow+j])
            img_.append(np.zeros((C,W,padding)))
        img_all.append(np.concatenate(img_, 2))
        img_all.append(np.zeros((C,padding,img_all[0].shape[2])))
    img = np.concatenate(img_all, 1)
    img = np.concatenate([np.zeros((C,padding,img.shape[2])), img], 1)
    img = np.concatenate([np.zeros((C,img.shape[1],padding)), img], 2)
    min_=img.min()
    max_=img.max()
    img=(img-min_)/(max_-min_)*255
    img=img.transpose((1,2,0))
    if C==3:
        img = img[:,:,::-1]
    cv2.imwrite(path,img)

# ----------
#  Test
# ----------
number = "201962517"
n_row = len(number)
z = jt.array(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))).float32().stop_grad()
labels = jt.array(np.array([int(number[num]) for _ in range(n_row) for num in range(n_row)])).float32().stop_grad()
gen_imgs = generator(z,labels)
save_image(gen_imgs.numpy(), "result.png", nrow=n_row)