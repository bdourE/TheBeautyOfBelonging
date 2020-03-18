import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets as  datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

from torchvision.utils import make_grid
import matplotlib.pyplot as plt


lastCheckpoint = 514

lr = 0.0002
b1 = 0.5
b2  = 0.999
latent_dim = 100

img_height = 512
img_width  = 272
channels = 3
img_shape = (channels, img_height,  img_width )

outputDir = "Results"
CheckpointsDir = "state_dict"

os.makedirs("{}".format(output), exist_ok=True)



class Generator(nn.Module):
  def __init__(self, input_dim=100, channels=3, memory=8):
    super().__init__()
    self.input_dim = input_dim
    self.decoder = nn.Sequential(  # fully convolutional model

      # input is Z, going into a convolution
      nn.ConvTranspose2d( input_dim, memory * 64, 4, 1, 0, bias=False),
      nn.BatchNorm2d(memory * 64),
      nn.ReLU(True),
      # state size.  4 x 4
      nn.ConvTranspose2d(memory * 64, memory * 32, 4, 2, 1, bias=False),
      nn.BatchNorm2d(memory * 32),
      nn.ReLU(True),
      # state size. 8 x 8
      nn.ConvTranspose2d( memory * 32, memory * 16, 4, 2, 1, bias=False),
      nn.BatchNorm2d(memory * 16),
      nn.ReLU(True),
      # state size. 16 x 16
      nn.ConvTranspose2d( memory * 16, memory * 8, 4, (2,1), 1, bias=False),
      nn.BatchNorm2d(memory * 8),
      nn.ReLU(True),
      # state size. 32 x 64
      nn.ConvTranspose2d( memory * 8, memory * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(memory * 4),
      nn.ReLU(True),
      # state size. 64 x 128
      nn.ConvTranspose2d( memory * 4, memory * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(memory * 2),
      nn.ReLU(True),
      # state size. 128 x 256
      nn.ConvTranspose2d( memory * 2, memory, 4, 2, 1, bias=False),
      nn.BatchNorm2d(memory),
      nn.ReLU(True),
      # state size.  256 x 512
      nn.ConvTranspose2d( memory, channels, 4, 2, 1, bias=False),
      nn.Tanh()
      # state size. 512 x 256
    )

  def forward(self, latent_code):
    return self.decoder(latent_code.view(-1, self.input_dim, 1, 1))


class Discriminator(nn.Module):
  def __init__(self, channels=3, memory=8):
    super().__init__()
    self.features = nn.Sequential(  # fully convolutional model

      nn.Conv2d(channels, memory, 4, (2,1), 1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(memory, memory * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(memory * 2),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(memory * 2, memory * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(memory * 4),
      nn.LeakyReLU(0.2, inplace=True),
      
      nn.Conv2d(memory * 4, memory * 8, 4, 2, 1, bias=False),
      nn.BatchNorm2d(memory * 8),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(memory * 8, memory * 16, 4, 2, 1, bias=False),
      nn.BatchNorm2d(memory * 16),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(memory * 16, memory * 32, 4, 2, 1, bias=False),
      nn.BatchNorm2d(memory * 32),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(memory * 32, memory * 64, 4, 2, 1, bias=False),
      nn.BatchNorm2d(memory * 64),
      nn.LeakyReLU(0.2, inplace=True),
      
      
      nn.Conv2d(memory * 64, 1, 4, 1, 0, bias=False),
      
    
    )


  def forward(self, images):

    return self.features(images).flatten(1).mean(1, keepdim=True)




# Loss weight for gradient penalty
lambda_gp = 10

if torch.cuda.device_count() > 0:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# Initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))


def load_model(epoch):
    state = torch.load(f'state_dict/state_dict_{epoch}.pt')
    generator.load_state_dict(state['generator'])
    discriminator.load_state_dict(state['discriminator'])
    #optimizer_G.load_state_dict(state['g_optimizer'])
    #optimizer_D.load_state_dict(state['d_optimizer'])

load_model(lastCheckpoint)

def plot_image_grid(images, columns=8, ax=None, show=True):
  if ax is None:
    _, ax = plt.subplots(figsize=(10, 10))
  image_grid = make_grid(images.detach().cpu(), columns, normalize=True)
  ax.imshow(image_grid.permute(1, 2, 0), interpolation='nearest')
  ax.axis('off')
  if show:
    plt.show(ax.figure)
  return ax

# every time you run this, you will plot new randomly generated samples
with torch.no_grad():
  generator.train(False)
  batch_size = 1
  z = torch.randn(batch_size, 100, device=device)
  plot_image_grid(generator(z))



