import argparse
import os
import numpy as np
import math
import sys

import imageio

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

img_height = 480
img_width  = 270
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
        nn.ConvTranspose2d(input_dim, memory * 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(memory * 64),
        nn.ReLU(True),
        nn.ConvTranspose2d(memory * 64, memory * 32, kernel_size=4, stride=3, padding=1),
        nn.BatchNorm2d(memory * 32),
        nn.ReLU(True),
        nn.ConvTranspose2d(memory * 32, memory * 16, kernel_size=4, stride=3, padding=1, output_padding=1),
        nn.BatchNorm2d(memory * 16),
        nn.ReLU(True),
        nn.ConvTranspose2d(memory * 16, memory * 8, kernel_size=4, stride=(2, 3), padding=1, output_padding=(0, 2)),
        nn.BatchNorm2d(memory * 8),
        nn.ReLU(True),
        nn.ConvTranspose2d(memory * 8, memory * 4, kernel_size=4, stride=(2, 3), padding=(1, 2)),
        nn.BatchNorm2d(memory * 4),
        nn.ReLU(True),
        nn.ConvTranspose2d(memory * 4, memory * 2, kernel_size=4, stride=(2, 1), padding=(1, 2)),
        nn.BatchNorm2d(memory * 2),
        nn.ReLU(True),
        nn.ConvTranspose2d(memory * 2, memory * 1, kernel_size=4, stride=(2, 1), padding=1),
        nn.BatchNorm2d(memory * 1),
        nn.ReLU(True),
        nn.ConvTranspose2d(memory * 1, channels, kernel_size=4, stride=2, padding=1),
        nn.Tanh(),
    )

  def forward(self, latent_code):
    return self.decoder(latent_code.view(-1, self.input_dim, 1, 1))



class Discriminator(nn.Module):
  def __init__(self, channels=3, memory=8):
    super().__init__()
    self.features = nn.Sequential(  # fully convolutional model
        nn.Conv2d(channels, memory * 64, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(memory * 64),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv2d(memory * 64, memory * 32, kernel_size=4, stride=(2, 1), padding=1),
        nn.BatchNorm2d(memory * 32),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(memory * 32, memory * 16, kernel_size=4, stride=(2, 1), padding=(1, 2)),
        nn.BatchNorm2d(memory * 16),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv2d(memory * 16, memory * 8, kernel_size=4, stride=(2, 3), padding=(1, 2)),
        nn.BatchNorm2d(memory * 8),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv2d(memory * 8, memory * 4, kernel_size=4, stride=(2, 3), padding=1),
        nn.BatchNorm2d(memory * 4),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(memory * 4, memory * 2, kernel_size=4, stride=3, padding=1),
        nn.BatchNorm2d(memory * 2),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv2d(memory * 2, memory * 1, kernel_size=4, stride=3, padding=1),
        nn.BatchNorm2d(memory * 1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(memory * 1, 1, kernel_size=3, stride=2, padding=1),
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
    # state = torch.load(f'/content/drive/My Drive/Colab Notebooks/Saudi_artist/Models/state_dict_{epoch}.pt')
    state = torch.load(f'state_dict/state_dict_{epoch}.pt')
    generator.load_state_dict(state['generator'])
    discriminator.load_state_dict(state['discriminator'])
    optimizer_G.load_state_dict(state['g_optimizer'])
    optimizer_D.load_state_dict(state['d_optimizer'])

load_model(lastCheckpoint)

def create_animation(images):
  
  imageio.mimsave('LatentSpaceInterpolation.gif', images, format = 'GIF-PIL')


seed = 42 
num_pairs = 5
num_interpolations = 7

def interpolate(x, y, count=num_interpolations):
  alphas = torch.linspace(0, 1, count, device=x.device)
  out = torch.stack([(1 - a) * x + a * y for a in alphas])
  return out.permute(1, 0, 2).contiguous()

torch.manual_seed(seed)
z_1 = torch.randn(num_pairs, 100, device=device)
z_2 = torch.randn(num_pairs, 100, device=device)

images = []
with torch.no_grad():
  generator.train(False)
  fake_im = generator(interpolate(z_1, z_2))
  print(fake_im.shape)
      #plot_image_grid(fake_im, number_int )
  #num_pairs * num_interpolations
  for i in range(34) : 
    image0 = fake_im.detach().cpu().numpy()[i,:,:,:]
    data = np.transpose(image0,(1,2,0))
    images.append(data)

create_animation(images) 





