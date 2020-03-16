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


n_epochs = 500
batch_size = 64
lr = 0.0002
b1 = 0.5
b2  = 0.999
latent_dim = 100

#img_size  = 28
img_height = 480
img_width  = 270
channels = 3
n_critic = 5
clip_value = 0.01
sample_interval = 400
img_shape = (channels, img_height,  img_width )


outputDir = "Results"
CheckpointsDir = "State_dict"

os.makedirs("{}".format(output), exist_ok=True)
os.makedirs("{}".format(CheckpointsDir), exist_ok=True)


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


def pixel_stats(image):
    return image.reshape(-1, 3).mean(0), image.reshape(-1, 3).std(0)

def color_transfer(target, source, source_std=None):
    target = rgb2lab(target)
    target_mean, target_std = pixel_stats(target)
    if source_std is None:
        source, source_std = pixel_stats(rgb2lab(source))
    scale = source_std / target_std
    bias = source - scale * target_mean
    transformed = scale * target + bias
    lightness_bound = (116. / 200.) * transformed[..., 2] + (1e-10 - 16.)
    transformed[..., 0] = np.maximum(transformed[..., 0], lightness_bound)
    output = lab2rgb(transformed) * 255
    return output.round().astype(np.uint8)

from PIL import Image
from skimage.color import rgb2lab, lab2rgb
# now, we can define the color swap transform as follows
class ColorSwap:
    def __init__(self, 
                 scale_mean=(15, 25, 25), scale_std=(5, 10, 10),
                 shift_mean=(70, 0, 0), shift_std=(10, 20, 20)):
        self.scale_mean = scale_mean
        self.scale_std = scale_std
        self.shift_mean = shift_mean
        self.shift_std = shift_std

    def __call__(self, image):
        mean = np.random.randn(3) * self.shift_std + self.shift_mean
        std = np.random.randn(3) * self.scale_std + self.scale_mean
        return Image.fromarray(color_transfer(image, mean, std))



# Configure data loader
directory = "Dataset"


image_size = (512, 272)

train_transform = transforms.Compose([
    transforms.Resize(int(min(image_size) * 1.5)),
    transforms.RandomCrop(image_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([ColorSwap()], p=0.5),
    transforms.Normalize([0.5], [0.5]),
    transforms.ToTensor(),
])

Dataset =  datasets.ImageFolder (directory,train_transform)

dataloader = DataLoader(Dataset, batch_size,
                        shuffle=True, drop_last=True, num_workers=4,
                        pin_memory=device.type == 'cuda')


im , _ = iter(dataloader).next()
print(im.shape)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    # alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
    # Get random interpolation between real and fake samples
    #print(alpha.shape)
    #print(real_samples.shape)
    #print(fake_samples.shape)

    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(torch.empty(real_samples.shape[0], 1, device=device).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def save_model(epoch):
    state = {
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'g_optimizer': optimizer_G.state_dict(),
        'd_optimizer': optimizer_D.state_dict(),
    }
    # torch.save(state, f'/content/drive/My Drive/Colab Notebooks/Saudi_artist/Models/state_dict_{epoch}.pt')
    torch.save(state, f'state_dict/state_dict_{epoch}.pt')
    print('Models save')

def load_model(epoch):
    # state = torch.load(f'/content/drive/My Drive/Colab Notebooks/Saudi_artist/Models/state_dict_{epoch}.pt')
    state = torch.load(f'state_dict/state_dict_{epoch}.pt')
    generator.load_state_dict(state['generator'])
    discriminator.load_state_dict(state['discriminator'])
    optimizer_G.load_state_dict(state['g_optimizer'])
    optimizer_D.load_state_dict(state['d_optimizer'])

# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.float().to(device))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(torch.randn(imgs.shape[0], latent_dim, device=device))

        # Generate a batch of images
        fake_imgs = generator(z)

        # Real images
        real_validity = discriminator(real_imgs)
        # Fake images
        fake_validity = discriminator(fake_imgs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty


        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            # Wasserstein Disctance
            real_validity = real_validity.mean()
            fake_validity = -torch.mean(fake_validity)
            Wasserstein_D = real_validity - fake_validity
            print ("Wasserstein disctance  : ")
            print(Wasserstein_D.item())

            if batches_done % sample_interval == 0:
                save_image(fake_imgs.data[:25], "{}/%d.png".format(output) % batches_done, nrow=5, normalize=True)
                save_model(epoch)

            batches_done += n_critic




