"""
StarGAN discriminator for adversarial loss.

This module contains the StarGAN discriminator implementation
from the original paper and functions to load pretrained weights from Dropbox.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        # Negate source output because the checkpoint has inverted behavior
        # (higher values mean more fake, lower values mean more real)
        return -out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


def load_stargan_discriminator(use_gpu=True, img_size=128):
    """
    Load StarGAN discriminator from pretrained weights.

    Args:
        use_gpu: Whether to load the discriminator on GPU
        img_size: Image size (128 or 256)

    Returns:
        StarGAN discriminator model
    """
    # Create discriminator
    discriminator = Discriminator(
        image_size=img_size,
        conv_dim=64,
        c_dim=5,  # Number of attributes in CelebA
        repeat_num=6,
    )

    # Try to load from local data directory first
    weights_file = os.path.join(os.path.dirname(__file__), '../../../data/celeba-128x128-5attrs/200000-D.ckpt')
    weights_file = os.path.abspath(weights_file)

    if os.path.exists(weights_file):
        print(f"Loading StarGAN discriminator weights from {weights_file}")
        state_dict = torch.load(weights_file, map_location='cpu')
    else:
        print(f"Warning: StarGAN discriminator weights not found at {weights_file}")
        print("Using randomly initialized discriminator (not recommended)")
        state_dict = None

    # Load weights
    if state_dict is not None:
        # Handle different checkpoint formats
        if 'discriminator' in state_dict:
            discriminator.load_state_dict(state_dict['discriminator'])
        elif 'model_state_dict' in state_dict:
            discriminator.load_state_dict(state_dict['model_state_dict'])
        elif 'state_dict' in state_dict:
            discriminator.load_state_dict(state_dict['state_dict'])
        else:
            # Try loading directly (checkpoint is already a state_dict)
            discriminator.load_state_dict(state_dict)

    if use_gpu and torch.cuda.is_available():
        discriminator = discriminator.cuda()

    discriminator.eval()
    return discriminator
