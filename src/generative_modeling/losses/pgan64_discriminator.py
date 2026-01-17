"""
Adversarial loss using pretrained PGAN discriminator at 64x64 resolution.

This uses the PGAN discriminator but downscales images to 64x64
before feeding them, significantly reducing memory usage.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_pgan_discriminator_scale4(use_gpu=True):
    """
    Load PGAN discriminator at scale 4 (64x64 resolution).

    Args:
        use_gpu: Whether to load the discriminator on GPU

    Returns:
        PGAN discriminator model
    """
    # First, ensure the hub repo is downloaded
    hub_dir = torch.hub.get_dir()
    repo_dir = os.path.join(hub_dir, 'facebookresearch_pytorch_GAN_zoo_hub')

    if not os.path.exists(repo_dir):
        # Download repo
        print("Downloading PGAN repository...")
        torch.hub._get_cache_or_reload(
            'facebookresearch/pytorch_GAN_zoo:hub',
            force_reload=False,
            trust_repo=True,
            calling_fn='load'
        )

    # Add repo to path temporarily
    sys.path.insert(0, repo_dir)

    try:
        # Import the discriminator network class directly
        from models.networks.progressive_conv_net import DNet
        from torch.utils import model_zoo

        # CelebA cropped PGAN config: trained at 64x64 (scale 4)
        # Scale progression: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        # Scale 4 = 64x64, with 4 addScale calls after init

        # Create discriminator with base config
        netD = DNet(
            depthScale0=512,
            initBiasToZero=True,
            leakyReluLeak=0.2,
            sizeDecisionLayer=1,
            miniBatchNormalization=True,
            dimInput=3,
            equalizedlR=True,
        )

        # Add scales: depths are [512, 512, 512, 256] for scales 1-4
        scale_depths = [512, 512, 512, 256]
        for depth in scale_depths:
            netD.addScale(depth)

        # Set alpha to 1.0 (fully transitioned to highest resolution)
        netD.setNewAlpha(1.0)

        # Download checkpoint for scale 4 (64x64)
        checkpoint_url = 'https://dl.fbaipublicfiles.com/gan_zoo/PGAN/celebaCropped_s4_i83000-2b0acc76.pth'
        print("Downloading PGAN checkpoint for scale 4 (64x64)...")
        state_dict = model_zoo.load_url(checkpoint_url, map_location='cpu')

        # Load discriminator weights
        netD.load_state_dict(state_dict['netD'])

        if use_gpu and torch.cuda.is_available():
            netD = netD.cuda()

        netD.eval()

        return netD

    finally:
        # Remove repo from path
        if repo_dir in sys.path:
            sys.path.remove(repo_dir)


class PGAN64DiscriminatorLoss(nn.Module):
    """
    Adversarial loss using a pretrained PGAN discriminator at 64x64 resolution.

    Images are downscaled to 64x64 before being fed to the discriminator,
    which significantly reduces memory usage.

    The discriminator outputs a scalar: higher values mean "more real".
    For VAE training, we want reconstructions to fool the discriminator,
    so we maximize D(recon) which equals minimizing -D(recon).
    """

    def __init__(self, use_gpu=True):
        super().__init__()
        self.use_gpu = use_gpu
        self.discriminator = None
        self._loaded = False

    def _ensure_loaded(self, device):
        """Lazy load the discriminator on first use."""
        if self._loaded:
            return

        print("Loading pretrained PGAN discriminator (64x64)...")
        self.discriminator = _load_pgan_discriminator_scale4(use_gpu=self.use_gpu)

        # Freeze discriminator weights
        for param in self.discriminator.parameters():
            param.requires_grad = False

        # Move to correct device
        self.discriminator = self.discriminator.to(device)
        self._loaded = True
        print("PGAN discriminator loaded successfully!")

    def _preprocess(self, x):
        """
        Preprocess images for PGAN discriminator.

        Converts [0,1] to [-1,1] and downscales to 64x64.
        """
        # Convert from [0, 1] to [-1, 1]
        x = x * 2.0 - 1.0

        # Downscale to 64x64 (from 128x128)
        if x.shape[2] != 64 or x.shape[3] != 64:
            x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)

        return x

    def forward(self, recon_x, target_x=None):
        """
        Compute adversarial loss.

        Args:
            recon_x: Reconstructed images (B, 3, H, W) in [0, 1]
            target_x: Original images (B, 3, H, W) in [0, 1], optional

        Returns:
            loss: Scalar adversarial loss (to be minimized)
            d_recon: Mean discriminator score on reconstructions (higher = more realistic)
            d_real: Mean discriminator score on real images (if target_x provided)
        """
        self._ensure_loaded(recon_x.device)

        # Get discriminator score for reconstructions
        recon_preprocessed = self._preprocess(recon_x)

        with torch.no_grad():
            d_recon = self.discriminator(recon_preprocessed).squeeze()

        # For generator/VAE training: we want to maximize D(recon)
        # Since we minimize loss, use -D(recon)
        # Using softplus for smoother gradients: -log(sigmoid(D(recon)))
        adv_loss = F.softplus(-d_recon).mean()

        d_real = None
        if target_x is not None:
            target_preprocessed = self._preprocess(target_x)
            with torch.no_grad():
                d_real = self.discriminator(target_preprocessed).squeeze().mean()

        return adv_loss, d_recon.mean(), d_real


def get_pgan64_discriminator(use_gpu=True):
    """
    Convenience function to load just the PGAN discriminator at 64x64.

    Args:
        use_gpu: Whether to load the discriminator on GPU

    Returns the raw discriminator network for direct use.
    """
    discriminator = _load_pgan_discriminator_scale4(use_gpu=use_gpu)

    for param in discriminator.parameters():
        param.requires_grad = False

    return discriminator


def pgan64_adversarial_loss(recon_x, discriminator, preprocess=True):
    """
    Compute PGAN adversarial loss with 64x64 downscaling.

    Args:
        recon_x: Reconstructed images (B, 3, H, W) in [0, 1]
        discriminator: PGAN discriminator model
        preprocess: Whether to preprocess images (convert to [-1, 1] and downscale)

    Returns:
        loss: Scalar adversarial loss
        d_score: Mean discriminator score
    """
    if preprocess:
        x = recon_x * 2.0 - 1.0  # discriminator expects [-1, 1]
        # Downscale to 64x64
        if x.shape[2] != 64 or x.shape[3] != 64:
            x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
    else:
        x = recon_x

    d_score = discriminator(x).squeeze()

    loss = F.softplus(-d_score).mean()

    return loss, d_score.mean()
