"""
Adversarial loss using pretrained discriminators.

Supports:
- Progressive GAN (PGAN) discriminator trained on CelebA (128x128 center-cropped)
- StarGAN discriminator trained on CelebA (128x128)

The discriminator judges whether images look like real CelebA faces.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import standalone StarGAN discriminator
from .stargan_discriminator import Discriminator, load_stargan_discriminator


def _load_pgan_discriminator_direct(use_gpu=True, scale=5):
    """
    Load PGAN discriminator directly, working around optimizer compatibility issues.

    The torch.hub PGAN loader has issues with newer PyTorch versions due to
    optimizer beta parameter format changes. This function builds the discriminator
    network directly and loads the weights from the checkpoint.

    Args:
        use_gpu: Whether to load the discriminator on GPU
        scale: PGAN scale (0=4x4, 1=8x8, 2=16x16, 3=32x32, 4=64x64, 5=128x128)
    """
    # First, ensure the hub repo is downloaded
    hub_dir = torch.hub.get_dir()
    repo_dir = os.path.join(hub_dir, 'facebookresearch_pytorch_GAN_zoo_hub')
    
    if not os.path.exists(repo_dir):
        # Download the repo
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

        # CelebA cropped PGAN config: trained at different resolutions
        # Scale progression: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
        # Scale 0=4x4, 1=8x8, 2=16x16, 3=32x32, 4=64x64, 5=128x128

        # Map scale to checkpoint URL and configuration
        scale_configs = {
            0: {"url": 'https://dl.fbaipublicfiles.com/gan_zoo/PGAN/celebaCropped_s0_i83000-2b0acc76.pth', "depths": []},
            1: {"url": 'https://dl.fbaipublicfiles.com/gan_zoo/PGAN/celebaCropped_s1_i83000-2b0acc76.pth', "depths": [512]},
            2: {"url": 'https://dl.fbaipublicfiles.com/gan_zoo/PGAN/celebaCropped_s2_i83000-2b0acc76.pth', "depths": [512, 512]},
            3: {"url": 'https://dl.fbaipublicfiles.com/gan_zoo/PGAN/celebaCropped_s3_i83000-2b0acc76.pth', "depths": [512, 512, 512]},
            4: {"url": 'https://dl.fbaipublicfiles.com/gan_zoo/PGAN/celebaCropped_s4_i83000-2b0acc76.pth', "depths": [512, 512, 512, 256]},
            5: {"url": 'https://dl.fbaipublicfiles.com/gan_zoo/PGAN/celebaCropped_s5_i83000-2b0acc76.pth', "depths": [512, 512, 512, 256, 128]},
        }

        if scale not in scale_configs:
            raise ValueError(f"Invalid scale {scale}. Must be 0-5.")

        config = scale_configs[scale]
        scale_depths = config["depths"]
        checkpoint_url = config["url"]

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

        # Add scales based on the selected resolution
        for depth in scale_depths:
            netD.addScale(depth)

        # Set alpha to 1.0 (fully transitioned to highest resolution)
        netD.setNewAlpha(1.0)

        # Download checkpoint
        print(f"Downloading PGAN checkpoint for scale {scale} ({4 * (2**scale)}x{4 * (2**scale)})...")
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


class PGANDiscriminatorLoss(nn.Module):
    """
    Adversarial loss using a pretrained PGAN discriminator.

    The discriminator outputs a scalar: higher values mean "more real".
    For VAE training, we want reconstructions to fool the discriminator,
    so we maximize D(recon) which equals minimizing -D(recon).

    We use a feature-matching loss (comparing intermediate features)
    combined with the discriminator score for more stable training.
    """

    def __init__(self, use_gpu=True, feature_matching=True, scale=5):
        super().__init__()
        self.use_gpu = use_gpu
        self.feature_matching = feature_matching
        self.scale = scale
        self.discriminator = None
        self._loaded = False
    
    def _ensure_loaded(self, device):
        """Lazy load the discriminator on first use."""
        if self._loaded:
            return

        print(f"Loading pretrained PGAN discriminator (scale {self.scale})...")
        self.discriminator = _load_pgan_discriminator_direct(use_gpu=self.use_gpu, scale=self.scale)
        
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

        PGAN expects images in [-1, 1] range, but our VAEs output [0, 1].
        Also handles potential size mismatches.
        """
        # Convert from [0, 1] to [-1, 1]
        x = x * 2.0 - 1.0

        # PGAN expects different sizes based on scale
        target_size = 4 * (2 ** self.scale)
        if x.shape[2] != target_size or x.shape[3] != target_size:
            x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)

        return x
    
    def get_discriminator_features(self, x):
        """
        Get discriminator output and intermediate features.
        
        Returns:
            output: Final discriminator score (B,)
            features: List of intermediate feature maps for feature matching
        """
        self._ensure_loaded(x.device)
        
        x = self._preprocess(x)
        features = []
        
        # PGAN discriminator architecture has progressive layers
        # We'll extract features from different scales
        with torch.no_grad():
            # Forward through discriminator, collecting intermediate features
            # The PGAN discriminator has layers: fromRGBLayers, scaleLayers, toRGBLayers
            # We need to inspect the actual structure
            
            # For PGAN, the discriminator processes through scale layers
            # Let's just get the final output and a few intermediate activations
            output = self.discriminator(x)
            
            # Note: For full feature matching, we'd need to hook into intermediate layers
            # For now, we'll use the output score directly
            
        return output.squeeze(), features
    
    def forward(self, recon_x, target_x=None):
        """
        Compute adversarial loss.
        
        Args:
            recon_x: Reconstructed images (B, 3, H, W) in [0, 1]
            target_x: Original images (B, 3, H, W) in [0, 1], optional for feature matching
            
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


def get_pgan_discriminator(use_gpu=True, scale=5):
    """
    Convenience function to load just the PGAN discriminator.

    Args:
        use_gpu: Whether to load the discriminator on GPU
        scale: PGAN scale (0=4x4, 1=8x8, 2=16x16, 3=32x32, 4=64x64, 5=128x128)

    Returns the raw discriminator network for direct use.
    """
    discriminator = _load_pgan_discriminator_direct(use_gpu=use_gpu, scale=scale)
    
    for param in discriminator.parameters():
        param.requires_grad = False
    
    return discriminator


def adversarial_loss(recon_x, discriminator, preprocess=True, scale=5):
    if preprocess:
        x = recon_x * 2.0 - 1.0 # discriminator expects [-1, 1]
        target_size = 4 * (2 ** scale)
        if x.shape[2] != target_size or x.shape[3] != target_size:
            x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)
    else:
        x = recon_x

    d_score = discriminator(x).squeeze()

    loss = F.softplus(-d_score).mean()

    return loss, d_score.mean()


# ============================================================================
# StarGAN Discriminator
# ============================================================================

def _load_stargan_discriminator(use_gpu=True, img_size=128):
    """
    Load StarGAN discriminator from pretrained weights.

    Args:
        use_gpu: Whether to load the discriminator on GPU
        img_size: Image size (128 or 256)

    Returns:
        StarGAN discriminator model
    """
    # Use standalone discriminator
    return load_stargan_discriminator(use_gpu=use_gpu, img_size=img_size)


class StarGANDiscriminatorLoss(nn.Module):
    """
    Adversarial loss using a pretrained StarGAN discriminator.

    The discriminator outputs a patch-based output: higher values mean "more real".
    For VAE training, we want reconstructions to fool the discriminator,
    so we maximize D(recon) which equals minimizing -D(recon).
    """

    def __init__(self, use_gpu=True, img_size=128):
        super().__init__()
        self.use_gpu = use_gpu
        self.img_size = img_size
        self.discriminator = None
        self._loaded = False

    def _ensure_loaded(self, device):
        """Lazy load the discriminator on first use."""
        if self._loaded:
            return

        print(f"Loading pretrained StarGAN discriminator ({self.img_size}x{self.img_size})...")
        self.discriminator = load_stargan_discriminator(use_gpu=self.use_gpu, img_size=self.img_size)

        # Freeze discriminator weights
        for param in self.discriminator.parameters():
            param.requires_grad = False

        # Move to correct device
        self.discriminator = self.discriminator.to(device)
        self._loaded = True
        print("StarGAN discriminator loaded successfully!")

    def _preprocess(self, x):
        """
        Preprocess images for StarGAN discriminator.

        StarGAN expects images in [-1, 1] range, but our VAEs output [0, 1].
        Also handles potential size mismatches.
        """
        # Convert from [0, 1] to [-1, 1]
        x = x * 2.0 - 1.0

        # Resize if needed
        if x.shape[2] != self.img_size or x.shape[3] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

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
            # StarGAN returns (patch_output, class_output)
            d_recon_patch, _ = self.discriminator(recon_preprocessed)
            d_recon = d_recon_patch.mean()

        # For generator/VAE training: we want to maximize D(recon)
        # Using BCE with logits: we want D(recon) to be close to 1 (real)
        adv_loss = F.binary_cross_entropy_with_logits(
            d_recon_patch,
            torch.ones_like(d_recon_patch)
        ).mean()

        d_real = None
        if target_x is not None:
            target_preprocessed = self._preprocess(target_x)
            with torch.no_grad():
                d_real_patch, _ = self.discriminator(target_preprocessed)
                d_real = d_real_patch.mean()

        return adv_loss, d_recon, d_real


def get_stargan_discriminator(use_gpu=True, img_size=128):
    """
    Convenience function to load just the StarGAN discriminator.

    Args:
        use_gpu: Whether to load the discriminator on GPU
        img_size: Image size (128 or 256)

    Returns the raw discriminator network for direct use.
    """
    discriminator = load_stargan_discriminator(use_gpu=use_gpu, img_size=img_size)

    for param in discriminator.parameters():
        param.requires_grad = False

    return discriminator


def stargan_adversarial_loss(recon_x, discriminator, preprocess=True, img_size=128):
    """
    Compute StarGAN adversarial loss.

    Args:
        recon_x: Reconstructed images (B, 3, H, W) in [0, 1]
        discriminator: StarGAN discriminator model
        preprocess: Whether to preprocess images (convert to [-1, 1] and resize)
        img_size: Target image size for discriminator

    Returns:
        loss: Scalar adversarial loss
        d_score: Mean discriminator score
    """
    if preprocess:
        x = recon_x * 2.0 - 1.0  # discriminator expects [-1, 1]
        if x.shape[2] != img_size or x.shape[3] != img_size:
            x = F.interpolate(x, size=(img_size, img_size), mode='bilinear', align_corners=False)
    else:
        x = recon_x

    # StarGAN returns (patch_output, class_output)
    d_patch, _ = discriminator(x)
    d_score = d_patch.mean()

    # For generator/VAE training: we want to maximize D(recon)
    # Using BCE with logits: we want D(recon) to be close to 1 (real)
    loss = F.binary_cross_entropy_with_logits(
        d_patch,
        torch.ones_like(d_patch)
    ).mean()

    return loss, d_score
