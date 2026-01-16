"""
LPIPS (Learned Perceptual Image Patch Similarity) loss.

LPIPS uses features from a pretrained CNN (typically VGG) to measure perceptual
similarity between images. It's particularly effective at capturing perceptual
differences that pixel-wise losses like MSE miss, making it ideal for improving
reconstruction quality of faces and their backgrounds.

Reference: Zhang et al. "The Unreasonable Effectiveness of Deep Features as a
Perceptual Metric", CVPR 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LPIPSLoss(nn.Module):
    """
    LPIPS (Learned Perceptual Image Patch Similarity) loss.
    
    Uses a pretrained CNN to extract features and computes perceptual similarity
    between images. This is particularly effective for improving reconstruction
    quality of faces and their backgrounds.
    
    Args:
        net: Network backbone to use ('vgg', 'alex', or 'squeeze')
            - 'vgg': VGG-like network (default, best quality)
            - 'alex': AlexNet (faster, slightly lower quality)
            - 'squeeze': SqueezeNet (fastest, lower quality)
        use_gpu: Whether to use GPU if available
        spatial: Whether to output spatial map (True) or average (False)
    """
    
    def __init__(self, net='vgg', use_gpu=True, spatial=False):
        super().__init__()
        self.net = net
        self.use_gpu = use_gpu
        self.spatial = spatial
        self.lpips_model = None
        self._loaded = False
    
    def _ensure_loaded(self, device):
        """Lazy load LPIPS model on first use."""
        if self._loaded:
            return
        
        try:
            import lpips
        except ImportError:
            raise ImportError(
                "lpips is required for LPIPS loss. "
                "Install it with: pip install lpips"
            )
        
        print(f"Loading LPIPS model with {self.net} backbone...")
        self.lpips_model = lpips.LPIPS(net=self.net, spatial=self.spatial)
        
        # Freeze LPIPS weights
        for param in self.lpips_model.parameters():
            param.requires_grad = False
        
        # Move to correct device
        self.lpips_model = self.lpips_model.to(device)
        self._loaded = True
        print(f"LPIPS model ({self.net} backbone) loaded successfully!")
    
    def _preprocess(self, x):
        """
        Preprocess images for LPIPS.
        
        LPIPS expects images in [-1, 1] range, but our VAEs output [0, 1].
        """
        # Convert from [0, 1] to [-1, 1]
        x = x * 2.0 - 1.0
        return x
    
    def forward(self, recon_x, target_x):
        """
        Compute LPIPS loss between reconstructed and target images.
        
        Args:
            recon_x: Reconstructed images (B, 3, H, W) in [0, 1]
            target_x: Original images (B, 3, H, W) in [0, 1]
            
        Returns:
            loss: Scalar LPIPS loss (perceptual distance)
            lpips_distance: Mean LPIPS distance
        """
        self._ensure_loaded(recon_x.device)
        
        # Preprocess to [-1, 1]
        recon_preprocessed = self._preprocess(recon_x)
        target_preprocessed = self._preprocess(target_x)
        
        # Compute LPIPS distance
        with torch.no_grad():
            lpips_distance = self.lpips_model(recon_preprocessed, target_preprocessed)
        
        # LPIPS returns (B, 1, H, W) if spatial=True, or (B, 1) if spatial=False
        if self.spatial:
            # Average over spatial dimensions and batch
            loss = lpips_distance.mean()
        else:
            # Average over batch
            loss = lpips_distance.mean()
        
        return loss, loss


def get_lpips_model(net='vgg', use_gpu=True, spatial=False):
    """
    Convenience function to load just the LPIPS model.
    
    Returns the raw LPIPS network for direct use.
    """
    try:
        import lpips
    except ImportError:
        raise ImportError(
            "lpips is required for LPIPS loss. "
            "Install it with: pip install lpips"
        )
    
    lpips_model = lpips.LPIPS(net=net, spatial=spatial)
    
    for param in lpips_model.parameters():
        param.requires_grad = False
    
    if use_gpu and torch.cuda.is_available():
        lpips_model = lpips_model.cuda()
    
    return lpips_model


def lpips_loss(recon_x, target_x, lpips_model):
    """
    Simple functional interface for LPIPS loss.
    
    Args:
        recon_x: Reconstructed images in [0, 1]
        target_x: Original images in [0, 1]
        lpips_model: LPIPS model
        
    Returns:
        loss: LPIPS loss (perceptual distance)
        lpips_distance: Mean LPIPS distance
    """
    # Preprocess to [-1, 1]
    recon_preprocessed = recon_x * 2.0 - 1.0
    target_preprocessed = target_x * 2.0 - 1.0
    
    # Compute LPIPS distance
    with torch.no_grad():
        lpips_distance = lpips_model(recon_preprocessed, target_preprocessed)
    
    # Average over batch
    loss = lpips_distance.mean()
    
    return loss, loss