"""
Perceptual loss using FaceNet (Inception-ResNet V1).

Uses the facenet-pytorch library which provides a PyTorch implementation of
the FaceNet Inception-ResNet (V1) model pretrained on VGGFace2 (and CASIA-Webface).
This model is commonly used for face feature extraction.

The perceptual loss computes L2 distance between 512-dimensional face embeddings
of real and reconstructed images, encouraging the VAE to preserve perceptual
face features rather than just pixel-level similarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FaceNetPerceptualLoss(nn.Module):
    """
    Perceptual loss using FaceNet (Inception-ResNet V1).
    
    Uses a pretrained FaceNet model to extract 512-dimensional face embeddings,
    then computes L2 distance between embeddings of real and reconstructed images.
    
    Args:
        pretrained: Which pretrained weights to use ('vggface2' or 'casia-webface')
        use_gpu: Whether to use GPU if available
        normalize_embeddings: Whether to L2-normalize embeddings before computing loss
    """
    
    def __init__(self, pretrained='vggface2', use_gpu=True, normalize_embeddings=True):
        super().__init__()
        self.pretrained = pretrained
        self.use_gpu = use_gpu
        self.normalize_embeddings = normalize_embeddings
        self.facenet = None
        self._loaded = False
    
    def _ensure_loaded(self, device):
        """Lazy load FaceNet on first use."""
        if self._loaded:
            return
        
        try:
            from facenet_pytorch import InceptionResnetV1
        except ImportError:
            raise ImportError(
                "facenet-pytorch is required for perceptual loss. "
                "Install it with: pip install facenet-pytorch"
            )
        
        print(f"Loading FaceNet (InceptionResnetV1) pretrained on {self.pretrained}...")
        self.facenet = InceptionResnetV1(pretrained=self.pretrained).eval()
        
        # Freeze FaceNet weights
        for param in self.facenet.parameters():
            param.requires_grad = False
        
        # Move to correct device
        self.facenet = self.facenet.to(device)
        self._loaded = True
        print("FaceNet loaded successfully!")
    
    def _preprocess(self, x):
        """
        Preprocess images for FaceNet.
        
        FaceNet expects images in [0, 1] range (same as our VAEs).
        It handles 160x160 input by default, but can work with other sizes.
        """
        # FaceNet expects [0, 1] range, which is what our VAEs output
        # No normalization needed - FaceNet handles it internally
        
        # FaceNet works best with 160x160, but can handle other sizes
        # Resize to 160x160 for optimal performance
        if x.shape[2] != 160 or x.shape[3] != 160:
            x = F.interpolate(x, size=(160, 160), mode='bilinear', align_corners=False)
        
        return x
    
    def get_embeddings(self, x):
        """
        Get FaceNet embeddings for images.
        
        Args:
            x: Images (B, 3, H, W) in [0, 1] range
            
        Returns:
            embeddings: 512-dimensional face embeddings (B, 512)
        """
        self._ensure_loaded(x.device)
        
        x = self._preprocess(x)
        
        with torch.no_grad():
            embeddings = self.facenet(x)
            
            # Optionally normalize embeddings
            if self.normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def forward(self, recon_x, target_x):
        """
        Compute perceptual loss between reconstructed and target images.
        
        Args:
            recon_x: Reconstructed images (B, 3, H, W) in [0, 1]
            target_x: Original images (B, 3, H, W) in [0, 1]
            
        Returns:
            loss: Scalar perceptual loss (L2 distance between embeddings)
            embedding_distance: Mean L2 distance between embeddings
        """
        # Get embeddings
        feat_recon = self.get_embeddings(recon_x)
        feat_target = self.get_embeddings(target_x)
        
        # Compute L2 distance between embeddings
        embedding_distance = F.mse_loss(feat_recon, feat_target, reduction='mean')
        
        return embedding_distance, embedding_distance


def get_facenet_model(pretrained='vggface2', use_gpu=True):
    """
    Convenience function to load just the FaceNet model.
    
    Returns the raw FaceNet network for direct use.
    """
    try:
        from facenet_pytorch import InceptionResnetV1
    except ImportError:
        raise ImportError(
            "facenet-pytorch is required for perceptual loss. "
            "Install it with: pip install facenet-pytorch"
        )
    
    facenet = InceptionResnetV1(pretrained=pretrained).eval()
    
    for param in facenet.parameters():
        param.requires_grad = False
    
    if use_gpu and torch.cuda.is_available():
        facenet = facenet.cuda()
    
    return facenet


def perceptual_loss(recon_x, target_x, facenet, normalize_embeddings=True):
    """
    Simple functional interface for perceptual loss.
    
    Args:
        recon_x: Reconstructed images in [0, 1]
        target_x: Original images in [0, 1]
        facenet: FaceNet model
        normalize_embeddings: Whether to L2-normalize embeddings
        
    Returns:
        loss: Perceptual loss (L2 distance between embeddings)
        embedding_distance: Mean L2 distance between embeddings
    """
    # Resize to 160x160 for FaceNet
    if recon_x.shape[2] != 160 or recon_x.shape[3] != 160:
        recon_x = F.interpolate(recon_x, size=(160, 160), mode='bilinear', align_corners=False)
        target_x = F.interpolate(target_x, size=(160, 160), mode='bilinear', align_corners=False)
    
    # Get embeddings
    with torch.no_grad():
        feat_recon = facenet(recon_x)
        feat_target = facenet(target_x)
        
        if normalize_embeddings:
            feat_recon = F.normalize(feat_recon, p=2, dim=1)
            feat_target = F.normalize(feat_target, p=2, dim=1)
    
    # Compute L2 distance
    embedding_distance = F.mse_loss(feat_recon, feat_target, reduction='mean')
    
    return embedding_distance, embedding_distance