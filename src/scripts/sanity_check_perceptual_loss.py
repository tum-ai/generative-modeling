"""
Sanity check script for FaceNet perceptual loss.

This script:
1. Loads raw CelebA images
2. Loads a few trained VAE checkpoints
3. Computes reconstructions
4. Evaluates perceptual loss on real vs reconstructed images
5. Provides recommendations for perceptual loss weight
"""

import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from generative_modeling.variational.celeba_hierarchical_vae import CelebAHierarchicalVAE
from generative_modeling.variational.celeba_beta_vae import CelebABetaVAE
from generative_modeling.losses import get_facenet_model, perceptual_loss


def load_model(checkpoint_path, model_type='hierarchical', device='cuda'):
    """Load a trained VAE model from checkpoint."""
    if model_type == 'hierarchical':
        model = CelebAHierarchicalVAE(
            image_size=128,
            latent_dims=(128, 256, 512),
            hidden_dims=[32, 64, 128, 256, 512]
        )
    elif model_type == 'beta':
        model = CelebABetaVAE(
            latent_dim=128,
            image_size=128,
            hidden_dims=[32, 64, 128, 256, 512]
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def get_dataloader(batch_size=16, num_samples=100):
    """Get a small dataloader for sanity checking."""
    celeb_transform = transforms.Compose([
        transforms.Resize(128, antialias=True),
        transforms.CenterCrop(128),
        transforms.ToTensor()
    ])
    
    dataset = CelebA('./data/', transform=celeb_transform, download=False, split='valid')
    
    # Use only a subset for sanity check
    subset = torch.utils.data.Subset(dataset, range(num_samples))
    
    loader = DataLoader(
        dataset=subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return loader


def compute_perceptual_statistics(model, dataloader, facenet, device='cuda'):
    """Compute perceptual loss statistics for a model."""
    perceptual_losses = []
    mse_losses = []
    
    with torch.no_grad():
        for data, _ in tqdm(dataloader, desc="Computing perceptual losses"):
            data = data.to(device)
            
            # Get reconstructions
            if hasattr(model, 'latent_dims'):  # Hierarchical VAE
                recon, _, _ = model(data)
            else:  # Beta VAE
                recon, _, _ = model(data)
            
            # Compute perceptual loss
            perc_loss, _ = perceptual_loss(recon, data, facenet)
            mse_loss = F.mse_loss(recon, data)
            
            perceptual_losses.append(perc_loss.item())
            mse_losses.append(mse_loss.item())
    
    return {
        'perceptual_mean': np.mean(perceptual_losses),
        'perceptual_std': np.std(perceptual_losses),
        'perceptual_median': np.median(perceptual_losses),
        'mse_mean': np.mean(mse_losses),
        'mse_std': np.std(mse_losses),
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load FaceNet
    print("\n" + "="*60)
    print("Loading FaceNet for perceptual loss...")
    print("="*60)
    facenet = get_facenet_model(pretrained='vggface2', use_gpu=(device == 'cuda'))
    facenet = facenet.to(device)
    print("FaceNet loaded successfully!")
    
    # Get dataloader
    print("\n" + "="*60)
    print("Loading CelebA validation data...")
    print("="*60)
    dataloader = get_dataloader(batch_size=16, num_samples=100)
    print(f"Loaded {len(dataloader.dataset)} samples")
    
    # Define checkpoints to test
    checkpoints = [
        {
            'path': 'out/checkpoints/celeba-hvae-kl0.000001-sobel0.0-adv0.0-ep40/hvae_model_epoch_10.pth',
            'type': 'hierarchical',
            'name': 'Hierarchical VAE (epoch 10)'
        },
        {
            'path': 'out/checkpoints/celeba-hvae-kl0.000001-sobel0.0-adv0.0-ep40/hvae_model_epoch_20.pth',
            'type': 'hierarchical',
            'name': 'Hierarchical VAE (epoch 20)'
        },
        {
            'path': 'out/checkpoints/celeba-hvae-kl0.000001-sobel0.0-adv0.0-ep40/hvae_model_epoch_40.pth',
            'type': 'hierarchical',
            'name': 'Hierarchical VAE (epoch 40)'
        },
        {
            'path': 'out/checkpoints/celeba-vae-kl0.001-sobel0.1-adv0.001/vae_model_epoch_5.pth',
            'type': 'beta',
            'name': 'Beta VAE (epoch 5)'
        },
        {
            'path': 'out/checkpoints/celeba-vae-kl0.001-sobel0.1-adv0.001/vae_model_epoch_10.pth',
            'type': 'beta',
            'name': 'Beta VAE (epoch 10)'
        },
    ]
    
    # If no checkpoints provided, test with random noise
    if not checkpoints:
        print("\n" + "="*60)
        print("No checkpoints provided. Testing with random noise...")
        print("="*60)
        
        # Create a simple model for testing
        model = CelebAHierarchicalVAE(
            image_size=128,
            latent_dims=(128, 256, 512),
            hidden_dims=[32, 64, 128, 256, 512]
        ).to(device)
        model.eval()
        
        # Test with random noise
        with torch.no_grad():
            data = next(iter(dataloader))[0].to(device)
            recon, _, _ = model(data)
            
            perc_loss, _ = perceptual_loss(recon, data, facenet)
            mse_loss = F.mse_loss(recon, data)
            
            print(f"\nRandom initialization:")
            print(f"  Perceptual loss: {perc_loss.item():.6f}")
            print(f"  MSE loss: {mse_loss.item():.6f}")
            
            # Save some samples
            os.makedirs('out/sanity_check_perceptual', exist_ok=True)
            comparison = torch.cat([data[:8], recon[:8]])
            save_image(comparison, 'out/sanity_check_perceptual/random_init.png', nrow=8)
            print(f"  Saved samples to out/sanity_check_perceptual/random_init.png")
    else:
        # Test each checkpoint
        results = []
        
        for ckpt in checkpoints:
            print("\n" + "="*60)
            print(f"Testing: {ckpt['name']}")
            print(f"Checkpoint: {ckpt['path']}")
            print("="*60)
            
            if not os.path.exists(ckpt['path']):
                print(f"WARNING: Checkpoint not found: {ckpt['path']}")
                continue
            
            # Load model
            model = load_model(ckpt['path'], ckpt['type'], device)
            
            # Compute statistics
            stats = compute_perceptual_statistics(model, dataloader, facenet, device)
            
            print(f"\nResults for {ckpt['name']}:")
            print(f"  Perceptual loss:")
            print(f"    Mean:   {stats['perceptual_mean']:.6f}")
            print(f"    Std:    {stats['perceptual_std']:.6f}")
            print(f"    Median: {stats['perceptual_median']:.6f}")
            print(f"  MSE loss:")
            print(f"    Mean:   {stats['mse_mean']:.6f}")
            print(f"    Std:    {stats['mse_std']:.6f}")
            
            results.append({
                'name': ckpt['name'],
                **stats
            })
            
            # Save some reconstructions
            with torch.no_grad():
                data = next(iter(dataloader))[0].to(device)
                if hasattr(model, 'latent_dims'):
                    recon, _, _ = model(data)
                else:
                    recon, _, _ = model(data)
                
                os.makedirs('out/sanity_check_perceptual', exist_ok=True)
                comparison = torch.cat([data[:8], recon[:8]])
                save_image(comparison, f"out/sanity_check_perceptual/{ckpt['name'].replace(' ', '_')}.png", nrow=8)
        
        # Print summary and recommendations
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        if results:
            print("\nPerceptual Loss Statistics:")
            print(f"{'Model':<40} {'Mean':>10} {'Std':>10} {'Median':>10}")
            print("-" * 70)
            for r in results:
                print(f"{r['name']:<40} {r['perceptual_mean']:>10.6f} {r['perceptual_std']:>10.6f} {r['perceptual_median']:>10.6f}")
            
            # Compute recommended weight
            # Strategy: weight should be such that perceptual loss contributes ~10-20% of total loss
            # Assuming MSE is around 0.01-0.05 for decent reconstructions
            avg_mse = np.mean([r['mse_mean'] for r in results])
            avg_perc = np.mean([r['perceptual_mean'] for r in results])
            
            # Target: perceptual contribution = 15% of total loss
            # perc_weight * avg_perc = 0.15 * (mse_weight * avg_mse + perc_weight * avg_perc)
            # Solving: perc_weight = 0.15 * mse_weight * avg_mse / (0.85 * avg_perc)
            
            mse_weight = 1.0  # Default MSE weight
            recommended_weight = 0.15 * mse_weight * avg_mse / (0.85 * avg_perc)
            
            print("\n" + "="*60)
            print("RECOMMENDATION")
            print("="*60)
            print(f"\nAverage MSE loss: {avg_mse:.6f}")
            print(f"Average perceptual loss: {avg_perc:.6f}")
            print(f"\nRecommended perceptual loss weight: {recommended_weight:.6f}")
            print(f"\nRationale:")
            print(f"  - We want perceptual loss to contribute ~15% of total reconstruction loss")
            print(f"  - With MSE weight = {mse_weight}, this gives perceptual weight ≈ {recommended_weight:.6f}")
            print(f"  - This balances pixel-level accuracy (MSE) with perceptual quality")
            print(f"\nSuggested range: {recommended_weight/2:.6f} to {recommended_weight*2:.6f}")
            print(f"  - Lower end: {recommended_weight/2:.6f} (more emphasis on MSE)")
            print(f"  - Higher end: {recommended_weight*2:.6f} (more emphasis on perceptual quality)")
        else:
            print("\nNo valid checkpoints found to analyze.")
            print("\nRECOMMENDATION:")
            print("  - Train a model first, then run this script with checkpoints")
            print("  - Based on typical VAE behavior, a reasonable starting point is:")
            print("    perceptual_weight = 0.01 to 0.1")
            print("  - Adjust based on visual quality of reconstructions")


if __name__ == "__main__":
    main()