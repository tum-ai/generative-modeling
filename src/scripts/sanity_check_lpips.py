"""
Sanity check script for LPIPS loss.

This script:
1. Loads raw CelebA images
2. Loads a few trained VAE checkpoints
3. Computes reconstructions
4. Evaluates LPIPS loss on real vs reconstructed images
5. Provides recommendations for LPIPS loss weight
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
from generative_modeling.losses import get_lpips_model, lpips_loss


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


def compute_lpips_statistics(model, dataloader, lpips_model, device='cuda'):
    """Compute LPIPS loss statistics for a model."""
    lpips_losses = []
    mse_losses = []

    with torch.no_grad():
        for data, _ in tqdm(dataloader, desc="Computing LPIPS losses"):
            data = data.to(device)

            # Get reconstructions
            if hasattr(model, 'latent_dims'):  # Hierarchical VAE
                recon, _, _ = model(data)
            else:  # Beta VAE
                recon, _, _ = model(data)

            # Compute LPIPS loss
            lpips_loss_val, _ = lpips_loss(recon, data, lpips_model)
            mse_loss = F.mse_loss(recon, data)

            lpips_losses.append(lpips_loss_val.item())
            mse_losses.append(mse_loss.item())

    return {
        'lpips_mean': np.mean(lpips_losses),
        'lpips_std': np.std(lpips_losses),
        'lpips_median': np.median(lpips_losses),
        'mse_mean': np.mean(mse_losses),
        'mse_std': np.std(mse_losses),
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Get dataloader
    print("\n" + "="*60)
    print("Loading CelebA validation data...")
    print("="*60)
    dataloader = get_dataloader(batch_size=16, num_samples=100)
    print(f"Loaded {len(dataloader.dataset)} samples")

    # Load LPIPS model with alexnet backbone
    print("\n" + "="*60)
    print("Loading LPIPS model with alexnet backbone...")
    print("="*60)
    lpips_model = get_lpips_model(net='alex', use_gpu=(device == 'cuda'))
    lpips_model = lpips_model.to(device)
    print("LPIPS model loaded!")

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
        stats = compute_lpips_statistics(model, dataloader, lpips_model, device)

        print(f"\nResults for {ckpt['name']}:")
        print(f"  LPIPS loss:")
        print(f"    Mean:   {stats['lpips_mean']:.6f}")
        print(f"    Std:    {stats['lpips_std']:.6f}")
        print(f"    Median: {stats['lpips_median']:.6f}")
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

            os.makedirs('out/sanity_check_lpips', exist_ok=True)
            comparison = torch.cat([data[:8], recon[:8]])
            save_image(comparison, f"out/sanity_check_lpips/{ckpt['name'].replace(' ', '_')}.png", nrow=8)

    # Print summary and recommendations
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if results:
        print("\nLPIPS Loss Statistics:")
        print(f"{'Model':<40} {'Mean':>10} {'Std':>10} {'Median':>10}")
        print("-" * 70)
        for r in results:
            print(f"{r['name']:<40} {r['lpips_mean']:>10.6f} {r['lpips_std']:>10.6f} {r['lpips_median']:>10.6f}")

        # Compute recommended weight
        # Strategy: weight should be such that LPIPS loss contributes ~20% of total loss
        # Assuming MSE is around 0.01-0.05 for decent reconstructions
        avg_mse = np.mean([r['mse_mean'] for r in results])
        avg_lpips = np.mean([r['lpips_mean'] for r in results])

        # Target: LPIPS contribution = 20% of total loss
        # lpips_weight * avg_lpips = 0.20 * (mse_weight * avg_mse + lpips_weight * avg_lpips)
        # Solving: lpips_weight = 0.20 * mse_weight * avg_mse / (0.80 * avg_lpips)

        mse_weight = 1.0  # Default MSE weight
        recommended_weight = 0.20 * mse_weight * avg_mse / (0.80 * avg_lpips)

        print("\n" + "="*60)
        print("RECOMMENDATION")
        print("="*60)
        print(f"\nAverage MSE loss: {avg_mse:.6f}")
        print(f"Average LPIPS loss: {avg_lpips:.6f}")
        print(f"\nRecommended LPIPS loss weight: {recommended_weight:.6f}")
        print(f"\nRationale:")
        print(f"  - We want LPIPS loss to contribute ~20% of total reconstruction loss")
        print(f"  - With MSE weight = {mse_weight}, this gives LPIPS weight ≈ {recommended_weight:.6f}")
        print(f"  - This balances pixel-level accuracy (MSE) with perceptual quality")
        print(f"\nSuggested range: {recommended_weight/2:.6f} to {recommended_weight*2:.6f}")
        print(f"  - Lower end: {recommended_weight/2:.6f} (more emphasis on MSE)")
        print(f"  - Higher end: {recommended_weight*2:.6f} (more emphasis on perceptual quality)")
    else:
        print("\nNo valid checkpoints found to analyze.")
        print("\nRECOMMENDATION:")
        print("  - Train a model first, then run this script with checkpoints")
        print("  - Based on typical VAE behavior, a reasonable starting point is:")
        print("    lpips_weight = 0.01 to 0.1")
        print("  - Adjust based on visual quality of reconstructions")


if __name__ == "__main__":
    main()
