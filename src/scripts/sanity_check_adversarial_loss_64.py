"""
Sanity check for adversarial loss using 64x64 PGAN discriminator.

This script:
1. Loads raw CelebA images and computes discriminator scores at 64x64
2. Loads VAE checkpoints and computes scores on reconstructions (downscaled to 64x64)
3. Computes the adversarial loss values to suggest appropriate weights
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

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from generative_modeling.variational.celeba_beta_vae import CelebABetaVAE
from generative_modeling.variational.celeba_hierarchical_vae import CelebAHierarchicalVAE
from generative_modeling.losses import get_pgan_discriminator, adversarial_loss


# Configuration
CONFIG = {
    "image_size": 128,  # VAE works at 128x128
    "discriminator_scale": 4,  # 64x64 discriminator
    "batch_size": 64,
    "celeb_path": "./data/",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir": "out/sanity_check_adv_64/",
    "num_batches": 5,  # Number of batches to evaluate
}

# VAE checkpoints to evaluate
BETA_VAE_CHECKPOINTS = [
    ("out/checkpoints/celeba-vae-kl0.001-sobel1.0/vae_model_epoch_5.pth", "beta_vae_kl0.001_sobel1.0_ep5"),
    ("out/checkpoints/celeba-vae-kl0.001-sobel1.0/vae_model_epoch_10.pth", "beta_vae_kl0.001_sobel1.0_ep10"),
    ("out/checkpoints/celeba-vae-kl0.001-sobel1.0/vae_model_epoch_20.pth", "beta_vae_kl0.001_sobel1.0_ep20"),
]

HVAE_CHECKPOINTS = [
    ("out/checkpoints/celeba-hvae-kl0.000001-sobel0.25-z1-128-ep40/hvae_model_epoch_5.pth", "hvae_kl1e-6_sobel0.25_ep5"),
]


def get_dataloader(config):
    celeb_transform = transforms.Compose([
        transforms.Resize(config["image_size"], antialias=True),
        transforms.CenterCrop(config["image_size"]),
        transforms.ToTensor()
    ])

    dataset = CelebA(config["celeb_path"], transform=celeb_transform, download=False, split='valid')
    loader = DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    return loader


def preprocess_for_discriminator(x, scale):
    """Convert from [0,1] to [-1,1] and resize for PGAN discriminator."""
    x = x * 2.0 - 1.0
    target_size = 4 * (2 ** scale)
    if x.shape[2] != target_size or x.shape[3] != target_size:
        x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)
    return x


def evaluate_discriminator_on_images(discriminator, images, name="", scale=4):
    """Evaluate discriminator scores on a batch of images."""
    with torch.no_grad():
        # Preprocess: [0,1] -> [-1,1] and resize
        images_prep = preprocess_for_discriminator(images, scale)
        scores = discriminator(images_prep).squeeze()

        # Compute adversarial loss (what we'd use in training)
        adv_loss = F.softplus(-scores).mean()

    return {
        "name": name,
        "mean_score": scores.mean().item(),
        "std_score": scores.std().item(),
        "min_score": scores.min().item(),
        "max_score": scores.max().item(),
        "adv_loss": adv_loss.item(),
    }


def load_beta_vae(checkpoint_path, config):
    """Load a beta VAE model from checkpoint."""
    model = CelebABetaVAE(
        latent_dim=128,
        image_size=config["image_size"],
        hidden_dims=[32, 64, 128, 256, 512]
    ).to(config["device"])

    state_dict = torch.load(checkpoint_path, map_location=config["device"])
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_hvae(checkpoint_path, config):
    """Load a hierarchical VAE model from checkpoint."""
    model = CelebAHierarchicalVAE(
        image_size=config["image_size"],
        latent_dims=(128, 256, 512),
        hidden_dims=[32, 64, 128, 256, 512]
    ).to(config["device"])

    state_dict = torch.load(checkpoint_path, map_location=config["device"])
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    print("=" * 60)
    print("Adversarial Loss Sanity Check (64x64 PGAN)")
    print("=" * 60)

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # Load discriminator
    print(f"\n[1] Loading PGAN discriminator (scale {CONFIG['discriminator_scale']} = 64x64)...")
    discriminator = get_pgan_discriminator(
        use_gpu=(CONFIG["device"] == "cuda"),
        scale=CONFIG["discriminator_scale"]
    )
    discriminator = discriminator.to(CONFIG["device"])
    print("    Discriminator loaded!")

    # Load data
    print("\n[2] Loading CelebA validation data...")
    dataloader = get_dataloader(CONFIG)
    print(f"    Loaded {len(dataloader)} batches")

    # Collect results
    all_results = []

    # Evaluate on real images
    print("\n[3] Evaluating on real CelebA images (downscaled to 64x64)...")
    real_scores = []
    real_adv_losses = []

    sample_real_batch = None
    for batch_idx, (images, _) in enumerate(dataloader):
        if batch_idx >= CONFIG["num_batches"]:
            break

        images = images.to(CONFIG["device"])
        if sample_real_batch is None:
            sample_real_batch = images[:8].clone()

        result = evaluate_discriminator_on_images(
            discriminator, images, f"real_batch_{batch_idx}", CONFIG["discriminator_scale"]
        )
        real_scores.append(result["mean_score"])
        real_adv_losses.append(result["adv_loss"])
        print(f"    Batch {batch_idx}: score={result['mean_score']:.4f}, adv_loss={result['adv_loss']:.4f}")

    real_result = {
        "name": "Real CelebA (64x64)",
        "mean_score": np.mean(real_scores),
        "std_score": np.std(real_scores),
        "adv_loss": np.mean(real_adv_losses),
    }
    all_results.append(real_result)
    print(f"\n    Overall Real: score={real_result['mean_score']:.4f} ± {real_result['std_score']:.4f}, adv_loss={real_result['adv_loss']:.4f}")

    # Evaluate on VAE reconstructions
    print("\n[4] Evaluating on VAE reconstructions (downscaled to 64x64)...")

    # Get a fixed batch for reconstruction comparison
    fixed_batch, _ = next(iter(dataloader))
    fixed_batch = fixed_batch.to(CONFIG["device"])

    # Beta VAE checkpoints
    for checkpoint_path, name in BETA_VAE_CHECKPOINTS:
        if not os.path.exists(checkpoint_path):
            print(f"    Skipping {name}: checkpoint not found at {checkpoint_path}")
            continue

        print(f"\n    Loading {name}...")
        model = load_beta_vae(checkpoint_path, CONFIG)

        recon_scores = []
        recon_adv_losses = []

        for batch_idx, (images, _) in enumerate(dataloader):
            if batch_idx >= CONFIG["num_batches"]:
                break

            images = images.to(CONFIG["device"])
            with torch.no_grad():
                recon, _, _ = model(images)

            result = evaluate_discriminator_on_images(
                discriminator, recon, f"{name}_batch_{batch_idx}", CONFIG["discriminator_scale"]
            )
            recon_scores.append(result["mean_score"])
            recon_adv_losses.append(result["adv_loss"])

        recon_result = {
            "name": name,
            "mean_score": np.mean(recon_scores),
            "std_score": np.std(recon_scores),
            "adv_loss": np.mean(recon_adv_losses),
        }
        all_results.append(recon_result)
        print(f"    {name}: score={recon_result['mean_score']:.4f} ± {recon_result['std_score']:.4f}, adv_loss={recon_result['adv_loss']:.4f}")

        # Save sample reconstructions
        with torch.no_grad():
            sample_recon, _, _ = model(sample_real_batch)
        comparison = torch.cat([sample_real_batch, sample_recon], dim=0)
        save_image(comparison, os.path.join(CONFIG["output_dir"], f"{name}_recon.png"), nrow=8)

        del model
        torch.cuda.empty_cache()

    # HVAE checkpoints
    for checkpoint_path, name in HVAE_CHECKPOINTS:
        if not os.path.exists(checkpoint_path):
            print(f"    Skipping {name}: checkpoint not found at {checkpoint_path}")
            continue

        print(f"\n    Loading {name}...")
        model = load_hvae(checkpoint_path, CONFIG)

        recon_scores = []
        recon_adv_losses = []

        for batch_idx, (images, _) in enumerate(dataloader):
            if batch_idx >= CONFIG["num_batches"]:
                break

            images = images.to(CONFIG["device"])
            with torch.no_grad():
                recon, _, _ = model(images)

            result = evaluate_discriminator_on_images(
                discriminator, recon, f"{name}_batch_{batch_idx}", CONFIG["discriminator_scale"]
            )
            recon_scores.append(result["mean_score"])
            recon_adv_losses.append(result["adv_loss"])

        recon_result = {
            "name": name,
            "mean_score": np.mean(recon_scores),
            "std_score": np.std(recon_scores),
            "adv_loss": np.mean(recon_adv_losses),
        }
        all_results.append(recon_result)
        print(f"    {name}: score={recon_result['mean_score']:.4f} ± {recon_result['std_score']:.4f}, adv_loss={recon_result['adv_loss']:.4f}")

        # Save sample reconstructions
        with torch.no_grad():
            sample_recon, _, _ = model(sample_real_batch)
        comparison = torch.cat([sample_real_batch, sample_recon], dim=0)
        save_image(comparison, os.path.join(CONFIG["output_dir"], f"{name}_recon.png"), nrow=8)

        del model
        torch.cuda.empty_cache()

    # Evaluate on random noise (baseline - should have very negative scores)
    print("\n[5] Evaluating on random noise (baseline, downscaled to 64x64)...")
    noise = torch.rand(CONFIG["batch_size"], 3, 128, 128).to(CONFIG["device"])
    noise_result = evaluate_discriminator_on_images(
        discriminator, noise, "random_noise", CONFIG["discriminator_scale"]
    )
    noise_result["name"] = "Random Noise (64x64)"
    all_results.append(noise_result)
    print(f"    Random Noise: score={noise_result['mean_score']:.4f}, adv_loss={noise_result['adv_loss']:.4f}")

    # Summary and weight recommendation
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\n{:<40} {:>12} {:>12}".format("Source", "D Score", "Adv Loss"))
    print("-" * 64)
    for r in all_results:
        print("{:<40} {:>12.4f} {:>12.4f}".format(r['name'], r['mean_score'], r['adv_loss']))

    # Weight recommendation
    print("\n" + "=" * 60)
    print("WEIGHT RECOMMENDATION")
    print("=" * 60)

    # Get reconstruction loss scale from existing training
    # Typical MSE loss is ~0.01-0.05, sobel loss is similar
    # Total weighted recon loss is typically ~0.05-0.1
    # KL loss is ~100-1000 before weighting (weighted by ~0.001 -> ~0.1)

    real_adv = all_results[0]['adv_loss']

    # Find worst reconstruction adv loss
    recon_adv_losses = [r['adv_loss'] for r in all_results[1:-1] if r['adv_loss'] > 0]
    if recon_adv_losses:
        worst_recon_adv = max(recon_adv_losses)
        best_recon_adv = min(recon_adv_losses)
    else:
        worst_recon_adv = 10.0
        best_recon_adv = 9.0

    print(f"\nReal images adv loss:     {real_adv:.4f}")
    print(f"Best recon adv loss:      {best_recon_adv:.4f}")
    print(f"Worst recon adv loss:     {worst_recon_adv:.4f}")

    # For VAE training, we want:
    # - Reconstruction loss (MSE + Sobel) to be primary (~0.05-0.1 total)
    # - KL loss to be weighted down significantly (~0.001 weight gives ~0.1 contribution)
    # - Adversarial loss to be a regularizer, not dominant

    # If typical recon loss is ~0.05 and adv loss is ~10.0,
    # we want adv_weight such that adv_weight * 10 ≈ 0.005-0.01 (10-20% of recon)
    # So adv_weight ≈ 0.0005-0.001

    typical_recon = 0.05
    target_adv_contribution = 0.01  # ~20% of typical recon
    suggested_weight = target_adv_contribution / worst_recon_adv

    print(f"\nTypical reconstruction loss: ~{typical_recon}")
    print(f"Target adversarial contribution: ~{target_adv_contribution} (20% of recon)")
    print(f"\nSuggested adversarial weight: {suggested_weight:.4f}")
    print("(This keeps adversarial loss as ~10-20% of reconstruction loss)")
    print("\nRationale:")
    print("- Too high: VAE focuses on fooling discriminator, ignores pixel accuracy")
    print("- Too low: No perceptual improvement from discriminator")
    print("- Sweet spot: Improves sharpness without sacrificing reconstruction")

    print("\nAlternative weights to try:")
    print("  - 0.0005 (conservative)")
    print("  - 0.001 (balanced - RECOMMENDED)")
    print("  - 0.002 (aggressive)")
    print("  - 0.005 (very aggressive)")

    # Check if scores are meaningful
    print("\n" + "=" * 60)
    print("SCORE MEANINGFULNESS CHECK")
    print("=" * 60)

    score_gap = real_result['mean_score'] - noise_result['mean_score']
    recon_gap = real_result['mean_score'] - min([r['mean_score'] for r in all_results[1:-1]])

    print(f"\nReal vs Noise score gap: {score_gap:.4f}")
    print(f"Real vs Best Recon score gap: {recon_gap:.4f}")

    if score_gap > 2.0:
        print("✓ Discriminator clearly distinguishes real from noise")
    else:
        print("✗ Discriminator may not be distinguishing real from noise well")

    if recon_gap > 0.5:
        print("✓ Discriminator distinguishes real from reconstructions")
    else:
        print("✗ Discriminator may not be distinguishing real from reconstructions well")

    if score_gap > 2.0 and recon_gap > 0.5:
        print("\n✓✓ Scores appear meaningful - discriminator is working well!")
    else:
        print("\n✗✗ Scores may not be meaningful - consider using a different scale or loss")


if __name__ == "__main__":
    main()
