"""
Sanity check for adversarial loss using StarGAN discriminator.

This script:
1. Loads raw CelebA images and computes discriminator scores
2. Loads VAE checkpoints and computes scores on reconstructions
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
from generative_modeling.losses import get_stargan_discriminator, stargan_adversarial_loss


# Configuration
CONFIG = {
    "image_size": 128,
    "batch_size": 64,
    "celeb_path": "./data/",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir": "out/sanity_check_stargan/",
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


def evaluate_discriminator_on_images(discriminator, images, name="", target_images=None):
    """Evaluate discriminator scores on a batch of images."""
    with torch.no_grad():
        # StarGAN returns (patch_output, class_output)
        d_patch, _ = discriminator(images)
        d_score = d_patch.mean()

        # Compute adversarial loss (what we'd use in training)
        # For generator/VAE: we want D(recon) to be close to 1 (real)
        adv_loss = F.binary_cross_entropy_with_logits(
            d_patch,
            torch.ones_like(d_patch)
        ).mean()

        # Compute MSE if target images provided
        mse = None
        if target_images is not None:
            mse = F.mse_loss(images, target_images).item()

    result = {
        "name": name,
        "mean_score": d_score.item(),
        "adv_loss": adv_loss.item(),
    }
    if mse is not None:
        result["mse"] = mse

    return result


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
    print("Adversarial Loss Sanity Check (StarGAN)")
    print("=" * 60)

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # Load discriminator
    print("\n[1] Loading StarGAN discriminator...")
    discriminator = get_stargan_discriminator(
        use_gpu=(CONFIG["device"] == "cuda"),
        img_size=CONFIG["image_size"]
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
    print("\n[3] Evaluating on real CelebA images...")
    real_scores = []
    real_adv_losses = []

    sample_real_batch = None
    for batch_idx, (images, _) in enumerate(dataloader):
        if batch_idx >= CONFIG["num_batches"]:
            break

        images = images.to(CONFIG["device"])
        if sample_real_batch is None:
            sample_real_batch = images[:8].clone()

        result = evaluate_discriminator_on_images(discriminator, images, f"real_batch_{batch_idx}")
        real_scores.append(result["mean_score"])
        real_adv_losses.append(result["adv_loss"])
        print(f"    Batch {batch_idx}: score={result['mean_score']:.4f}, adv_loss={result['adv_loss']:.4f}")

    real_result = {
        "name": "Real CelebA",
        "mean_score": np.mean(real_scores),
        "std_score": np.std(real_scores),
        "adv_loss": np.mean(real_adv_losses),
    }
    all_results.append(real_result)
    print(f"\n    Overall Real: score={real_result['mean_score']:.4f} ± {real_result['std_score']:.4f}, adv_loss={real_result['adv_loss']:.4f}")

    # Evaluate on VAE reconstructions
    print("\n[4] Evaluating on VAE reconstructions...")

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
        recon_mses = []

        for batch_idx, (images, _) in enumerate(dataloader):
            if batch_idx >= CONFIG["num_batches"]:
                break

            images = images.to(CONFIG["device"])
            with torch.no_grad():
                recon, _, _ = model(images)

            result = evaluate_discriminator_on_images(discriminator, recon, f"{name}_batch_{batch_idx}", target_images=images)
            recon_scores.append(result["mean_score"])
            recon_adv_losses.append(result["adv_loss"])
            if "mse" in result:
                recon_mses.append(result["mse"])

        recon_result = {
            "name": name,
            "mean_score": np.mean(recon_scores),
            "std_score": np.std(recon_scores),
            "adv_loss": np.mean(recon_adv_losses),
        }
        if recon_mses:
            recon_result["mse"] = np.mean(recon_mses)
        all_results.append(recon_result)
        print(f"    {name}: score={recon_result['mean_score']:.4f} ± {recon_result['std_score']:.4f}, adv_loss={recon_result['adv_loss']:.4f}" + (f", mse={recon_result['mse']:.4f}" if "mse" in recon_result else ""))

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
        recon_mses = []

        for batch_idx, (images, _) in enumerate(dataloader):
            if batch_idx >= CONFIG["num_batches"]:
                break

            images = images.to(CONFIG["device"])
            with torch.no_grad():
                recon, _, _ = model(images)

            result = evaluate_discriminator_on_images(discriminator, recon, f"{name}_batch_{batch_idx}", target_images=images)
            recon_scores.append(result["mean_score"])
            recon_adv_losses.append(result["adv_loss"])
            if "mse" in result:
                recon_mses.append(result["mse"])

        recon_result = {
            "name": name,
            "mean_score": np.mean(recon_scores),
            "std_score": np.std(recon_scores),
            "adv_loss": np.mean(recon_adv_losses),
        }
        if recon_mses:
            recon_result["mse"] = np.mean(recon_mses)
        all_results.append(recon_result)
        print(f"    {name}: score={recon_result['mean_score']:.4f} ± {recon_result['std_score']:.4f}, adv_loss={recon_result['adv_loss']:.4f}" + (f", mse={recon_result['mse']:.4f}" if "mse" in recon_result else ""))

        # Save sample reconstructions
        with torch.no_grad():
            sample_recon, _, _ = model(sample_real_batch)
        comparison = torch.cat([sample_real_batch, sample_recon], dim=0)
        save_image(comparison, os.path.join(CONFIG["output_dir"], f"{name}_recon.png"), nrow=8)

        del model
        torch.cuda.empty_cache()

    # Evaluate on random noise (baseline - should have very negative scores)
    print("\n[5] Evaluating on random noise (baseline)...")
    noise = torch.rand(CONFIG["batch_size"], 3, 128, 128).to(CONFIG["device"])
    noise_result = evaluate_discriminator_on_images(discriminator, noise, "random_noise")
    noise_result["name"] = "Random Noise"
    all_results.append(noise_result)
    print(f"    Random Noise: score={noise_result['mean_score']:.4f}, adv_loss={noise_result['adv_loss']:.4f}")

    # Summary and weight recommendation
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\n{:<40} {:>12} {:>12} {:>12}".format("Source", "D Score", "Adv Loss", "MSE"))
    print("-" * 76)
    for r in all_results:
        mse_str = f"{r['mse']:.4f}" if 'mse' in r else "N/A"
        print("{:<40} {:>12.4f} {:>12.4f} {:>12}".format(r['name'], r['mean_score'], r['adv_loss'], mse_str))

    # Weight recommendation
    print("\n" + "=" * 60)
    print("WEIGHT RECOMMENDATION")
    print("=" * 60)

    # Get reconstruction loss scale from existing training
    # Typical MSE loss is ~0.01-0.05, sobel loss is similar
    # Total weighted recon loss is typically ~0.05-0.1
    # KL loss is ~100-1000 before weighting (weighted by ~0.001 -> ~0.1)

    real_adv = all_results[0]['adv_loss']

    # Find worst reconstruction adv loss and corresponding MSE
    recon_adv_losses = [r['adv_loss'] for r in all_results[1:-1] if r['adv_loss'] > 0]
    recon_mses = [r['mse'] for r in all_results[1:-1] if 'mse' in r]

    if recon_adv_losses:
        worst_recon_adv = max(recon_adv_losses)
        best_recon_adv = min(recon_adv_losses)
    else:
        worst_recon_adv = 10.0
        best_recon_adv = 9.0

    if recon_mses:
        typical_mse = np.mean(recon_mses)
    else:
        typical_mse = 0.05  # fallback to typical value

    print(f"\nReal images adv loss:     {real_adv:.4f}")
    print(f"Best recon adv loss:      {best_recon_adv:.4f}")
    print(f"Worst recon adv loss:     {worst_recon_adv:.4f}")
    print(f"Typical recon MSE:        {typical_mse:.4f}")

    # For VAE training, we want:
    # - Reconstruction loss (MSE + Sobel) to be primary (~0.05-0.1 total)
    # - KL loss to be weighted down significantly (~0.001 weight gives ~0.1 contribution)
    # - Adversarial loss to be a regularizer, not dominant

    # Calculate weight based on MSE * 0.2
    # If typical MSE is ~0.05, we want adv_weight * adv_loss ≈ 0.05 * 0.2 = 0.01
    target_adv_contribution = typical_mse * 0.2  # 20% of MSE
    suggested_weight = target_adv_contribution / worst_recon_adv

    print(f"\nTarget adversarial contribution: ~{target_adv_contribution:.4f} (20% of MSE)")
    print(f"\nSuggested adversarial weight: {suggested_weight:.6f}")
    print("(This keeps adversarial loss at ~20% of MSE loss)")
    print("\nRationale:")
    print("- Too high: VAE focuses on fooling discriminator, ignores pixel accuracy")
    print("- Too low: No perceptual improvement from discriminator")
    print("- Sweet spot: Improves sharpness without sacrificing reconstruction")

    print("\nAlternative weights to try:")
    print(f"  - {suggested_weight * 0.5:.6f} (conservative - 10% of MSE)")
    print(f"  - {suggested_weight:.6f} (balanced - 20% of MSE - RECOMMENDED)")
    print(f"  - {suggested_weight * 2:.6f} (aggressive - 40% of MSE)")
    print(f"  - {suggested_weight * 5:.6f} (very aggressive - 100% of MSE)")

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
        print("\n✗✗ Scores may not be meaningful - consider using a different discriminator")


if __name__ == "__main__":
    main()
