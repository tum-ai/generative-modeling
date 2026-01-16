"""Compare model sizes across VAE variants."""
import torch
from generative_modeling.variational.celeba_beta_vae import CelebABetaVAE
from generative_modeling.variational.celeba_hierarchical_vae import CelebAHierarchicalVAE
from generative_modeling.variational.celeba_vqvae import CelebAVQVAE


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Beta-VAE
    beta_vae = CelebABetaVAE(
        latent_dim=128,
        image_size=128,
        hidden_dims=[32, 64, 128, 256, 512]
    ).to(device)
    beta_params = count_params(beta_vae)

    # Hierarchical VAE
    hvae = CelebAHierarchicalVAE(
        image_size=128,
        latent_dims=(128, 256, 512),
        hidden_dims=[32, 64, 128, 256, 512]
    ).to(device)
    hvae_params = count_params(hvae)

    # VQ-VAE
    vqvae = CelebAVQVAE(
        num_embeddings=512,
        embedding_dim=256,
        image_size=128,
        hidden_dims=[32, 64, 128, 256, 512]
    ).to(device)
    vqvae_params = count_params(vqvae)

    print("Model size comparison:")
    print(f"  Beta-VAE:        {beta_params:,} parameters")
    print(f"  Hierarchical VAE: {hvae_params:,} parameters")
    print(f"  VQ-VAE:          {vqvae_params:,} parameters")
    print()
    print(f"VQ-VAE is {vqvae_params / beta_params:.2f}x the size of Beta-VAE")
    print(f"VQ-VAE is {vqvae_params / hvae_params:.2f}x the size of Hierarchical VAE")


if __name__ == "__main__":
    main()
