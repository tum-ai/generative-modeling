"""Test VQ-VAE shapes and basic functionality."""
import torch
from generative_modeling.variational.celeba_vqvae import CelebAVQVAE


def test_vqvae():
    print("Testing VQ-VAE implementation...")

    # Model config (same as training script)
    config = {
        "num_embeddings": 512,
        "embedding_dim": 256,
        "image_size": 128,
        "hidden_dims": [32, 64, 128, 256, 512],
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model
    model = CelebAVQVAE(**config).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, config["image_size"], config["image_size"]).to(device)
    print(f"\nInput shape: {x.shape}")

    recon, vq_loss, perplexity = model(x)
    print(f"Reconstruction shape: {recon.shape}")
    print(f"VQ loss: {vq_loss.item():.4f}")
    print(f"Perplexity: {perplexity.item():.4f}")

    # Check shapes
    assert recon.shape == x.shape, f"Reconstruction shape mismatch: {recon.shape} vs {x.shape}"
    assert vq_loss.item() >= 0, "VQ loss should be non-negative"
    assert 1 <= perplexity.item() <= config["num_embeddings"], f"Perplexity out of range: {perplexity.item()}"

    # Test encode/decode separately
    z_e = model.encode(x)
    print(f"\nEncoder output shape: {z_e.shape}")
    assert z_e.shape[1] == config["embedding_dim"], f"Embedding dim mismatch: {z_e.shape[1]} vs {config['embedding_dim']}"

    # Test sampling
    samples = model.sample(8, device)
    print(f"\nSamples shape: {samples.shape}")
    assert samples.shape == (8, 3, config["image_size"], config["image_size"]), f"Samples shape mismatch: {samples.shape}"

    # Test gradient flow
    loss = torch.nn.functional.mse_loss(recon, x) + vq_loss
    loss.backward()
    print("\nGradient flow test passed")

    # Check codebook usage
    with torch.no_grad():
        z_e = model.encode(x)
        z_q, _, _, encodings = model.vq_layer(z_e)
        print(f"\nCodebook usage: {encodings.sum(dim=0).nonzero().numel()} / {config['num_embeddings']} codes used")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_vqvae()
