import torch
from torch import nn


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Codebook: [num_embeddings, embedding_dim]
        # Each row is a learnable embedding vector
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def compute_vq_loss(self, z_e_flat, z_q_flat, encodings):
        # Codebook loss: move codebook entries towards encoder outputs
        # detach() on z_e ensures gradients only flow to codebook (z_q)
        codebook_loss = torch.mean((z_q_flat - z_e_flat.detach()) ** 2)

        # Commitment loss: encourage encoder to commit to codebook
        # detach() on z_q ensures gradients only flow to encoder (z_e)
        commitment_loss = torch.mean((z_q_flat.detach() - z_e_flat) ** 2)

        # Total VQ loss
        loss = codebook_loss + self.commitment_cost * commitment_loss

        # Perplexity: measures how uniformly the codebook is used
        e_mean = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        return loss, perplexity

    def forward(self, z_e):
        # Reshape to [B*H*W, C] for distance computation
        z_e_flat = z_e.permute(0, 2, 3, 1).contiguous()
        z_e_flat = z_e_flat.view(-1, self.embedding_dim)

        # Compute distances to all codebook entries: [B*H*W, num_embeddings]
        # Using expanded form: ||z_e - e_k||^2 = ||z_e||^2 - 2*z_e*e_k + ||e_k||^2
        distances = (
            torch.sum(z_e_flat ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(z_e_flat, self.embedding.weight.t())
            + torch.sum(self.embedding.weight ** 2, dim=1)
        )

        # Get nearest codebook indices (argmin is non-differentiable)
        indices = torch.argmin(distances, dim=1)

        # One-hot encodings for perplexity computation
        encodings = torch.zeros(z_e_flat.shape[0], self.num_embeddings, device=z_e.device)
        encodings.scatter_(1, indices.unsqueeze(1), 1)

        # Quantize: look up codebook entries using indices
        z_q_flat = self.embedding(indices)
        z_q = z_q_flat.view(z_e.shape[0], z_e.shape[2], z_e.shape[3], self.embedding_dim).permute(0, 3, 1, 2).contiguous()

        # Compute VQ loss (codebook + commitment)
        loss, perplexity = self.compute_vq_loss(z_e_flat, z_q_flat, encodings)

        # Straight-through estimator (STE): copy gradients from decoder to encoder
        z_q = z_e + (z_q - z_e).detach()

        return z_q, loss, perplexity, encodings


class CelebAVQVAE(nn.Module):
    """VQ-VAE for CelebA with vector quantized latent codes."""

    def __init__(self, num_embeddings=512, embedding_dim=256, image_size=128, hidden_dims=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.image_size = image_size

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = hidden_dims
        self.final_dim = hidden_dims[-1]

        # Encoder: same as Beta-VAE
        in_channels = 3
        modules = []
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU()
            ))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)

        # Project to embedding_dim for quantization
        self.pre_quant_conv = nn.Conv2d(self.final_dim, embedding_dim, kernel_size=1)

        # Vector quantizer
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim)

        # Project back from embedding_dim
        self.post_quant_conv = nn.Conv2d(embedding_dim, self.final_dim, kernel_size=1)

        # Decoder: same as Beta-VAE
        modules = []
        reversed_hidden_dims = list(reversed(hidden_dims))
        for i in range(len(reversed_hidden_dims) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(reversed_hidden_dims[i], reversed_hidden_dims[i + 1],
                                   kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(reversed_hidden_dims[i + 1]),
                nn.LeakyReLU()
            ))
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(reversed_hidden_dims[-1], reversed_hidden_dims[-1],
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(reversed_hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(reversed_hidden_dims[-1], 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        """Encode to continuous latent codes."""
        z = self.encoder(x)
        z_e = self.pre_quant_conv(z)
        return z_e

    def decode(self, z_q):
        """Decode from quantized latent codes."""
        z = self.post_quant_conv(z_q)
        z = self.decoder(z)
        recon = self.final_layer(z)
        return recon

    def forward(self, x):
        """Forward pass: encode -> quantize -> decode."""
        z_e = self.encode(x)
        z_q, vq_loss, perplexity, encodings = self.vq_layer(z_e)
        recon = self.decode(z_q)
        return recon, vq_loss, perplexity

    def sample(self, num_samples, device):
        """Sample random codes from codebook and decode."""
        # Sample random codebook indices
        indices = torch.randint(0, self.num_embeddings, (num_samples, 4, 4), device=device)
        z_q = self.vq_layer.embedding(indices).permute(0, 3, 1, 2)
        return self.decode(z_q)
