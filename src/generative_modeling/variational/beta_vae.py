import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaVAE(nn.Module):
    def __init__(self, latent_dim=2, hidden_dim=400, mnist_mean=0.1307, mnist_std=0.3081):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.input_dim = 784  # 28x28 MNIST images
        self.mnist_mean = mnist_mean
        self.mnist_std = mnist_std
        
        self.encoder_fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.encoder_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_log_var = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_out = nn.Linear(hidden_dim, self.input_dim)
    
    def encode(self, x):
        """evaluate posterior parameters"""
        h = F.relu(self.encoder_fc1(x))
        h = F.relu(self.encoder_fc2(h))
        mu = self.encoder_mu(h)
        log_var = self.encoder_log_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """sample from posterior"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def decode(self, z):
        """sample likelihood parameters"""
        h = F.relu(self.decoder_fc1(z))
        h = F.relu(self.decoder_fc2(h))
        return self.decoder_out(h)  # mean for Gaussian likelihood
    
    def forward(self, x):
        """sample from posterior and decode"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var, z
    
    def sample(self, num_samples, device='cpu'):
        """sample from prior and decode"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            x_recon = self.decode(z)
            x_recon = x_recon * self.mnist_std + self.mnist_mean
            return torch.clamp(x_recon, 0, 1)  # clamp to valid range
