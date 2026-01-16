import torch
from torch import nn

def soft_clamp(x, clamp_val=5.0):
    """Soft differentiable clamp for numerical stability (NVAE)"""
    return clamp_val * torch.tanh(x / clamp_val)


class BottomUp(nn.Module):
    """Bottom-up encoder: extracts features at multiple scales, projects to vector latent params"""
    
    def __init__(self, image_size=128, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
        in_ch = 3
        modules = []
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Conv2d(in_ch, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU()
            ))
            in_ch = h_dim
        self.encoder = nn.ModuleList(modules)
        
        # Compute flattened sizes at each level
        with torch.no_grad():
            x = torch.zeros(1, 3, image_size, image_size)
            self.flat_sizes = []
            for layer in self.encoder:
                x = layer(x)
                self.flat_sizes.append(x.shape[1] * x.shape[2] * x.shape[3])
    
    def forward(self, x):
        """Returns flattened features at each level (fine to coarse)"""
        features = []
        for layer in self.encoder:
            x = layer(x)
            features.append(x.flatten(start_dim=1))
        return features


class TopDown(nn.Module):
    """Shared top-down network for generation and inference with vector latents"""
    
    def __init__(self, latent_dims=(32, 128, 512), bu_flat_sizes=None, decoder_dim=512):
        super().__init__()
        self.latent_dims = latent_dims  # z1=32, z2=128, z3=512
        self.num_levels = len(latent_dims)
        self.decoder_dim = decoder_dim
        
        # Prior networks: hidden state -> (mu, log_sigma) for each level
        self.prior_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(decoder_dim, decoder_dim), nn.LeakyReLU(), nn.Linear(decoder_dim, 2 * latent_dims[i]))
            for i in range(self.num_levels)
        ])
        
        # Posterior delta networks: bottom-up features -> (delta_mu, delta_log_sigma)
        self.posterior_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(bu_flat_sizes[4 - i], decoder_dim), nn.LeakyReLU(), nn.Linear(decoder_dim, 2 * latent_dims[i]))
            for i in range(self.num_levels)
        ])
        
        # Combiners: merge z with hidden state
        self.combiners = nn.ModuleList([
            nn.Sequential(nn.Linear(decoder_dim + latent_dims[i], decoder_dim), nn.LeakyReLU())
            for i in range(self.num_levels)
        ])
        
        # Learnable initial hidden state
        self.h0 = nn.Parameter(torch.randn(decoder_dim) * 0.01)
    
    def forward(self, bu_features):
        """Inference: sample from posterior q(z|x) and compute KL losses"""
        batch_size = bu_features[0].shape[0]
        device = self.h0.device
        
        zs, kl_losses = [], []
        h = self.h0.unsqueeze(0).expand(batch_size, -1)
        
        for i in range(self.num_levels):
            # Prior: top level is N(0,I), subsequent levels are learned from h
            if i == 0:
                mu_p = torch.zeros(batch_size, self.latent_dims[i], device=device)
                log_sig_p = torch.zeros(batch_size, self.latent_dims[i], device=device)
            else:
                prior_params = self.prior_nets[i](h)
                mu_p, log_sig_p = torch.chunk(prior_params, 2, dim=1)
                mu_p, log_sig_p = soft_clamp(mu_p), soft_clamp(log_sig_p)
            
            # Posterior: residual parameterization (use features from coarse to fine: indices 4, 3, 2)
            delta_params = self.posterior_nets[i](bu_features[4 - i])
            delta_mu, delta_log_sig = torch.chunk(delta_params, 2, dim=1)
            delta_mu, delta_log_sig = soft_clamp(delta_mu), soft_clamp(delta_log_sig)
            
            mu_q = mu_p + delta_mu
            log_sig_q = log_sig_p + delta_log_sig
            
            # Sample from posterior
            std_q = torch.exp(log_sig_q) + 1e-2
            z = mu_q + std_q * torch.randn_like(std_q)
            
            # KL(q || p)
            std_p = torch.exp(log_sig_p) + 1e-2
            kl = 0.5 * (((mu_q - mu_p) / std_p) ** 2 + (std_q / std_p) ** 2 - 1 - 2 * (log_sig_q - log_sig_p))
            kl_losses.append(kl.sum(dim=1).mean())
            
            zs.append(z)
            h = self.combiners[i](torch.cat([h, z], dim=1))
        
        return h, zs, kl_losses
    
    def sample(self, batch_size, device):
        """Generate samples from prior: z1 ~ N(0,I), z2 ~ p(z2|z1), z3 ~ p(z3|z1,z2)"""
        h = self.h0.unsqueeze(0).expand(batch_size, -1).to(device)
        
        for i in range(self.num_levels):
            if i == 0:
                mu_p = torch.zeros(batch_size, self.latent_dims[i], device=device)
                log_sig_p = torch.zeros(batch_size, self.latent_dims[i], device=device)
            else:
                prior_params = self.prior_nets[i](h)
                mu_p, log_sig_p = torch.chunk(prior_params, 2, dim=1)
                mu_p, log_sig_p = soft_clamp(mu_p), soft_clamp(log_sig_p)
            
            std_p = torch.exp(log_sig_p) + 1e-2
            z = mu_p + std_p * torch.randn_like(std_p)
            
            h = self.combiners[i](torch.cat([h, z], dim=1))
        
        return h


class CelebAHierarchicalVAE(nn.Module):
    """Hierarchical VAE with vector latents and shared top-down network"""
    
    def __init__(self, image_size=128, latent_dims=(32, 128, 512), hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
        self.image_size = image_size
        self.latent_dims = latent_dims
        decoder_dim = hidden_dims[-1]
        
        # Bottom-up encoder
        self.bottom_up = BottomUp(image_size, hidden_dims)
        
        # Top-down network
        self.top_down = TopDown(latent_dims, self.bottom_up.flat_sizes, decoder_dim)
        
        # Decoder: project hidden state to spatial, then upsample to image
        self.decoder_proj = nn.Linear(decoder_dim, decoder_dim * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(decoder_dim, hidden_dims[3], kernel_size=4, stride=2, padding=1),  # 4->8
            nn.BatchNorm2d(hidden_dims[3]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_dims[3], hidden_dims[2], kernel_size=4, stride=2, padding=1),  # 8->16
            nn.BatchNorm2d(hidden_dims[2]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_dims[2], hidden_dims[1], kernel_size=4, stride=2, padding=1),  # 16->32
            nn.BatchNorm2d(hidden_dims[1]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_dims[1], hidden_dims[0], kernel_size=4, stride=2, padding=1),  # 32->64
            nn.BatchNorm2d(hidden_dims[0]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_dims[0], hidden_dims[0], kernel_size=4, stride=2, padding=1),  # 64->128
            nn.BatchNorm2d(hidden_dims[0]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[0], 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.decoder_dim = decoder_dim
    
    def forward(self, x):
        # Bottom-up: extract flattened features at each level
        bu_features = self.bottom_up(x)
        
        # Top-down: get latents and KL losses
        h, zs, kl_losses = self.top_down(bu_features)
        
        # Decode to image
        h = self.decoder_proj(h).view(-1, self.decoder_dim, 4, 4)
        recon = self.decoder(h)
        
        return recon, zs, kl_losses
    
    def sample(self, batch_size, device):
        """Generate samples from the prior"""
        h = self.top_down.sample(batch_size, device)
        h = self.decoder_proj(h).view(-1, self.decoder_dim, 4, 4)
        return self.decoder(h)
