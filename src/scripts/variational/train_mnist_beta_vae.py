import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import time

from generative_modeling.variational import BetaVAE, vae_utils


def neg_elbo(x, x_recon, mu, log_var, beta=1.0):
    """
    Compute negative ELBO (Beta-VAE loss).
    
    Args:
        x: Input images (batch_size, 784)
        x_recon: Reconstructed images mean (batch_size, 784)
        mu: Latent mean (batch_size, latent_dim)
        log_var: Latent log variance (batch_size, latent_dim)
        beta: Weight for KL term
    
    Returns:
        loss: Total loss
        recon_loss: Reconstruction loss (MSE)
        kl_loss: KL divergence loss
    """
    recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
    
    kl_loss = -0.5 * torch.sum(
        1 + log_var - mu.pow(2) - log_var.exp()
    ) / x.size(0)
    
    loss = recon_loss + beta * kl_loss
    return loss, recon_loss, kl_loss


CONFIG = {
    "data_dir": Path("data/mnist"),
    "output_dir": Path("out/variational/beta_vae"),
    "seed": 42,
    "betas": [0.5, 1.0, 4.0],  # List of beta values to train
    "latent_dim": 2,
    "hidden_dim": 400,
    "batch_size": 128,
    "n_epochs": 50,
    "learning_rate": 1e-3,
    "save_interval": 5,  # Save checkpoint every N epochs
}


def train_epoch(model, train_loader, optimizer, beta, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    for x, _ in train_loader:
        x = x.view(-1, 784).to(device)
        
        # Forward pass
        x_recon, mu, log_var, z = model(x)
        
        # Compute loss
        loss, recon_loss, kl_loss = neg_elbo(x, x_recon, mu, log_var, beta)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item() * x.size(0)
        total_recon += recon_loss.item() * x.size(0)
        total_kl += kl_loss.item() * x.size(0)
    
    n_samples = len(train_loader.dataset)
    return total_loss / n_samples, total_recon / n_samples, total_kl / n_samples


def evaluate(model, test_loader, beta, device):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.view(-1, 784).to(device)
            
            x_recon, mu, log_var, z = model(x)
            
            loss, recon_loss, kl_loss = neg_elbo(x, x_recon, mu, log_var, beta)
            
            total_loss += loss.item() * x.size(0)
            total_recon += recon_loss.item() * x.size(0)
            total_kl += kl_loss.item() * x.size(0)
    
    n_samples = len(test_loader.dataset)
    return total_loss / n_samples, total_recon / n_samples, total_kl / n_samples


def train_beta_vae(beta, config, train_loader, test_loader, device):
    """Train a Beta-VAE with a specific beta value."""
    print(f"\n{'='*60}")
    print(f"Training Beta-VAE with β={beta}")
    print(f"{'='*60}")
    
    # Initialize model
    model = BetaVAE(
        latent_dim=config["latent_dim"],
        hidden_dim=config["hidden_dim"]
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # Track losses
    losses = {
        "total": [],
        "recon": [],
        "kl": [],
    }
    
    # Training loop
    pbar = tqdm(range(config["n_epochs"]), desc=f"β={beta}")
    for epoch in pbar:
        # Train
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, beta, device
        )
        
        # Evaluate
        test_loss, test_recon, test_kl = evaluate(
            model, test_loader, beta, device
        )
        
        # Store losses (using test set losses for visualization)
        losses["total"].append(test_loss)
        losses["recon"].append(test_recon)
        losses["kl"].append(test_kl)
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{test_loss:.2f}",
            "recon": f"{test_recon:.2f}",
            "kl": f"{test_kl:.2f}"
        })
        
        # Save checkpoint
        if epoch % config["save_interval"] == 0 or epoch == config["n_epochs"] - 1:
            vae_utils.save_iteration_checkpoint(
                model, beta, test_loader, epoch, losses, config["output_dir"], device
            )
    
    return model, losses


def main():
    # Set random seeds
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    
    # Create output directory
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_dataset, test_dataset = vae_utils.load_mnist(CONFIG["data_dir"])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=0
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Train models for each beta
    models = {}
    losses_dict = {}
    start_time = time.time()
    
    for beta in CONFIG["betas"]:
        model, losses = train_beta_vae(
            beta, CONFIG, train_loader, test_loader, device
        )
        models[beta] = model
        losses_dict[beta] = losses
    
    train_time = time.time() - start_time
    
    # Save results
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Total training time: {train_time:.2f}s")
    print("="*60)
    
    # Create visualizations
    print("\nCreating visualizations...")
    vae_utils.save_results(models, losses_dict, train_time, CONFIG["output_dir"], CONFIG)
    vae_utils.visualize_loss_curves(losses_dict, CONFIG["output_dir"])
    vae_utils.create_animations(CONFIG["output_dir"], CONFIG["betas"])
    
    print(f"\nResults saved to {CONFIG['output_dir']}")
    
    # Print final losses
    print("\nFinal test losses:")
    for beta in CONFIG["betas"]:
        losses = losses_dict[beta]
        print(f"  β={beta}: Total={losses['total'][-1]:.2f}, "
              f"Recon={losses['recon'][-1]:.2f}, KL={losses['kl'][-1]:.2f}")


if __name__ == "__main__":
    main()
