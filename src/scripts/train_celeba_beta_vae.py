import os
import random
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
import wandb

# project modules
from generative_modeling.variational.celeba_beta_vae import CelebABetaVAE
from generative_modeling.losses import sobel_loss_2d

# Parameters in dict at top
CONFIG = {
    "latent_dim": 128,
    "image_size": 128,
    "hidden_dims": [32, 64, 128, 256, 512],
    "epochs": 20,
    "batch_size": 256,
    "lr": 1e-3,
    "kld_weight": 0.00025,
    "mse_weight": 0.5,
    "sobel_weight": 0.5,
    "sobel_loss_type": "L2",
    "celeb_path": "./data/",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "project_name": "gm-variational",
    "run_name": "celeba-vae-kl0.00025-sobel0.5",
    "seed": 42,
    "num_workers": 8,
}
CONFIG["checkpoint_dir"] = "out/checkpoints/" + CONFIG["run_name"]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataloaders(config):
    celeb_transform = transforms.Compose([
        transforms.Resize(config["image_size"], antialias=True),
        transforms.CenterCrop(config["image_size"]),
        transforms.ToTensor()
    ])

    # Ensure data directory exists
    os.makedirs(config["celeb_path"], exist_ok=True)

    # Manual download is required as Google Drive limits are often hit
    train_dataset = CelebA(config["celeb_path"], transform=celeb_transform, download=False, split='train')
    test_dataset = CelebA(config["celeb_path"], transform=celeb_transform, download=False, split='valid')

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )
    
    return train_loader, test_loader

def loss_function(recon_x, x, mu, log_var, config):
    # Compute MSE reconstruction loss
    MSE = F.mse_loss(recon_x, x, reduction='mean')
    
    # Compute Sobel gradient loss for sharper reconstructions
    sobel_loss = sobel_loss_2d(recon_x, x, loss_type=config["sobel_loss_type"])
    
    # Combined reconstruction loss
    recon_loss = config["mse_weight"] * MSE + config["sobel_weight"] * sobel_loss
    
    # KL divergence loss
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Total ELBO loss
    loss = recon_loss + config["kld_weight"] * KLD
    
    return loss, MSE, KLD, sobel_loss

def train(model, train_loader, optimizer, epoch, config):
    model.train()
    train_loss = 0
    total_mse = 0
    total_kld = 0
    total_sobel = 0
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} [Train]")
    for batch_idx, (data, _) in pbar:
        data = data.to(config["device"])
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = model(data)
        log_var = torch.clamp_(log_var, -10, 10)
        
        loss, mse, kld, sobel = loss_function(recon_batch, data, mu, log_var, config)
        loss.backward()
        
        optimizer.step()
        
        train_loss += loss.item()
        total_mse += mse.item()
        total_kld += kld.item()
        total_sobel += sobel.item()
        
        if batch_idx % 100 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            wandb.log({
                "batch_loss": loss.item(),
                "batch_mse": mse.item(),
                "batch_kld": kld.item(),
                "batch_sobel": sobel.item(),
                "epoch": epoch
            })

    avg_loss = train_loss / len(train_loader)
    avg_mse = total_mse / len(train_loader)
    avg_kld = total_kld / len(train_loader)
    avg_sobel = total_sobel / len(train_loader)
    
    return avg_loss, avg_mse, avg_kld, avg_sobel

def test(model, test_loader, epoch, config):
    model.eval()
    test_loss = 0
    test_mse = 0
    test_kld = 0
    test_sobel = 0
    
    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Epoch {epoch} [Test ]")
        for i, (data, _) in pbar:
            data = data.to(config["device"])
            recon_batch, mu, log_var = model(data)
            
            loss, mse, kld, sobel = loss_function(recon_batch, data, mu, log_var, config)
            test_loss += loss.item()
            test_mse += mse.item()
            test_kld += kld.item()
            test_sobel += sobel.item()
            
            if i == 0:
                # Log some reconstructions to wandb
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch[:n]])
                grid = make_grid(comparison, nrow=n)
                wandb.log({"reconstructions": wandb.Image(grid, caption=f"Epoch {epoch}")})

    avg_test_loss = test_loss / len(test_loader)
    avg_test_mse = test_mse / len(test_loader)
    avg_test_kld = test_kld / len(test_loader)
    avg_test_sobel = test_sobel / len(test_loader)
    
    return avg_test_loss, avg_test_mse, avg_test_kld, avg_test_sobel

def main():
    # Set seed for reproducibility
    set_seed(CONFIG["seed"])
    
    # Initialize wandb
    wandb.init(
        project=CONFIG["project_name"], 
        name=CONFIG["run_name"],
        config=CONFIG
    )
    
    # Create checkpoint directory
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    
    # Initialize model and optimizer
    model = CelebABetaVAE(
        latent_dim=CONFIG["latent_dim"],
        image_size=CONFIG["image_size"],
        hidden_dims=CONFIG["hidden_dims"]
    ).to(CONFIG["device"])
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    
    # Get dataloaders
    train_loader, test_loader = get_dataloaders(CONFIG)
    
    # Create fixed latent vectors for consistent sampling across epochs
    # We set the seed again here just to be double sure about fixed_z consistency if main is called multiple times
    torch.manual_seed(CONFIG["seed"])
    fixed_z = torch.randn(64, CONFIG["latent_dim"]).to(CONFIG["device"])
    
    print(f'Starting training for {CONFIG["epochs"]} epochs on {CONFIG["device"]}...')
    
    for epoch in range(1, CONFIG["epochs"] + 1):
        avg_train_loss, avg_train_mse, avg_train_kld, avg_train_sobel = train(model, train_loader, optimizer, epoch, CONFIG)
        avg_test_loss, avg_test_mse, avg_test_kld, avg_test_sobel = test(model, test_loader, epoch, CONFIG)
        
        print(f'====> Epoch: {epoch} Average train loss: {avg_train_loss:.4f}, Test loss: {avg_test_loss:.4f}')
        
        # Log epoch metrics
        wandb.log({
            "epoch_train_loss": avg_train_loss,
            "epoch_train_mse": avg_train_mse,
            "epoch_train_kld": avg_train_kld,
            "epoch_train_sobel": avg_train_sobel,
            "epoch_test_loss": avg_test_loss,
            "epoch_test_mse": avg_test_mse,
            "epoch_test_kld": avg_test_kld,
            "epoch_test_sobel": avg_test_sobel,
            "epoch": epoch
        })
        
        # Save checkpoint locally
        checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], f"vae_model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        
        # Log samples using fixed latent vectors
        with torch.no_grad():
            samples = model.decode(fixed_z)
            sample_grid = make_grid(samples, nrow=8)
            wandb.log({"samples": wandb.Image(sample_grid, caption=f"Epoch {epoch}")})

    wandb.finish()

if __name__ == "__main__":
    main()
