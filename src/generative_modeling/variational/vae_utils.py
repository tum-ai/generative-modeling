import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms


def load_mnist(data_dir, normalize=True):
    """Load MNIST dataset."""
    transform_list = [transforms.ToTensor()]
    if normalize:
        # Normalize using MNIST mean and std
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    
    transform = transforms.Compose(transform_list)
    
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )
    
    return train_dataset, test_dataset


def save_results(models, losses_dict, train_time, output_dir, config):
    """Save models and training results to disk."""
    for beta, model in models.items():
        model_dir = output_dir / f"beta_{beta}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
        }, model_dir / "final_model.pt")
    
    # Save losses
    results = {
        "losses": losses_dict,
        "train_time": train_time,
        "config": config,
    }
    np.savez(output_dir / "results.npz", **results)


def save_iteration_checkpoint(model, beta, test_loader, iteration, losses, output_dir, device):
    """Save model checkpoint and visualizations at a given iteration."""
    checkpoint_dir = output_dir / f"beta_{beta}" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Get one batch for visualization
    with torch.no_grad():
        # Get examples of each digit (0-9)
        test_images = []
        test_labels = []
        digit_counts = {i: 0 for i in range(10)}
        
        for images, labels in test_loader:
            for img, label in zip(images, labels):
                label_val = label.item()
                if digit_counts[label_val] == 0:
                    test_images.append(img)
                    test_labels.append(label_val)
                    digit_counts[label_val] += 1
                if all(c > 0 for c in digit_counts.values()):
                    break
            if all(c > 0 for c in digit_counts.values()):
                break
        
        # Sort by label
        test_images = torch.stack(test_images)
        test_labels = torch.tensor(test_labels)
        sort_idx = torch.argsort(test_labels)
        test_images = test_images[sort_idx]
        test_labels = test_labels[sort_idx]
        
        # Move to device and flatten
        x = test_images.view(-1, 784).to(device)
        
        # Forward pass
        x_recon, mu, log_var, z = model(x)
        
        # Denormalize for visualization: x = x_norm * std + mean
        x_denorm = x * model.mnist_std + model.mnist_mean
        x_recon_denorm = x_recon * model.mnist_std + model.mnist_mean
        x_recon_denorm = torch.clamp(x_recon_denorm, 0, 1)
        
        # Move to CPU for visualization
        x_np = x_denorm.cpu().numpy()
        x_recon_np = x_recon_denorm.cpu().numpy()
    
    # 1. Save reconstruction visualization
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    for i in range(10):
        # Original
        axes[0, i].imshow(x_np[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        axes[0, i].set_title(f'{i}', fontsize=12)
        
        # Reconstruction
        axes[1, i].imshow(x_recon_np[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Original', fontsize=12)
    axes[1, 0].set_ylabel('Reconstructed', fontsize=12)
    
    loss_str = f"Loss: {losses['total'][-1]:.2f} (R: {losses['recon'][-1]:.2f}, KL: {losses['kl'][-1]:.2f})"
    fig.suptitle(f'Beta={beta}, Iter {iteration}, {loss_str}', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(checkpoint_dir / f"recon_{iteration:04d}.png", dpi=150)
    plt.close()
    
    # 2. Save latent space visualization
    # Get full test set for latent space visualization
    all_z = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            x = images.view(-1, 784).to(device)
            mu, _ = model.encode(x)
            all_z.append(mu.cpu())
            all_labels.append(labels)
    
    all_z = torch.cat(all_z, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Color map for digits
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for digit in range(10):
        mask = all_labels == digit
        ax.scatter(all_z[mask, 0], all_z[mask, 1], c=[colors[digit]], 
                  label=str(digit), alpha=0.6, s=10, edgecolors='none')
    
    ax.set_xlabel('z₁', fontsize=12)
    ax.set_ylabel('z₂', fontsize=12)
    ax.set_title(f'Latent Space (Beta={beta}, Iter {iteration})', fontsize=14)
    ax.legend(fontsize=10, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(checkpoint_dir / f"latent_{iteration:04d}.png", dpi=150)
    plt.close()
    
    # Save model checkpoint
    torch.save({
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "losses": {k: v[-1] for k, v in losses.items()},
    }, checkpoint_dir / f"model_{iteration:04d}.pt")


def visualize_loss_curves(losses_dict, output_dir):
    """Create loss curves for all betas."""
    betas = sorted(losses_dict.keys())
    
    # Create subplots for each loss type
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    loss_types = ['total', 'recon', 'kl']
    titles = ['Total Loss (Recon + β·KL)', 'Reconstruction Loss', 'KL Divergence']
    colors = plt.cm.viridis(np.linspace(0, 1, len(betas)))
    
    for ax, loss_type, title in zip(axes, loss_types, titles):
        for beta, color in zip(betas, colors):
            losses = losses_dict[beta][loss_type]
            iterations = np.arange(len(losses))
            ax.plot(iterations, losses, label=f'β={beta}', color=color, linewidth=2)
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def create_animations(output_dir, betas):
    """Create GIF animations from checkpoint images."""
    for beta in betas:
        checkpoint_dir = output_dir / f"beta_{beta}" / "checkpoints"
        
        # Create reconstruction animation
        recon_files = sorted(checkpoint_dir.glob("recon_*.png"))
        if recon_files:
            frames = []
            target_size = None
            
            for f in recon_files:
                img = Image.open(f)
                if target_size is None:
                    target_size = img.size
                if img.size != target_size:
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                frames.append(img)
            
            if frames:
                frames[0].save(
                    output_dir / f"beta_{beta}" / "recon_animation.gif",
                    save_all=True,
                    append_images=frames[1:],
                    duration=200,
                    loop=0
                )
                print(f"Created reconstruction animation for beta={beta} with {len(frames)} frames")
        
        # Create latent space animation
        latent_files = sorted(checkpoint_dir.glob("latent_*.png"))
        if latent_files:
            frames = []
            target_size = None
            
            for f in latent_files:
                img = Image.open(f)
                if target_size is None:
                    target_size = img.size
                if img.size != target_size:
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                frames.append(img)
            
            if frames:
                frames[0].save(
                    output_dir / f"beta_{beta}" / "latent_animation.gif",
                    save_all=True,
                    append_images=frames[1:],
                    duration=200,
                    loop=0
                )
                print(f"Created latent space animation for beta={beta} with {len(frames)} frames")
