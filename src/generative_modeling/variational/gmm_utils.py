import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image


def load_data(data_path):
    """Load GMM training data from file."""
    data = np.load(data_path)
    X = data["X"]
    y = data["y"]
    true_means = data["true_means"]
    true_covs = data["true_covs"]
    true_weights = data["true_weights"]
    return X, y, true_means, true_covs, true_weights


def save_results(model, log_likelihoods, train_time, output_dir, config=None):
    """Save model and training results to disk."""
    results = {
        "means": model.means.detach().cpu().numpy(),
        "variances": model.variances.detach().cpu().numpy(),
        "weights": model.weights.detach().cpu().numpy(),
        "log_likelihoods": log_likelihoods,
        "train_time": train_time,
    }
    np.savez(output_dir / "results.npz", **results)
    
    save_dict = {"model_state_dict": model.state_dict()}
    if config is not None:
        save_dict["config"] = config
    torch.save(save_dict, output_dir / "final_model.pt")


def plot_covariance_ellipse(ax, mean, variances, color, n_std=2.0):
    """Plot covariance ellipse for a Gaussian component."""
    width = 2 * n_std * np.sqrt(variances[0])
    height = 2 * n_std * np.sqrt(variances[1])
    ellipse = Ellipse(mean, width, height, facecolor='none', 
                     edgecolor=color, linewidth=2, linestyle='--', alpha=0.8)
    ax.add_patch(ellipse)
    ax.scatter(mean[0], mean[1], c=color, s=100, edgecolors='black', 
              linewidths=1.5, zorder=5, marker='o')


def save_iteration_checkpoint(model, X, y, iteration, log_likelihood, output_dir, config):
    """Save model checkpoint and visualization at a given iteration."""
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    with torch.no_grad():
        responsibilities = model.e_step(torch.tensor(X, dtype=torch.float32))
        assignments = torch.argmax(responsibilities, dim=1).cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    learned_colors = ["#d62728", "#9467bd", "#8c564b"]
    
    # Set fixed axis limits based on data range to ensure consistent image sizes
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    ax.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    for idx in range(config["n_components"]):
        mask = assignments == idx
        ax.scatter(X[mask, 0], X[mask, 1], c=learned_colors[idx], alpha=0.4, s=20, edgecolors="none")
    
    learned_means = model.means.detach().cpu().numpy()
    learned_vars = model.variances.detach().cpu().numpy()
    
    for idx in range(config["n_components"]):
        plot_covariance_ellipse(ax, learned_means[idx], learned_vars[idx], learned_colors[idx])
    
    ax.set_xlabel("X₁", fontsize=12)
    ax.set_ylabel("X₂", fontsize=12)
    ax.set_title(f"Learned Assignments (iter {iteration}, LL={log_likelihood:.1f})", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    
    # Use fixed layout to avoid changing image dimensions between frames
    plt.tight_layout()
    # Remove bbox_inches="tight" as it causes variable image dimensions
    plt.savefig(checkpoint_dir / f"iter_{iteration:04d}.png", dpi=150)
    plt.close()
    
    torch.save({
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "log_likelihood": log_likelihood,
    }, checkpoint_dir / f"model_iter_{iteration:04d}.pt")


def visualize_convergence(log_likelihoods, output_dir, title, color):
    """Create convergence plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    iterations = np.arange(0, len(log_likelihoods))
    ax.plot(iterations, log_likelihoods, linewidth=2, color=color)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Log Likelihood", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "convergence.png", dpi=150, bbox_inches="tight")
    plt.close()


def create_animation(output_dir):
    """Create GIF animation from checkpoint images."""
    checkpoint_dir = output_dir / "checkpoints"
    
    png_files = sorted(checkpoint_dir.glob("iter_*.png"))
    if not png_files:
        return
    
    frames = []
    target_size = None
    
    for f in png_files:
        img = Image.open(f)
        if target_size is None:
            target_size = img.size
        
        # Resize if somehow the image dimensions still vary
        if img.size != target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        frames.append(img)
    
    if not frames:
        return
    
    frames[0].save(
        output_dir / "training_animation.gif",
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0
    )
    
    print(f"Created animation with {len(frames)} frames")
