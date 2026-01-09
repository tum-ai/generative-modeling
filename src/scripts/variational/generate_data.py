import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CONFIG = {
    "output_dir": Path("data/variational"),
    "data_filename": "gaussian_cluster_data.npz",
    "viz_filename": "gaussian_cluster_data.png",
    "seed": 42,
    "n_samples": 1000,
    "n_components": 3,
}

# gaussian cluster params
COMPONENT_NAMES = ["Base", "Overlap", "Isolated"]
COMPONENT_MEANS = np.array(
    [
        [0.0, 0.0],  # Cluster 1: Base
        [1.5, 0.0],  # Cluster 2: Overlap
        [0.75, 2.5],  # Cluster 3: Isolated
    ]
)
COMPONENT_COVS = np.array(
    [
        [[0.5, 0.0], [0.0, 0.5]],  # Cluster 1
        [[0.5, 0.0], [0.0, 0.5]],  # Cluster 2
        [[0.4, 0.0], [0.0, 0.4]],  # Cluster 3
    ]
)
COMPONENT_WEIGHTS = np.array([1 / 3, 1 / 3, 1 / 3])


def generate_gaussian_cluster_data(config):
    np.random.seed(config["seed"])

    n_samples = config["n_samples"]

    # Calculate number of samples per component
    n_samples_per_component = np.random.multinomial(n_samples, COMPONENT_WEIGHTS)

    X_list = []
    y_list = []

    for idx, n_comp_samples in enumerate(n_samples_per_component):
        # Generate samples from this component
        mean = COMPONENT_MEANS[idx]
        cov = COMPONENT_COVS[idx]

        X_comp = np.random.multivariate_normal(mean, cov, size=n_comp_samples)
        y_comp = np.full(n_comp_samples, idx)

        X_list.append(X_comp)
        y_list.append(y_comp)

    # Concatenate all components
    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # Shuffle the data
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    return X, y


def visualize_data(X, y, config, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color map for clusters
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Plot each component with different color
    for idx in range(config["n_components"]):
        mask = y == idx
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            c=colors[idx],
            label=f"Component {idx + 1}: {COMPONENT_NAMES[idx]}",
            alpha=0.6,
            s=30,
            edgecolors="none",
        )

    # Plot true means
    for idx in range(config["n_components"]):
        mean = COMPONENT_MEANS[idx]
        ax.scatter(
            mean[0],
            mean[1],
            c=colors[idx],
            marker="X",
            s=300,
            edgecolors="black",
            linewidths=2,
            zorder=5,
        )

    ax.set_xlabel("X₁", fontsize=14)
    ax.set_ylabel("X₂", fontsize=14)
    ax.set_title(
        f"Gaussian Cluster Dataset (N={config['n_samples']}, K={config['n_components']})",
        fontsize=16,
    )
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def save_data(X, y, config, save_path):
    """
    Save the generated data to disk.

    Args:
        X: (N, 2) array of data points
        y: (N,) array of true component labels
        config: Configuration dictionary
        save_path: Path to save the data
    """
    np.savez(
        save_path,
        X=X,
        y=y,
        true_means=COMPONENT_MEANS,
        true_covs=COMPONENT_COVS,
        true_weights=COMPONENT_WEIGHTS,
        n_components=config["n_components"],
        seed=config["seed"],
    )
    print(f"Saved: {save_path}")


def main():
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

    X, y = generate_gaussian_cluster_data(CONFIG)

    data_path = CONFIG["output_dir"] / CONFIG["data_filename"]
    save_data(X, y, CONFIG, data_path)

    viz_path = CONFIG["output_dir"] / CONFIG["viz_filename"]
    visualize_data(X, y, CONFIG, viz_path)


if __name__ == "__main__":
    main()
