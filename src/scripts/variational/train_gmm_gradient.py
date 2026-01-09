import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import time

from generative_modeling.variational import GMM, gmm_utils

CONFIG = {
    "data_path": Path("data/variational/gaussian_cluster_data.npz"),
    "output_dir": Path("out/variational/gradient"),
    "seed": 42,
    "n_components": 3,
    "learning_rates": [0.1, 0.01],
    "n_iterations_list": [500, 2000],
    "save_interval": 5,
}


def train_gradient_ascent(model, X, X_np, y, learning_rate, n_iterations, save_interval, output_dir, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    log_likelihoods = []
    
    pbar = tqdm(range(n_iterations), desc=f"Gradient Ascent (lr={learning_rate})")
    for iteration in pbar:
        with torch.no_grad():
            ll = model.log_likelihood(X).item()
        log_likelihoods.append(ll)
        
        if iteration % save_interval == 0:
            gmm_utils.save_iteration_checkpoint(model, X_np, y, iteration, ll, output_dir, config)
        
        optimizer.zero_grad()
        loss = -model.log_likelihood(X)
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix({"log_likelihood": f"{ll:.2f}"})
    
    return np.array(log_likelihoods)


def main():
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    
    X, y, _, _, _ = gmm_utils.load_data(CONFIG["data_path"])
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    for lr, n_iters in zip(CONFIG["learning_rates"], CONFIG["n_iterations_list"]):
        output_dir = CONFIG["output_dir"] / f"lr_{lr}_iters_{n_iters}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model = GMM(n_components=CONFIG["n_components"], n_features=X.shape[1])
        model.initialize(X_tensor, seed=CONFIG["seed"])
        
        start_time = time.time()
        log_likelihoods = train_gradient_ascent(
            model, X_tensor, X, y, lr, n_iters, 
            CONFIG["save_interval"], output_dir, CONFIG
        )
        train_time = time.time() - start_time
        
        print(f"\nTraining time: {train_time:.2f}s")
        print(f"Final log likelihood: {log_likelihoods[-1]:.2f}")
        
        gmm_utils.visualize_convergence(
            log_likelihoods, output_dir, 
            f"Gradient Ascent Convergence (lr={lr})", "#1f77b4"
        )
        gmm_utils.save_results(model, log_likelihoods, train_time, output_dir, CONFIG)
        gmm_utils.create_animation(output_dir)
        
        print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
