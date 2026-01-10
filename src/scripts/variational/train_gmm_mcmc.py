import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import time

from generative_modeling.variational import GMM, gmm_utils

CONFIG = {
    "data_path": Path("data/variational/gaussian_cluster_data.npz"),
    "output_dir": Path("out/variational/mcmc"),
    "seed": 42,
    "n_components": 3,
    "learning_rate": 0.01,
    "n_iterations": 500,
    "save_interval": 5,
    "mcmc_samples": 10,
    "mcmc_burn_in": 5,
    "batch_size": 10000,
}


def fisher(model: GMM, X, mcmc_samples, mcmc_burn_in):
    """compute a scalar that results in fisher scores when backpropagated"""
    total_score = 0.0
    n_points = X.shape[0]
    
    for i in range(n_points): # we sample for each point!
        x = X[i]  # (D,)

        z_samples = model.sample_posterior_mcmc(
            x, 
            n_samples=mcmc_samples, 
            burn_in=mcmc_burn_in
        )  # (n_samples, K)
        
        x_batch = x.unsqueeze(0).expand(mcmc_samples, -1)  # (n_samples, D)
        log_joints = model.log_joint_prob(x_batch, z_samples)  # (n_samples,)
        
        avg_log_joint = log_joints.mean()
        total_score += avg_log_joint
    
    return total_score / n_points


def train_mcmc(model, X, X_np, y, config, output_dir):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    log_likelihoods = []
    
    pbar = tqdm(range(config["n_iterations"]), desc=f"MCMC-based Training (lr={config['learning_rate']})")
    for iteration in pbar:
        with torch.no_grad():
            ll = model.log_likelihood(X).item()
        log_likelihoods.append(ll)
        
        if iteration % config["save_interval"] == 0:
            gmm_utils.save_iteration_checkpoint(model, X_np, y, iteration, ll, output_dir, config)
        
        optimizer.zero_grad()
        batch_size = min(config["batch_size"], X.shape[0])
        indices = np.random.choice(X.shape[0], size=batch_size, replace=False)
        X_batch = X[indices]
        
        score = fisher(
            model, 
            X_batch, 
            config["mcmc_samples"],
            config["mcmc_burn_in"]
        )
        loss = -score # we want to maximize
        loss = loss * (X.shape[0] / batch_size) # scale to dataset log likelihood
        
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix({"log_likelihood": f"{ll:.2f}"})
    
    return np.array(log_likelihoods)


def main():
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    
    X, y, _, _, _ = gmm_utils.load_data(CONFIG["data_path"])
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    model = GMM(n_components=CONFIG["n_components"], n_features=X.shape[1])
    model.initialize(X_tensor, seed=CONFIG["seed"])
    
    print("Training GMM using Fisher's Identity + MCMC")
    print(f"MCMC samples: {CONFIG['mcmc_samples']}, burn-in: {CONFIG['mcmc_burn_in']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Learning rate: {CONFIG['learning_rate']}, iterations: {CONFIG['n_iterations']}")
    
    start_time = time.time()
    log_likelihoods = train_mcmc(model, X_tensor, X, y, CONFIG, CONFIG["output_dir"])
    train_time = time.time() - start_time
    
    print(f"\nTraining time: {train_time:.2f}s")
    print(f"Final log likelihood: {log_likelihoods[-1]:.2f}")
    
    gmm_utils.visualize_convergence(
        log_likelihoods, 
        CONFIG["output_dir"], 
        "MCMC-based Training Convergence (Fisher's Identity)", 
        "#ff7f0e"
    )
    gmm_utils.save_results(model, log_likelihoods, train_time, CONFIG["output_dir"], CONFIG)
    gmm_utils.create_animation(CONFIG["output_dir"])
    
    print(f"Results saved to {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
