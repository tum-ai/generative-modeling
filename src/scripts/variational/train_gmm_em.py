import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import time

from generative_modeling.variational import GMM, gmm_utils

CONFIG = {
    "data_path": Path("data/variational/gaussian_cluster_data.npz"),
    "output_dir": Path("out/variational/em"),
    "seed": 42,
    "n_components": 3,
    "n_iterations": 500,
    "save_interval": 5,
}


def train_em(model, X, X_np, y, config, output_dir):
    log_likelihoods = []
    
    pbar = tqdm(range(config["n_iterations"]), desc="EM Algorithm")
    for iteration in pbar:
        with torch.no_grad():
            ll = model.log_likelihood(X).item()
            log_likelihoods.append(ll)
            
            if iteration % config["save_interval"] == 0:
                gmm_utils.save_iteration_checkpoint(model, X_np, y, iteration, ll, output_dir, config)
            
            responsibilities = model.e_step(X)
            model.m_step(X, responsibilities)
            
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
    
    start_time = time.time()
    log_likelihoods = train_em(model, X_tensor, X, y, CONFIG, CONFIG["output_dir"])
    train_time = time.time() - start_time
    
    print(f"\nTraining time: {train_time:.2f}s")
    print(f"Final log likelihood: {log_likelihoods[-1]:.2f}")
    
    gmm_utils.visualize_convergence(log_likelihoods, CONFIG["output_dir"], 
                                     "EM Algorithm Convergence", "#2ca02c")
    gmm_utils.save_results(model, log_likelihoods, train_time, CONFIG["output_dir"], CONFIG)
    gmm_utils.create_animation(CONFIG["output_dir"])
    
    print(f"Results saved to {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
