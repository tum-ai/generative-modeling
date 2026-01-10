import numpy as np
import torch
import torch.nn as nn


class GMM(nn.Module):
    def __init__(self, n_components, n_features):
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features

        self.means = nn.Parameter(torch.randn(n_components, n_features))
        self.log_vars = nn.Parameter(torch.zeros(n_components, n_features))
        self.logits = nn.Parameter(torch.zeros(n_components))

    def initialize(self, X, seed=None):
        """initialize with random means and unit variances"""
        if seed is not None:
            np.random.seed(seed)

        X_np = X.cpu().numpy() if torch.is_tensor(X) else X

        random_means = np.random.uniform(
            X_np.min(axis=0),
            X_np.max(axis=0),
            size=(self.n_components, self.n_features),
        )
        self.means.data = torch.tensor(random_means, dtype=torch.float32)

        self.log_vars.data = torch.zeros(
            self.n_components, self.n_features
        )  # isotropic

        self.logits.data = torch.zeros(self.n_components)

    @property
    def weights(self):
        """component weights (prior / posterior)"""
        return torch.softmax(self.logits, dim=0)

    @property
    def variances(self):
        """component variances"""
        return torch.exp(self.log_vars)

    def log_prob(self, X):
        """log marginal likelihood for each point in X"""
        X_expanded = X.unsqueeze(1)  # (N, 1, D)
        means_expanded = self.means.unsqueeze(0)  # (1, K, D)
        vars_expanded = self.variances.unsqueeze(0)  # (1, K, D)

        diff = X_expanded - means_expanded  # (N, K, D)
        log_det = torch.sum(self.log_vars, dim=1)  # (K,)
        squared_distance = torch.sum((diff**2) / vars_expanded, dim=2)  # (N, K)

        log_gaussian = -0.5 * (
            self.n_features * np.log(2 * np.pi) + log_det + squared_distance
        )  # (N, K)

        log_weights = torch.log_softmax(self.logits, dim=0)  # (K,)
        log_weighted = log_gaussian + log_weights  # (N, K)

        return torch.logsumexp(log_weighted, dim=1)  # (N,)

    def log_likelihood(self, X):
        """summed log marginal likelihood of all data"""
        return torch.sum(self.log_prob(X))

    def e_step(self, X):
        """compute posterior probabilities for each sample"""
        X_expanded = X.unsqueeze(1) # (N, 1, D)
        means_expanded = self.means.unsqueeze(0) # (1, K, D)
        vars_expanded = self.variances.unsqueeze(0) # (1, K, D)

        diff = X_expanded - means_expanded # (N, K, D)
        log_det = torch.sum(self.log_vars, dim=1) # (K,)
        squared_distance = torch.sum((diff**2) / vars_expanded, dim=2) # (N, K)

        log_gauss = -0.5 * (self.n_features * np.log(2 * np.pi) + log_det + squared_distance) # (N, K)
        log_weights = torch.log_softmax(self.logits, dim=0) # (K,)
        log_resp = log_gauss + log_weights # (N, K)
        log_resp = log_resp - torch.logsumexp(log_resp, dim=1, keepdim=True) # (N, K)

        return torch.exp(log_resp) # (N, K)

    def m_step(self, X, responsibilities):
        """update parameters using posterior probabilities"""
        N = X.shape[0]

        N_k = torch.sum(responsibilities, dim=0) + 1e-8 # (K,)

        self.means.data = torch.sum(
            responsibilities.unsqueeze(2) * X.unsqueeze(1), dim=0
        ) / N_k.unsqueeze(1) # (K, D)

        diff = X.unsqueeze(1) - self.means.unsqueeze(0)
        weighted_var = torch.sum(
            responsibilities.unsqueeze(2) * (diff**2), dim=0
        ) / N_k.unsqueeze(1) # (K, D)
        self.log_vars.data = torch.log(weighted_var + 1e-6)

        self.logits.data = torch.log(N_k / N) # (K,)

    def log_joint_prob(self, X, Z):
        """compute log joint (= log unnormalized posterior) for a batch of samples"""
        X_expanded = X.unsqueeze(1)  # (N, 1, D)
        means_expanded = self.means.unsqueeze(0)  # (1, K, D)
        vars_expanded = self.variances.unsqueeze(0)  # (1, K, D)
        
        diff = X_expanded - means_expanded  # (N, K, D)
        log_det = torch.sum(self.log_vars, dim=1)  # (K,)
        squared_distance = torch.sum((diff ** 2) / vars_expanded, dim=2)  # (N, K)
        
        log_gaussian = -0.5 * (
            self.n_features * np.log(2 * np.pi) + log_det + squared_distance
        )  # (N, K)
        
        log_weights = torch.log_softmax(self.logits, dim=0)  # (K,)
        
        log_p_x_given_z = torch.sum(Z * log_gaussian, dim=1)  # (N,)
        log_p_z = torch.sum(Z * log_weights, dim=1)  # (N,)
        
        return log_p_x_given_z + log_p_z  # (N,)
    
    def sample_posterior_mcmc(self, x, n_samples=10, burn_in=5, proposal_std=0.5):
        """sample from posterior using metropolis hastings mcmc"""
        with torch.no_grad():
            samples = []
            
            # initialize random z
            current_z = torch.zeros(self.n_components)
            current_z[np.random.randint(self.n_components)] = 1.0
            current_log_prob = self.log_joint_prob(x.unsqueeze(0), current_z.unsqueeze(0))[0]
            
            total_steps = burn_in + n_samples
            for step in range(total_steps):
                # propose around current
                proposed_z = torch.zeros(self.n_components)
                proposed_z[np.random.randint(self.n_components)] = 1.0
                proposed_log_prob = self.log_joint_prob(x.unsqueeze(0), proposed_z.unsqueeze(0))[0]
                
                # accep or reject
                log_alpha = proposed_log_prob - current_log_prob
                if torch.log(torch.rand(1)) < log_alpha:
                    current_z = proposed_z
                    current_log_prob = proposed_log_prob
                
                # collect samples
                if step >= burn_in:
                    samples.append(current_z.clone())
            
            return torch.stack(samples)  # (n_samples, K)
