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
        """E-step: compute responsibilities (posterior probabilities over components)."""
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
        """M-step: update parameters using responsibilities from E-step."""
        N = X.shape[0]

        N_k = torch.sum(responsibilities, dim=0) + 1e-8

        self.means.data = torch.sum(
            responsibilities.unsqueeze(2) * X.unsqueeze(1), dim=0
        ) / N_k.unsqueeze(1)

        diff = X.unsqueeze(1) - self.means.unsqueeze(0)
        weighted_var = torch.sum(
            responsibilities.unsqueeze(2) * (diff**2), dim=0
        ) / N_k.unsqueeze(1)
        self.log_vars.data = torch.log(weighted_var + 1e-6)

        self.logits.data = torch.log(N_k / N)
