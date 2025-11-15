import torch
import os

class GaussianMixture:
    def __init__(self, n_components=10, z_dim=100, c=5.0, sigma=0.5, seed=42, device='cpu'):
        """
        Gaussian Mixture Model for sampling latent vectors.

        Parameters:
            n_components (int): Number of Gaussian components (K).
            z_dim (int): Dimensionality of latent space (input to generator).
            c (float): Range for component means (sampled uniformly in [-c, c]).
            sigma (float): Standard deviation for all Gaussians.
            seed (int): Random seed for reproducibility.
            device (str): 'cpu' or 'cuda'.
        """
        self.n_components = n_components
        self.z_dim = z_dim
        self.c = c
        self.sigma = sigma
        self.device = device

        torch.manual_seed(seed)

        # Initialize component means μ_k uniformly in [-c, c]
        self.mu = torch.empty(n_components, z_dim).uniform_(-c, c).to(device)
        # Shared covariance (isotropic)
        self.sigma_tensor = sigma * torch.ones(z_dim, device=device)

    def sample(self, batch_size, cluster_ids=None):
        """
        Sample from the Gaussian Mixture Model.

        Returns:
            z (torch.Tensor): Latent tensor of shape (batch_size, z_dim)
        """
        # Choose mixture components for each sample
        if cluster_ids is not None:
            comp_ids = cluster_ids.to(self.device)
        else:
            comp_ids = torch.randint(0, self.n_components, (batch_size,), device=self.device)

        # Get corresponding means and sample noise
        means = self.mu[comp_ids].to(self.device)
        noise = torch.randn(batch_size, self.z_dim, device=self.device) * self.sigma

        return means + noise, comp_ids

    def save(self, filepath):
        """
        Save GMM parameters to disk.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'n_components': self.n_components,
            'z_dim': self.z_dim,
            'c': self.c,
            'sigma': self.sigma,
            'mu': self.mu.detach().cpu(),
        }, filepath)
        print(f"GMM saved to {filepath}")

    @classmethod
    def load(cls, filepath, device='cpu'):
        """
        Load GMM parameters from disk.
        """
        checkpoint = torch.load(filepath, map_location=device)
        gmm = cls(
            n_components=checkpoint['n_components'],
            z_dim=checkpoint['z_dim'],
            c=checkpoint['c'],
            sigma=checkpoint['sigma'],
            device=device
        )
        gmm.mu = checkpoint['mu'].to(device)
        print(f"GMM loaded from {filepath}")
        return gmm