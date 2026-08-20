import numpy as np
import torch


def make_swiss_roll(n_samples: int) -> torch.Tensor:
    theta = np.linspace(0, 3 * np.pi, n_samples)
    r = theta

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    X = np.stack([x, y], axis=1)
    # Standardize to have mean 0 and std 1 (z-score standardization)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return torch.tensor(X, dtype=torch.float32)
