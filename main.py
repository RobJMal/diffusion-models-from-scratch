from typing import Any

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ---- STEP 1 ----
# Create 2D Swiss roll dataset
N_SAMPLES = 2000
theta = np.linspace(0, 3*np.pi, N_SAMPLES)
r = theta

x = r * np.cos(theta)
y = r * np.sin(theta)

X = np.stack([x, y], axis=1)    # bundles the points 
# Standardize to have mean 0 and std 1 (z-score standardization)
X = (X - X.mean(axis=0)) / X.std(axis=0)
X = torch.tensor(X, dtype=torch.float32)

# plt.scatter(X[:, 0], X[:, 1], s=5, alpha=0.5)
# plt.title("Target 2D distribution (Swiss Roll)")
# plt.show()

# ---- STEP 2 ----
# Define noise schedule pre-computations
T_NOISE_STEPS = 100
# Since we add noise backwrads, the noise starts large near T 
# (beta_end) and becomes smaller towards T=0 (beta_start)
beta_start = 0.001
beta_end = 0.2

# Linear schedule for beta
betas = torch.linspace(beta_start, beta_end, T)
alphas = 1.0 - betas    # <T, 1>, fraction of original signal kept at step t
alpha_bars = torch.cumprod(alphas, dim=0)   # measures how much of original data x0 survives after t consecutive steps

# Pre-calculate sqrts for close-form forward sampling equation
sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

# ---- STEP 3: Denoising MLP ----
class ToyDenoiser(nn.Module):
    def __init__(self, time_dim=16) -> None:
        super().__init__()

        coordinate_dim = 2  # 2D
        inner_layer_dim = 128

        # Embed single int timestep t into small vector
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim), #<1, 1> -> <time_dim, 1>
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)   # <time_dim, 1> -> <time_dim, 1>
        )

        # Main network takes 2D coordinate + time_dim embedding
        self.net = nn.Sequential(
            nn.Linear(coordinate_dim + time_dim, inner_layer_dim),
            nn.SiLU,
            nn.Linear(inner_layer_dim, inner_layer_dim),
            nn.SiLU(),
            nn.Linear(inner_layer_dim, coordinate_dim)   # Predicts 2D noise vector (?)
        )

    def forward(self, x_t, t):
        # Normalize t to range [-1, 1] for better neural net stability
        t_norm = (t.float() / T_NOISE_STEPS).unsqueeze(1) * 2.0 - 1.0
        t_emb = self.time_embed(t_norm)

        input_feat = torch.cat([x_t, t_emb], dim=1)
        return self.net(input_feat)

diffusion_model = ToyDenoiser()
