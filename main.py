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
T = 100
beta_start = 0.001
beta_end = 0.2

# Linear schedule for beta
betas = torch.linspace(beta_start, beta_end, T)
alphas = 1.0 - betas    # <T, 1>
alpha_bars = torch.cumprod(alphas, dim=0)

# Pre-calculate sqrts for close-form forward sampling equation
sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

