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
betas = torch.linspace(beta_start, beta_end, T_NOISE_STEPS)
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
            nn.SiLU(),
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

# ---- STEP 4: Training loop ----
optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=1e-3)
batch_size = 256
epochs = 10000

for epoch in range(epochs):
    # Sample random mini-batch from 2D dataset
    idx = torch.randint(0, N_SAMPLES, (batch_size,))
    x0 = X[idx]

    # Sampling random timesteps t for each sample in batch
    t = torch.randint(0, T_NOISE_STEPS, (batch_size,))

    # Sample random Gaussian noise
    epsilon = torch.randn_like(x0)

    # Compute noisy data points x_t at timestep t
    s_alpha_bar = sqrt_alpha_bars[t].unsqueeze(1)
    s_one_minus_alpha_bar = sqrt_one_minus_alpha_bars[t].unsqueeze(1)
    x_t = s_alpha_bar * x0 + s_one_minus_alpha_bar * epsilon

    # Predict noise and compute loss
    pred_epsilon = diffusion_model(x_t, t)
    loss = nn.functional.mse_loss(pred_epsilon, epsilon)

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.5f}")

# ---- STEP 5: Sampling (Reversing Noise into Data) ----
@torch.no_grad
def generate_samples(num_samples=1000):
    diffusion_model.eval()

    # Start from pure noise x_t
    x_t = torch.randn(num_samples, 2)

    for t_idx in reversed(range(T_NOISE_STEPS)):
        t_tensor = torch.full((num_samples,), t_idx, dtype=torch.long)

        # Predict noise
        pred_noise = diffusion_model(x_t, t_tensor)

        beta_t = betas[t_idx]
        alpha_t = alphas[t_idx]
        alpha_bar_t = alpha_bars[t_idx]

        # Compute mean equation 
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise
        )

        if t_idx > 0:
            # Add small noise z back for stochasticity
            z = torch.rand_like(x_t)
            sigma_t = torch.sqrt(beta_t)
            x_t = mean + sigma_t * z
        else:
            x_t = mean

    return x_t.numpy()

# Run reverse process 
generated_points = generate_samples()

plt.scatter(generated_points[:,0], generated_points[:,1], s=5, c='red', alpha=0.5)
plt.title("Generated 2D Swiss Roll via Reverse Diffusion")
plt.show()
