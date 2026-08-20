import time

import torch
import torch.nn as nn

from diffusion.config import Config
from diffusion.diffusion_process import GaussianDiffusion
from diffusion.metrics import count_parameters, estimate_forward_flops
from diffusion.models import MLPDenoiser


def train(
    model: MLPDenoiser, diffusion: GaussianDiffusion, x_data: torch.Tensor, config: Config
) -> tuple[list[float], dict]:
    n_params = count_parameters(model)
    flops = estimate_forward_flops(model, batch_size=config.batch_size)
    print(f"Model: {n_params:,} trainable params, ~{flops / 1e6:.2f} MFLOPs/forward pass (batch={config.batch_size})")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    n_samples = x_data.shape[0]
    loss_history: list[float] = []

    start_time = time.perf_counter()
    for epoch in range(config.epochs):
        # Sample random mini-batch from the dataset
        idx = torch.randint(0, n_samples, (config.batch_size,))
        x0 = x_data[idx]

        # Sample random timesteps t for each sample in the batch
        t = torch.randint(0, diffusion.timesteps, (config.batch_size,))

        # Sample random Gaussian noise and compute noisy x_t at timestep t
        epsilon = torch.randn_like(x0)
        x_t = diffusion.q_sample(x0, t, epsilon)

        # Predict noise and compute loss
        pred_epsilon = model(x_t, t)
        loss = nn.functional.mse_loss(pred_epsilon, epsilon)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{config.epochs} | Loss: {loss.item():.5f}")
    elapsed = time.perf_counter() - start_time

    epochs_per_sec = config.epochs / elapsed
    samples_per_sec = (config.epochs * config.batch_size) / elapsed
    print(f"Training took {elapsed:.1f}s ({epochs_per_sec:.1f} epochs/s, {samples_per_sec:,.0f} samples/s)")

    metrics = {
        "trainable_params": n_params,
        "flops_per_forward_pass": flops,
        "training_time_sec": elapsed,
        "epochs_per_sec": epochs_per_sec,
        "samples_per_sec": samples_per_sec,
        "final_loss": loss_history[-1],
    }
    return loss_history, metrics
