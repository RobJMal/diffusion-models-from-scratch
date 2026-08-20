import os

import torch
import torch.nn as nn

from diffusion.config import Config
from diffusion.diffusion_process import GaussianDiffusion
from diffusion.models import MLPDenoiser


def _model_config(config: Config) -> dict:
    """The subset of Config that determines MLPDenoiser's parameter shapes.
    Saved alongside checkpoints so a stale checkpoint is detected instead of
    crashing load_state_dict with a shape-mismatch error."""
    return {
        "time_dim": config.time_dim,
        "coordinate_dim": config.coordinate_dim,
        "inner_layer_dim": config.inner_layer_dim,
    }


def get_model(config: Config, diffusion: GaussianDiffusion, x_data: torch.Tensor) -> MLPDenoiser:
    """Load a cached, architecture-matching checkpoint if one exists; otherwise train fresh."""
    model = MLPDenoiser(
        time_dim=config.time_dim,
        coordinate_dim=config.coordinate_dim,
        inner_layer_dim=config.inner_layer_dim,
    )

    if os.path.exists(config.checkpoint_path):
        checkpoint = torch.load(config.checkpoint_path)
        if checkpoint.get("model_config") == _model_config(config):
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded cached model weights from {config.checkpoint_path}")
            return model
        print(f"Checkpoint at {config.checkpoint_path} doesn't match current model config, retraining...")

    train(model, diffusion, x_data, config)
    return model


def train(model: MLPDenoiser, diffusion: GaussianDiffusion, x_data: torch.Tensor, config: Config) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    n_samples = x_data.shape[0]

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

        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{config.epochs} | Loss: {loss.item():.5f}")

    os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "model_config": _model_config(config)},
        config.checkpoint_path,
    )
    print(f"Saved model weights to {config.checkpoint_path}")
