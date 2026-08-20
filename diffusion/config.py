from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 42

    # Dataset
    n_samples: int = 2000

    # Noise schedule
    timesteps: int = 100
    beta_start: float = 0.001
    beta_end: float = 0.2

    # Model architecture
    coordinate_dim: int = 2
    time_dim: int = 16
    inner_layer_dim: int = 256

    # Training
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 10000

    # Sampling
    num_generate_samples: int = 1000

    # Output paths
    checkpoint_path: str = "outputs/diffusion_model.pt"
    png_path: str = "outputs/diffusion_final_result.png"
    mp4_path: str = "outputs/diffusion_swiss_roll.mp4"
