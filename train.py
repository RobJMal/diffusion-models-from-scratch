import argparse
import os

import torch

from diffusion.config import Config
from diffusion.data import generate_swiss_roll_2d_dataset
from diffusion.diffusion_process import GaussianDiffusion
from diffusion.metrics import write_metrics
from diffusion.models import MLPDenoiser
from diffusion.naming import generate_unique_run_name
from diffusion.train import train
from diffusion.utils import set_seed
from diffusion.viz import save_loss_curve

RUNS_DIR = "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a diffusion denoiser network and save it as a new run.")
    parser.add_argument("--config", type=str, default="configs/default.json", help="Path to a config JSON file")
    parser.add_argument("--name", type=str, default=None, help="Run name (default: <model-class>_adjective-noun-42)")
    parser.add_argument("--epochs", type=int, default=None, help="Override the number of training epochs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = Config.load(args.config)
    if args.epochs is not None:
        config.epochs = args.epochs

    set_seed(config.seed)

    x_data = generate_swiss_roll_2d_dataset(config.n_samples)
    diffusion = GaussianDiffusion(config.timesteps, config.beta_start, config.beta_end)
    model = MLPDenoiser(
        time_dim=config.time_dim,
        coordinate_dim=config.coordinate_dim,
        inner_layer_dim=config.inner_layer_dim,
    )

    model_class_name = model.__class__.__name__.lower()
    run_name = args.name or generate_unique_run_name(RUNS_DIR, prefix=model_class_name)
    run_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run: {run_name} ({run_dir})")

    loss_history, metrics = train(model, diffusion, x_data, config)

    config.save(os.path.join(run_dir, "config.json"))
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    save_loss_curve(loss_history, os.path.join(run_dir, "loss_curve.png"))
    write_metrics(os.path.join(run_dir, "metrics.json"), "training", metrics)

    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
