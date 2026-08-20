import argparse
import os

import matplotlib.pyplot as plt
import torch

from diffusion.config import Config
from diffusion.data import generate_swiss_roll_2d_dataset
from diffusion.diffusion_process import GaussianDiffusion
from diffusion.metrics import wasserstein_distance_2d, write_metrics
from diffusion.models import MLPDenoiser
from diffusion.viz import save_animation, save_final_png

RUNS_DIR = "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate samples from a trained diffusion network.")
    parser.add_argument("run_name", type=str, help="Name of a trained run in runs/ (e.g. swift-comet-42)")
    parser.add_argument("--num-samples", type=int, default=None, help="Override the number of samples to generate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = os.path.join(RUNS_DIR, args.run_name)

    config = Config.load(os.path.join(run_dir, "config.json"))

    num_samples = args.num_samples or config.num_generate_samples

    model = MLPDenoiser(
        time_dim=config.time_dim,
        coordinate_dim=config.coordinate_dim,
        inner_layer_dim=config.inner_layer_dim,
    )
    model.load_state_dict(torch.load(os.path.join(run_dir, "model.pt")))

    diffusion = GaussianDiffusion(config.timesteps, config.beta_start, config.beta_end)
    generated_points, trajectory = diffusion.generate_samples(model, num_samples, config.coordinate_dim)

    x_data = generate_swiss_roll_2d_dataset(config.n_samples)
    distances = wasserstein_distance_2d(x_data.numpy(), generated_points)
    print(f"Wasserstein distance (real vs. generated) — x: {distances['x']:.4f}, y: {distances['y']:.4f}")

    write_metrics(
        os.path.join(run_dir, "metrics.json"),
        "generation",
        {
            "num_samples": num_samples,
            "wasserstein_distance_x": float(distances["x"]),
            "wasserstein_distance_y": float(distances["y"]),
        },
    )

    save_final_png(generated_points, os.path.join(run_dir, "diffusion_final_result.png"))
    save_animation(trajectory, config.timesteps, os.path.join(run_dir, "diffusion_swiss_roll.mp4"))

    plt.show()


if __name__ == "__main__":
    main()
