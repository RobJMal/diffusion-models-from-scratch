import matplotlib.pyplot as plt

from diffusion.config import Config
from diffusion.data import make_swiss_roll
from diffusion.diffusion_process import GaussianDiffusion
from diffusion.train import get_model
from diffusion.utils import set_seed
from diffusion.viz import save_animation, save_final_png


def main() -> None:
    config = Config()
    set_seed(config.seed)

    x_data = make_swiss_roll(config.n_samples)
    diffusion = GaussianDiffusion(config.timesteps, config.beta_start, config.beta_end)
    model = get_model(config, diffusion, x_data)

    generated_points, trajectory = diffusion.generate_samples(
        model, config.num_generate_samples, config.coordinate_dim
    )

    save_final_png(generated_points, config.png_path)
    save_animation(trajectory, config.timesteps, config.mp4_path)

    plt.show()


if __name__ == "__main__":
    main()
