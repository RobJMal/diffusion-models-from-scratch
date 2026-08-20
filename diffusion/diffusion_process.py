import torch


class GaussianDiffusion:
    """Owns the noise schedule and implements the forward (q_sample) and
    reverse (generate_samples) diffusion processes for a given denoiser model."""

    def __init__(self, timesteps: int, beta_start: float, beta_end: float) -> None:
        self.timesteps = timesteps

        # Since we add noise backwards, the noise starts large near T
        # (beta_end) and becomes smaller towards T=0 (beta_start)
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas    # fraction of original signal kept at step t
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)   # how much of x0 survives after t steps

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward process: sample x_t given x0, timestep t, and noise, in closed form."""
        s_alpha_bar = self.sqrt_alpha_bars[t].unsqueeze(1)
        s_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t].unsqueeze(1)
        return s_alpha_bar * x0 + s_one_minus_alpha_bar * noise

    @torch.no_grad()
    def generate_samples(self, model, num_samples: int, coordinate_dim: int):
        """Reverse process: start from pure noise and iteratively denoise to x0."""
        model.eval()

        x_t = torch.randn(num_samples, coordinate_dim)
        trajectory = [x_t.numpy()]

        for t_idx in reversed(range(self.timesteps)):
            t_tensor = torch.full((num_samples,), t_idx, dtype=torch.long)

            pred_noise = model(x_t, t_tensor)

            beta_t = self.betas[t_idx]
            alpha_t = self.alphas[t_idx]
            alpha_bar_t = self.alpha_bars[t_idx]

            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise
            )

            if t_idx > 0:
                # Add small noise z back for stochasticity
                z = torch.randn_like(x_t)
                sigma_t = torch.sqrt(beta_t)
                x_t = mean + sigma_t * z
            else:
                x_t = mean

            trajectory.append(x_t.numpy())

        return x_t.numpy(), trajectory
