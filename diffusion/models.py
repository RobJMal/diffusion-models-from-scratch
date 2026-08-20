import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ToyDenoiser(nn.Module):
    def __init__(self, time_dim=16, coordinate_dim=2, inner_layer_dim=256) -> None:
        super().__init__()

        # Embed single int timestep t into small vector
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Main network takes coordinate + time_dim embedding
        self.net = nn.Sequential(
            nn.Linear(coordinate_dim + time_dim, inner_layer_dim),
            nn.SiLU(),
            nn.Linear(inner_layer_dim, inner_layer_dim),
            nn.SiLU(),
            nn.Linear(inner_layer_dim, inner_layer_dim),
            nn.SiLU(),
            nn.Linear(inner_layer_dim, coordinate_dim)   # Predicts noise vector
        )

    def forward(self, x_t, t):
        t_emb = self.time_embed(t.float())
        input_feat = torch.cat([x_t, t_emb], dim=1)
        return self.net(input_feat)
