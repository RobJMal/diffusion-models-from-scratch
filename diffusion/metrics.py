import json
import os

import numpy as np
import torch.nn as nn
from scipy.stats import wasserstein_distance


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_forward_flops(model: nn.Module, batch_size: int = 1) -> int:
    """Rough FLOP estimate for a forward pass: 2 * in_features * out_features
    (one multiply + one add per weight) for every nn.Linear layer."""
    flops_per_sample = sum(
        2 * module.in_features * module.out_features
        for module in model.modules()
        if isinstance(module, nn.Linear)
    )
    return flops_per_sample * batch_size


def wasserstein_distance_2d(real: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    """Per-axis 1D Wasserstein (earth mover's) distance between the real and
    generated point clouds — a simple proxy for how well the generated
    distribution matches the target, since a true 2D EMD needs an optimal
    transport solver we don't otherwise depend on."""
    return {
        "x": wasserstein_distance(real[:, 0], generated[:, 0]),
        "y": wasserstein_distance(real[:, 1], generated[:, 1]),
    }


def write_metrics(path: str, section: str, values: dict) -> None:
    """Merge `values` into `section` of the metrics JSON at `path`, preserving
    other sections (e.g. training vs. generation metrics written by separate scripts)."""
    existing = {}
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    existing[section] = values
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
