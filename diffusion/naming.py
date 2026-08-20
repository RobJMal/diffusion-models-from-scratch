import os
import random

# Decoupled from `random.seed()` in diffusion.utils.set_seed, so run names stay
# unique across runs even when the training seed is fixed for reproducibility.
_rng = random.SystemRandom()

ADJECTIVES = [
    "swift", "lucid", "gentle", "wild", "crimson", "silent", "vivid", "bold",
    "quiet", "radiant", "curious", "brisk", "hazy", "electric", "mellow",
    "nimble", "stellar", "cosmic", "amber", "frosty", "golden", "shy",
    "restless", "serene", "fierce", "dusty", "glowing", "wandering", "quirky", "sunny",
]

NOUNS = [
    "tensor", "photon", "comet", "nebula", "quokka", "fractal", "gradient",
    "neuron", "vector", "otter", "falcon", "glacier", "meadow", "ripple",
    "prism", "canyon", "sparrow", "lattice", "cascade", "ember", "harbor",
    "orbit", "thicket", "beacon", "quasar", "willow", "cipher", "delta", "raven",
]


def generate_run_name(prefix: str | None = None) -> str:
    adjective = _rng.choice(ADJECTIVES)
    noun = _rng.choice(NOUNS)
    suffix = _rng.randint(0, 999)
    name = f"{adjective}-{noun}-{suffix}"
    return f"{prefix}_{name}" if prefix else name


def generate_unique_run_name(runs_dir: str, prefix: str | None = None, max_attempts: int = 20) -> str:
    for _ in range(max_attempts):
        name = generate_run_name(prefix)
        if not os.path.exists(os.path.join(runs_dir, name)):
            return name
    raise RuntimeError(f"Could not generate a unique run name after {max_attempts} attempts")
