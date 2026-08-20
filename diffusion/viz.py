import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def save_loss_curve(losses: list[float], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.title("Training Loss")
    plt.savefig(path)


def save_final_png(generated_points: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure()
    plt.scatter(generated_points[:, 0], generated_points[:, 1], s=5, c='red', alpha=0.5)
    plt.title("Generated 2D Swiss Roll via Reverse Diffusion")
    plt.savefig(path)


def save_animation(trajectory: list[np.ndarray], timesteps: int, path: str, fps: int = 20) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots()
    scatter = ax.scatter([], [], s=5, c='red', alpha=0.5)

    axis_lim = np.abs(np.stack(trajectory)).max() * 1.1
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    title = ax.set_title("")

    def update(frame_idx):
        points = trajectory[frame_idx]
        scatter.set_offsets(points)
        t_remaining = timesteps - frame_idx
        title.set_text(f"Reverse Diffusion (t={t_remaining})")
        return scatter, title

    anim = FuncAnimation(fig, update, frames=len(trajectory), interval=50, blit=False)
    anim.save(path, writer="ffmpeg", fps=fps)
