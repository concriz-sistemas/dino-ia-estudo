"""
loss_valley.py — Gradient descent seen from above, on a map of the loss.

The network in train_from_scratch.py has 25 weights, so its loss lives in
25 dimensions: impossible to draw. Here we shrink the problem until the
whole loss fits in a picture: the simplest neuron there is, with no
activation at all. That is exactly a LINEAR REGRESSION:

    jump distance = w · speed + b

The task still comes from the dino game: the faster the game, the earlier
(farther from the cactus) you have to jump. We have 200 noisy examples of
"good jump distance" and want the line that fits them best.

Each point (w, b) of the map is one possible line; its color is the loss
(mean squared error) of that line. Training is walking downhill on this
map. We walk it four times, changing only the learning rate:

    too small     — safe, but does not reach the bottom in time
    good          — goes down and settles at the bottom
    too big       — overshoots, bounces from side to side, and blows up
    cosine        — starts with the same "too big" rate and shrinks it
                    along half a cosine wave (a "cosine schedule",
                    common when training LLMs)

Run from the repository root:
    python lessons/01-neuron-and-gradient/loss_valley.py

Outputs (in results/):
    07_loss_valley.png  — the four paths, the loss per step and the learning rate per step
    loss_valley.gif     — the same paths, step by step
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

FOLDER = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(FOLDER, "results")
os.makedirs(OUTPUT, exist_ok=True)

SEED = 0
N_EXAMPLES = 200
STEPS = 60
START = (-1.0, 1.0)  # (w, b) where all four walks begin: a bad line

# name: (learning rate at the start, use the cosine schedule?, color)
WALKS = {
    "too small (0.05)": (0.05, False, "#868e96"),
    "good (0.5)": (0.5, False, "#2f9e44"),
    "too big (0.78)": (0.78, False, "#e03131"),
    "cosine (0.78 → 0)": (0.78, True, "#7048e8"),
}

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Data: the good jump distance grows with the speed, plus some noise
# ---------------------------------------------------------------------------
def make_data(n):
    speed = rng.uniform(0, 1, n)
    jump_distance = 0.15 + 0.25 * speed + rng.normal(0, 0.03, n)
    return speed, jump_distance


def loss(w, b, x, y):
    """Mean squared error. Works on whole grids of (w, b) at once."""
    prediction = np.multiply.outer(w, x) + np.expand_dims(b, -1)
    return np.mean((prediction - y) ** 2, axis=-1)


def gradient(w, b, x, y):
    error = w * x + b - y
    return 2 * np.mean(error * x), 2 * np.mean(error)


def learning_rate(start, step, cosine):
    if not cosine:
        return start
    # Cosine schedule: half a cosine wave, from `start` at step 0 down to 0 at the last step
    return start * 0.5 * (1 + np.cos(np.pi * step / STEPS))


def walk(x, y, start_rate, cosine):
    """Gradient descent from START. Returns the path, the loss and the learning rate at each step."""
    w, b = START
    path, rates = [(w, b)], []
    for step in range(STEPS):
        rate = learning_rate(start_rate, step, cosine)
        gw, gb = gradient(w, b, x, y)
        w, b = w - rate * gw, b - rate * gb   # the same update as always: weight -= rate × gradient
        path.append((w, b))
        rates.append(rate)
    path = np.array(path)
    return path, loss(path[:, 0], path[:, 1], x, y), np.array(rates)


def main():
    x, y = make_data(N_EXAMPLES)
    walks = {name: walk(x, y, rate, cosine) for name, (rate, cosine, _) in WALKS.items()}

    # The map: the loss of every line (w, b) in a grid
    W, B = np.meshgrid(np.linspace(-1.6, 1.6, 200), np.linspace(-1.2, 1.8, 200))
    L = loss(W, B, x, y)
    best = np.unravel_index(L.argmin(), L.shape)

    fig = plt.figure(figsize=(14, 5.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.3, 1])
    ax_map = fig.add_subplot(grid[:, 0])
    ax_loss = fig.add_subplot(grid[0, 1])
    ax_rate = fig.add_subplot(grid[1, 1])

    # Log scale on the colors: otherwise the whole bottom of the bowl looks the same
    levels = np.geomspace(L.min(), L.max(), 25)
    ax_map.contourf(W, B, L, levels=levels, cmap="Blues_r", norm=matplotlib.colors.LogNorm())
    ax_map.contour(W, B, L, levels=levels[::2], colors="white", linewidths=0.4, alpha=0.6)
    ax_map.scatter(W[best], B[best], marker="*", s=240, c="gold", edgecolors="k", zorder=5,
                   label="bottom of the valley (the best line)")
    ax_map.scatter(*START, marker="o", s=60, c="k", zorder=5, label="start")
    ax_map.set_xlabel("w (how much the speed matters)")
    ax_map.set_ylabel("b (bias)")
    ax_map.set_title("The loss of every possible line (darker = smaller loss)")

    paths, dots, loss_lines, rate_lines = [], [], [], []
    for name, (_, _, color) in WALKS.items():
        paths.append(ax_map.plot([], [], c=color, lw=1.6, marker=".", ms=4, label=name)[0])
        dots.append(ax_map.plot([], [], "o", c=color, ms=7)[0])
        loss_lines.append(ax_loss.plot([], [], c=color, lw=1.6)[0])
        rate_lines.append(ax_rate.plot([], [], c=color, lw=1.6)[0])
    ax_map.legend(loc="lower left", fontsize=8)
    ax_map.set_xlim(W.min(), W.max())
    ax_map.set_ylim(B.min(), B.max())

    ax_loss.set_xlim(0, STEPS)
    ax_loss.set_yscale("log")
    ax_loss.set_ylim(L.min() * 0.7, 10)
    ax_loss.axhline(L.min(), c="gold", lw=1, ls="--")
    ax_loss.set_title("Loss at each step (log scale; dashed = bottom of the valley)", fontsize=10)
    ax_loss.grid(alpha=0.3)
    ax_rate.set_xlim(0, STEPS)
    ax_rate.set_ylim(0, 0.9)
    ax_rate.set_title("Learning rate at each step", fontsize=10)
    ax_rate.set_xlabel("step")
    ax_rate.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    def draw(step):
        for (path, losses, rates), p, d, ll, rl in zip(walks.values(), paths, dots, loss_lines, rate_lines):
            p.set_data(path[:step + 1, 0], path[:step + 1, 1])
            d.set_data([path[step, 0]], [path[step, 1]])
            ll.set_data(range(step + 1), losses[:step + 1])
            rl.set_data(range(min(step, STEPS - 1) + 1), rates[:step + 1])
        fig.suptitle(f"Gradient descent on a linear regression — step {step} of {STEPS}")

    draw(STEPS)
    fig.savefig(os.path.join(OUTPUT, "07_loss_valley.png"), dpi=110)

    frames = list(range(STEPS + 1)) + [STEPS] * 15
    FuncAnimation(fig, draw, frames=frames).save(os.path.join(OUTPUT, "loss_valley.gif"),
                                                 writer=PillowWriter(fps=6), dpi=70)
    plt.close(fig)

    print(f"Bottom of the valley: w = {W[best]:.3f}, b = {B[best]:.3f}, loss = {L.min():.5f}")
    for name, (path, losses, _) in walks.items():
        w, b = path[-1]
        print(f"  {name:<18} after {STEPS} steps: w = {w:7.3f}, b = {b:7.3f}, loss = {losses[-1]:.5f}")
    print(f"Charts in: {OUTPUT}")


if __name__ == "__main__":
    main()
