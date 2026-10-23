"""
memory.py — Does a model fit in the memory of a computer, and how fast can it write there?

Uses the published memory size and memory speed (bandwidth) of a few computers
and chips to answer two questions:

    1. Does the model fit?   size of the weights = number of weights × bytes per weight
    2. How fast can it write, at most?
       To write ONE token, the model has to read ALL its weights from memory once.
       So the memory speed sets a ceiling:  tokens per second ≤ bandwidth ÷ model size.
       Real speeds are lower than this ceiling.

Run from the repository root (a few seconds):
    python deep-dives/hardware/memory.py

Outputs (in charts/):
    04_does_it_fit.png        — model sizes vs memory of each device
    05_tokens_per_second.png  — the writing-speed ceiling of each device, for two model sizes
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOLDER = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(FOLDER, "charts")
os.makedirs(OUTPUT, exist_ok=True)

# name, kind, memory the model can use (GB), memory bandwidth (GB/s).
DEVICES = [
    ("PC, CPU only (DDR3)", "pc", 16, 26),
    ("PC, CPU only (DDR4)", "pc", 32, 51),
    ("PC, CPU only (DDR5)", "pc", 32, 90),
    ("GeForce RTX 4060", "gaming", 8, 272),
    ("GeForce RTX 4090", "gaming", 24, 1008),
    ("GeForce RTX 5090", "gaming", 32, 1792),
    ("Mac mini M4, 24 GB (this study)", "unified", 18, 120),   # macOS lets the GPU use ~75% by default
    ("Mac with M4 Max, 128 GB", "unified", 96, 546),
    ("AMD Ryzen AI Max+ 395, 128 GB", "unified", 96, 256),  # up to 96 GB can go to the GPU
    ("NVIDIA H100", "datacenter", 80, 3350),
    ("AMD Instinct MI300X", "datacenter", 192, 5300),
    ("NVIDIA B200", "datacenter", 192, 8000),
]
COLORS = {"pc": "#868e96", "gaming": "#2f9e44", "unified": "#e8590c", "datacenter": "#1c7ed6"}
KIND_NAMES = {"pc": "regular PC", "gaming": "gaming graphics card", "unified": "unified memory (CPU + GPU share it)",
              "datacenter": "data-center GPU"}

# Weights in billions, bytes per weight
MODELS = [
    ("7B model, 16 bits", 7, 2),
    ("7B model, 4 bits", 7, 0.5),
    ("70B model, 16 bits", 70, 2),
    ("70B model, 4 bits", 70, 0.5),
    ("training a 7B model", 7, 16),   # weights + gradients + optimizer state, about 16 bytes per weight
]


def main():
    # 1) does it fit?
    fig, ax = plt.subplots(figsize=(12, 5.5))
    y = np.arange(len(DEVICES))[::-1]
    ax.barh(y, [d[2] for d in DEVICES], color=[COLORS[d[1]] for d in DEVICES])
    for yi, d in zip(y, DEVICES):
        ax.text(d[2] + 2, yi, f"{d[2]} GB", va="center", fontsize=8)
    ax.set_yticks(y, [d[0] for d in DEVICES], fontsize=9)
    styles = ["-", "--", "-", "--", ":"]
    for i, ((name, billions, bytes_per), style) in enumerate(zip(MODELS, styles)):
        size = billions * bytes_per
        top = len(DEVICES) - 0.5 + 0.45 * (len(MODELS) - i)   # one row per label, so they don't overlap
        ax.plot([size, size], [-0.7, top], c="k", ls=style, lw=1.2)
        ax.text(size, top, f" {name}: {size:.1f} GB", fontsize=8, va="center")
    ax.set_xlim(0, 215)
    ax.set_ylim(-0.7, len(DEVICES) + 2.2)
    ax.set_xlabel("memory the model can use (GB). A model fits where the bar reaches past its line.")
    ax.set_title("Does the model fit? Memory of each device vs the size of the weights", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
    ax.legend(handles, KIND_NAMES.values(), loc="center right", fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "04_does_it_fit.png"), dpi=110)
    plt.close(fig)

    # 2) the writing-speed ceiling
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (name, billions, bytes_per) in zip(axs, [MODELS[1], MODELS[3]]):
        size = billions * bytes_per
        values, labels = [], []
        for d in DEVICES:
            if d[2] >= size:
                values.append(d[3] / size)
                labels.append(f"{d[3] / size:,.0f} tokens/s")
            else:
                values.append(0)
                labels.append("doesn't fit")
        ax.barh(y, values, color=[COLORS[d[1]] for d in DEVICES])
        for yi, v, lab in zip(y, values, labels):
            ax.text(v + max(values) * 0.01, yi, lab, va="center", fontsize=8,
                    color="#c92a2a" if lab == "doesn't fit" else "k")
        ax.set_yticks(y, [d[0] for d in DEVICES], fontsize=9)
        ax.set_xlim(0, max(values) * 1.3)
        ax.set_title(f"{name} ({size:.1f} GB): tokens per second, at most", fontsize=10)
        ax.set_xlabel("memory bandwidth ÷ model size")
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle("Writing one token means reading all the weights once: memory speed sets the ceiling", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "05_tokens_per_second.png"), dpi=110)
    plt.close(fig)

    print(f"{'device':<28} {'memory':>7} {'bandwidth':>10}   7B 4-bit   70B 4-bit   70B 16-bit")
    for d in DEVICES:
        cells = []
        for _, billions, bytes_per in [MODELS[1], MODELS[3], MODELS[2]]:
            size = billions * bytes_per
            cells.append(f"{d[3] / size:7,.0f}/s" if d[2] >= size else "   no fit")
        print(f"{d[0]:<28} {d[2]:>5} GB {d[3]:>7} GB/s  " + "   ".join(cells))
    print(f"\nCharts in: {OUTPUT}")


if __name__ == "__main__":
    main()
