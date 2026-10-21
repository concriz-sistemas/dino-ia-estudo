"""
benchmark.py — CPU vs GPU, measured on your own computer.

Multiplies two square matrices of many sizes, first on the CPU, then on the GPU
(Apple's "mps" on a Mac, "cuda" on an NVIDIA card), and times each one.
Multiplying two n × n matrices takes n³ multiplications and n³ additions,
so we can also say how many operations per second each one does.

Run from the repository root (about 1 minute):
    python deep-dives/gpu/benchmark.py

Outputs (in charts/):
    01_time.png        — how long one multiplication takes, by matrix size
    02_speed.png       — operations per second, by matrix size
    03_tiny_steps.png  — many tiny multiplications in a row (like a small network): who wins?
    benchmark.json
"""
import json
import os
import platform
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

FOLDER = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(FOLDER, "charts")
os.makedirs(OUTPUT, exist_ok=True)

GPU = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else None)
SIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096]


def wait(device):
    """GPU work runs in the background; wait for it to finish before stopping the clock."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def time_matmul(n, device, repeats):
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    a @ b  # warm-up: the first call also prepares things
    wait(device)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        a @ b
        wait(device)
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def time_tiny_steps(device, steps=2000, width=32, batch=64):
    """A tiny network's worth of work: many small multiplications, one after another."""
    x = torch.randn(batch, width, device=device)
    w = torch.randn(width, width, device=device) * 0.1
    for _ in range(20):
        x = torch.tanh(x @ w)
    wait(device)
    start = time.perf_counter()
    for _ in range(steps):
        x = torch.tanh(x @ w)
    wait(device)
    return time.perf_counter() - start


def main():
    if GPU is None:
        print("No GPU that PyTorch can use was found: only the CPU will be measured.")
    devices = ["cpu"] + ([GPU] if GPU else [])
    print(f"CPU: {platform.processor() or platform.machine()}, {torch.get_num_threads()} threads.  "
          f"GPU: {GPU or 'none'}\n")
    results = {d: {} for d in devices}
    for n in SIZES:
        repeats = 20 if n <= 1024 else 5
        line = f"  {n:>5} × {n:<5}"
        for d in devices:
            t = time_matmul(n, d, repeats)
            ops = 2 * n ** 3
            results[d][n] = {"seconds": t, "ops_per_second": ops / t}
            line += f"   {d}: {t * 1000:9.3f} ms ({ops / t / 1e9:8.1f} billion operations/s)"
        print(line)

    tiny = {d: time_tiny_steps(d) for d in devices}
    print("\n2,000 tiny multiplications in a row (64 × 32 times 32 × 32):")
    for d, t in tiny.items():
        print(f"  {d}: {t * 1000:.0f} ms")

    with open(os.path.join(OUTPUT, "benchmark.json"), "w") as f:
        json.dump({"gpu": GPU, "matmul": {d: {str(n): v for n, v in r.items()} for d, r in results.items()},
                   "tiny_steps_seconds": tiny}, f, indent=1)

    colors = {"cpu": "#868e96", GPU: "#e8590c"}
    for name, key, ylabel, title in [
        ("01_time.png", "seconds", "time for one multiplication (seconds, log scale)",
         "Multiplying two n × n matrices: time"),
        ("02_speed.png", "ops_per_second", "operations per second (log scale)",
         "Multiplying two n × n matrices: speed"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for d in devices:
            ax.plot(SIZES, [results[d][n][key] for n in SIZES], "o-", c=colors[d], lw=2,
                    label="CPU" if d == "cpu" else f"GPU ({d})")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(SIZES, [str(n) for n in SIZES])
        ax.set_xlabel("matrix size n")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT, name), dpi=110)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(["CPU" if d == "cpu" else f"GPU ({d})" for d in devices], [tiny[d] * 1000 for d in devices],
           color=[colors[d] for d in devices])
    ax.set_ylabel("milliseconds")
    ax.set_title("2,000 tiny steps in a row: the GPU's overhead shows", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "03_tiny_steps.png"), dpi=110)
    plt.close(fig)
    print(f"\nCharts in: {OUTPUT}")


if __name__ == "__main__":
    main()
