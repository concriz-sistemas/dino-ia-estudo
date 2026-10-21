"""
scale.py — Making the numbers of the GPU deep dive easier to picture.

Run from the repository root, after benchmark.py (a few seconds):
    python deep-dives/gpu/scale.py

Outputs (in charts/):
    04_by_hand.png   — how long each training would take a person doing one multiply-add per second
    05_how_long.png  — how long training a large LLM would take, from by hand to 10,000 data-center GPUs
    fill.gif         — a few chefs vs many helpers, filling the result table of a matrix multiplication
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOLDER = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(FOLDER, "charts")

# Operations needed by each training (counted in the README, section 4)
TRAININGS = [
    ("Lesson 1: 25-weight network", 2.6e8),
    ("Lesson 4: PPO agent", 3.3e11),
    ("Lesson 7: tiny transformer", 5.5e12),
    ("Lesson 6: CNN eyes", 7.6e13),
    ("A large LLM (70 billion weights)", 6.3e24),
]

SECOND, MINUTE, HOUR, DAY, YEAR = 1, 60, 3600, 86400, 3.15e7
AGE_OF_UNIVERSE = 13.8e9 * YEAR


def human(seconds):
    """Turns seconds into something a person can picture."""
    if seconds < 1e-3:
        return "under a millisecond"
    if seconds < 1:
        return f"{seconds * 1000:.0f} milliseconds"
    for unit, size in [("seconds", MINUTE), ("minutes", HOUR), ("hours", DAY), ("days", YEAR)]:
        if seconds < size:
            div = {"seconds": SECOND, "minutes": MINUTE, "hours": HOUR, "days": DAY}[unit]
            return f"{seconds / div:.0f} {unit}" if seconds / div >= 2 else f"{seconds / div:.1f} {unit}"
    years = seconds / YEAR
    if seconds > AGE_OF_UNIVERSE:
        return f"{round_2(seconds / AGE_OF_UNIVERSE):,.0f} × the age of the universe"
    if years >= 1e6:
        return f"{round_2(years / 1e6):,} million years"
    return f"{round_2(years):,.0f} years"


def round_2(x):
    """Keeps 2 significant digits: 10,476 -> 10,000 and 174,603 -> 170,000. Exact digits would be fake precision."""
    if x < 10:
        return round(x, 1)
    if x < 100:
        return round(x)
    digits = int(np.floor(np.log10(x))) - 1
    return round(x / 10 ** digits) * 10 ** digits


def bar_chart(ax, values, labels, title, xlabel):
    y = np.arange(len(values))[::-1]
    ax.barh(y, values, color="#1c7ed6")
    for yi, v, lab in zip(y, values, labels):
        ax.text(v * 1.6, yi, lab, va="center", fontsize=9)
    ax.set_yticks(y, [name for name, _ in TRAININGS], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(min(values) / 10, max(values) * 1e5)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, axis="x")


def main():
    with open(os.path.join(OUTPUT, "benchmark.json")) as f:
        bench = json.load(f)
    best = {d: max(v["ops_per_second"] for v in r.values()) for d, r in bench["matmul"].items()}
    cpu, gpu = best["cpu"], best.get(bench["gpu"], best["cpu"])
    # One current data-center GPU: about 1e15 operations per second at its peak in training
    # number formats; in practice training runs well below the peak, here assumed 40%.
    datacenter = 1e15 * 0.4

    # 1) by hand: one multiply-add per second = 2 operations per second
    seconds = [ops / 2 for _, ops in TRAININGS]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    bar_chart(ax, seconds, [human(s) for s in seconds],
              "If a person did the training by hand, one multiply-add per second, without stopping",
              "seconds (log scale: each tick is 1,000× the one before)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "04_by_hand.png"), dpi=110)
    plt.close(fig)

    # 2) the large LLM on each machine, on ONE shared scale
    machines = [("a person, by hand (1 multiply-add per second)", 2.0),
                ("this Mac's CPU (measured)", cpu),
                ("this Mac's GPU (measured)", gpu),
                ("one data-center GPU (assumed: 40% of ~1e15)", datacenter),
                ("10,000 data-center GPUs", datacenter * 10_000)]
    llm_ops = TRAININGS[-1][1]
    times = [llm_ops / speed for _, speed in machines]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    y = np.arange(len(machines))[::-1]
    ax.barh(y, times, color=["#868e96", "#adb5bd", "#e8590c", "#1c7ed6", "#0b4f9c"])
    for yi, t in zip(y, times):
        ax.text(t * 2, yi, human(t), va="center", fontsize=9)
    ax.set_yticks(y, [name for name, _ in machines], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(DAY / 10, max(times) * 1e7)
    for label, sec in [("1 day", DAY), ("1 year", YEAR), ("age of the universe", AGE_OF_UNIVERSE)]:
        ax.axvline(sec, c="k", ls=":", lw=0.8)
        ax.text(sec, len(machines) - 0.45, " " + label, fontsize=7, va="bottom")
    ax.set_xlabel("seconds (log scale)")
    ax.set_title("Training a large LLM (70 billion weights, 15 trillion tokens): ideal times, real ones are longer",
                 fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "05_how_long.png"), dpi=110)
    plt.close(fig)

    # 3) the animation: filling a 16 × 16 result table
    from matplotlib.animation import FuncAnimation, PillowWriter

    n = 16
    workers = {"A few chefs (4 at a time)": 4, "Many helpers (64 at a time)": 64}
    order = np.random.default_rng(0).permutation(n * n)
    steps = int(np.ceil(n * n / min(workers.values())))
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.6))

    def draw(t):
        for ax, (name, k) in zip(axs, workers.items()):
            done = min(n * n, k * t)
            grid = np.zeros(n * n)
            grid[order[:done]] = 1
            grid[order[max(0, done - k):done]] = 2  # the cells being computed right now
            ax.clear()
            ax.imshow(grid.reshape(n, n), cmap=matplotlib.colors.ListedColormap(["#f1f3f5", "#1c7ed6", "#e8590c"]),
                      vmin=0, vmax=2)
            ax.set_xticks([])
            ax.set_yticks([])
            status = "done!" if done == n * n else f"{done} of {n * n} cells"
            ax.set_title(f"{name}\nstep {t}: {status}", fontsize=10)
        fig.suptitle("Filling the result table of a matrix multiplication (each cell = one row × one column)",
                     fontsize=10)

    frames = list(range(steps + 1)) + [steps] * 10
    FuncAnimation(fig, draw, frames=frames).save(os.path.join(OUTPUT, "fill.gif"), writer=PillowWriter(fps=6), dpi=70)
    plt.close(fig)

    print(f"Best measured: CPU {cpu:.2e}, GPU {gpu:.2e} operations per second")
    for name, ops in TRAININGS:
        print(f"  {name:<34} by hand: {human(ops / 2):<32} this Mac's GPU: {human(ops / gpu):<22} "
              f"10,000 data-center GPUs: {human(ops / (datacenter * 10_000))}")
    print(f"\nCharts in: {OUTPUT}")


if __name__ == "__main__":
    main()
