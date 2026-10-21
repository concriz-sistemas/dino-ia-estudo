"""
report.py — Turns the training log into charts and a REPORT.md.

Run from the repository root, after training:
    python lessons/04-reinforcement/report.py

Besides reading the CSVs, it makes each saved model play 10 new matches
and asks each one: "in this situation, what is the chance you jump?".
"""
import csv
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO

FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FOLDER)
from environment import DinoEnv, ACTION_NAMES
import environment
import train

OUTPUT = os.path.join(FOLDER, "results")
MODELS_FOLDER = os.path.join(OUTPUT, "models")
N_TEST_MATCHES = 10


def read_csv(name):
    with open(os.path.join(OUTPUT, name)) as f:
        rows = list(csv.reader(f))
    return rows[0], np.array(rows[1:], dtype=float)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, name), dpi=110)
    plt.close(fig)


def list_models():
    names = sorted(a for a in os.listdir(MODELS_FOLDER) if a.endswith(".zip"))
    return [(int(a[4:9]) * 1000, os.path.join(MODELS_FOLDER, a)) for a in names]


def play(model, seed):
    env = DinoEnv()
    obs, _ = env.reset(seed=seed)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return info["score"]


def probabilities(model, observations):
    """The policy returns one probability for each action. That is what the network 'thinks'."""
    with torch.no_grad():
        obs = torch.as_tensor(observations, dtype=torch.float32)
        return model.policy.get_distribution(obs).distribution.probs.numpy()


# Test situations: the same obstacle at several distances and speeds
DISTANCES = np.linspace(-0.05, 0.6, 66)
SPEEDS = np.linspace(0.4, 1.0, 31)
CACTUS = (20 / 60, 40 / 60, 0.0)
BIRD = (40 / 60, 25 / 60, 35 / 60)      # at the height of the dino's head
ACTION_COLORS = ["#dee2e6", "#e03131", "#1971c2"]  # nothing, jump, duck


def decision_map(model, obstacle):
    """Probability of each action, for the obstacle at every distance and speed."""
    width, height, elevation = obstacle
    D, V = np.meshgrid(DISTANCES, SPEEDS)
    obs = np.column_stack([D.ravel(), np.full(D.size, width), np.full(D.size, height),
                           np.full(D.size, elevation), np.zeros(D.size), V.ravel()])
    return probabilities(model, obs).reshape(*D.shape, 3)


def number(n):
    """1234567 -> '1,234,567'"""
    return f"{int(n):,}"


def moving_average(x, n=50):
    return np.convolve(x, np.ones(n) / n, mode="valid")


def main():
    _, ep = read_csv("episodes.csv")
    header, up = read_csv("updates.csv")
    models = [(steps, PPO.load(path, device="cpu")) for steps, path in list_models()]

    # 1) score per match over the course of training
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.scatter(ep[:, 0], ep[:, 1], s=3, alpha=0.15, c="#868e96", label="each match")
    ax.plot(ep[49:, 0], moving_average(ep[:, 1]), c="#1c7ed6", lw=2, label="mean of the last 50")
    ax.set_xlabel("training steps")
    ax.set_ylabel("score")
    ax.set_title("Score per match during training")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    save(fig, "01_scores.png")

    # 2) what happens inside PPO
    col = {name: i for i, name in enumerate(header)}
    panels = [
        ("entropy", "Entropy: how undecided the agent is", "#ae3ec9"),
        ("critic_error", "Critic's error when predicting the reward", "#e8590c"),
        ("approx_kl", "How much the policy changes at each update (KL)", "#2f9e44"),
        ("clip_fraction", "Fraction of steps held back by PPO's clip", "#1c7ed6"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (name, title, color) in zip(axs.ravel(), panels):
        ax.plot(up[:, 0], up[:, col[name]], c=color, lw=1.3)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("training steps")
        ax.grid(alpha=0.3)
    save(fig, "02_ppo.png")

    # 3) each saved model playing matches it has never seen
    print(f"Testing {len(models)} models on {N_TEST_MATCHES} new matches each...")
    evaluation = []
    for steps, model in models:
        scores = [play(model, 10_000 + s) for s in range(N_TEST_MATCHES)]
        evaluation.append((steps, scores))
        print(f"  {steps:>9,} steps: mean {np.mean(scores):6.0f}")
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = [f"{s // 1000}k" for s, _ in evaluation]
    ax.bar(labels, [np.mean(s) for _, s in evaluation], color="#1c7ed6", alpha=0.8)
    for i, (_, scores) in enumerate(evaluation):
        ax.scatter([i] * len(scores), scores, c="k", s=8, zorder=3)
    ax.set_xlabel("model saved after N training steps")
    ax.set_ylabel("score")
    ax.set_title(f"Each model playing {N_TEST_MATCHES} new matches (dots = each match)")
    ax.grid(alpha=0.3, axis="y")
    save(fig, "03_evaluation.png")

    # 4) decision map: what each model does in each situation
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    chosen = [models[i] for i in sorted({0, len(models) // 3, 2 * len(models) // 3, len(models) - 1})]
    extent = (DISTANCES[0], DISTANCES[-1], SPEEDS[0], SPEEDS[-1])
    fig, axs = plt.subplots(2, len(chosen), figsize=(4 * len(chosen), 8),
                            squeeze=False, layout="constrained")
    for c, (steps, model) in enumerate(chosen):
        cactus = decision_map(model, CACTUS)
        im = axs[0, c].imshow(cactus[..., 1], origin="lower", aspect="auto", cmap="RdBu_r",
                              vmin=0, vmax=1, extent=extent)
        axs[0, c].set_title(f"{steps // 1000}k steps\nchance of jumping (cactus)", fontsize=9)
        axs[0, c].set_xlabel("distance to the cactus", fontsize=8)

        bird = decision_map(model, BIRD).argmax(axis=-1)
        axs[1, c].imshow(bird, origin="lower", aspect="auto", cmap=ListedColormap(ACTION_COLORS),
                         vmin=0, vmax=2, extent=extent, interpolation="nearest")
        axs[1, c].set_title(f"{steps // 1000}k steps\naction chosen (bird at head height)", fontsize=9)
        axs[1, c].set_xlabel("distance to the bird", fontsize=8)
    for row in range(2):
        axs[row, 0].set_ylabel("game speed")
    fig.colorbar(im, ax=axs[0, :], shrink=0.8, label="probability of jumping")
    axs[1, -1].legend(handles=[Patch(color=color, label=name) for color, name in zip(ACTION_COLORS, ACTION_NAMES)],
                      loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.savefig(os.path.join(OUTPUT, "04_decisions.png"), dpi=110)
    plt.close(fig)

    # 5) REPORT.md
    final = evaluation[-1][1]
    first_good = next((s for s, scores in evaluation if np.mean(scores) >= 4000), None)
    start, end = ep[:200, 1].mean(), ep[-200:, 1].mean()
    with open(os.path.join(OUTPUT, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(f"""# Training report — reinforcement learning (PPO)

> Generated automatically by `report.py` from `episodes.csv`, `updates.csv` and the models in `models/`.
> Run the training again and this report changes.

## Summary

| | |
|---|---|
| Training steps | {number(ep[-1, 0])} |
| Matches played during training | {number(len(ep))} |
| Mean score: first 200 → last 200 matches | {start:.0f} → {end:.0f} |
| First saved model with mean ≥ 4000 on new matches | {f"{first_good // 1000}k steps" if first_good is not None else "none"} |
| Final model on {N_TEST_MATCHES} new matches | mean {np.mean(final):.0f}, worst {min(final)}, best {max(final)} |

## 1. Score per match

![scores](01_scores.png)

Notice the **plateau**: for a good while the agent does not improve, and then suddenly the average shoots up. It discovered that jumping at the right moment pays off more than jumping nonstop.

## 2. Inside PPO

![ppo](02_ppo.png)

- **Entropy** plunges together with the jump in scores: at first the agent picks almost at random; once it discovers what works, it becomes decisive. It does not go to zero because, with the obstacle far away, the action does not matter, and the entropy bonus rewards it for keeping on exploring.
- **Critic's error** rises while the agent improves (matches get long and unpredictable) and falls once it survives almost every time: then the reward is easy to predict.
- **KL** and **clip fraction** stay small the whole time: that is PPO's *clip* holding back each update so the policy does not change too much at once.

## 3. Each saved model on new matches

![evaluation](03_evaluation.png)

| Model | Mean | Worst | Best |
|---|---|---|---|
""")
        for steps, scores in evaluation:
            f.write(f"| {steps // 1000}k steps | {np.mean(scores):.0f} | {min(scores)} | {max(scores)} |\n")
        f.write("""
## 4. What the network "thinks"

![decisions](04_decisions.png)

Each point is a question asked to the network: "with the obstacle at this distance and the game at this speed, what do you do?".

- **Top, the cactus:** at first the network answers the same for everything. At the end a **slanted boundary** appears: jump when the cactus gets close, and earlier the faster the game. It is the same idea as the band from lesson 1, but there the network was given ready-made examples; here, only the reward.
- **Bottom, the bird at head height:** the color shows the most likely action. Notice that the network does not have to choose what we would choose: jumping over the bird also works. The reward only says whether it worked, not how.
""")
    print(f"Report generated: {os.path.join(OUTPUT, 'REPORT.md')}")


if __name__ == "__main__":
    main()
