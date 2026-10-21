"""
rl_attempts.py — Can reinforcement learning (lesson 4) learn the hard game by itself?

Four attempts, each one fixing what might have gone wrong in the previous one:
    1. from scratch      — a new agent, trained only on the hard game
    2. fine-tuning       — lesson 4's agent, trained some more on the hard game
    3. curriculum        — like 2, but in training the formations appear from the
                           very start and in half of the obstacles, to practice them
    4. more exploration  — like 3, with a much bigger bonus for trying other keys

Each one is tested on 10 matches of the original game and 10 of the hard game.
Then the last attempt is followed while it approaches a staircase, to see why.

Run from the repository root (about 10 minutes, mostly training):
    python lessons/08-hard-mode/rl_attempts.py
    # redraw the charts from the saved models, without training again
    python lessons/08-hard-mode/rl_attempts.py --charts-only

Outputs (in results/):
    02_rl_attempts.png   — the four attempts on both games
    03_why_rl_fails.png  — the chance it gives to each key while it approaches a staircase
    rl_attempts.json
"""
import json
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FOLDER)
from agents import LESSONS, LESSON_4_FILE, Lesson4Agent, play
import hard_game
from hard_game import HardGame
from dino.game import Game

sys.path.insert(0, os.path.join(LESSONS, "04-reinforcement"))
from environment import DinoEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

OUTPUT = os.path.join(FOLDER, "results")
MODELS = os.path.join(OUTPUT, "rl_models")
SEEDS = [5000 + s for s in range(10)]
# The same settings as lesson 4
PPO_SETTINGS = dict(n_steps=1024, batch_size=256, learning_rate=3e-4, gamma=0.99, clip_range=0.2, verbose=0)


class PracticeGame(HardGame):
    """For the curriculum: formations from the very first obstacle, in half of them."""

    def _spawn_obstacle(self):
        if self.rng.random() < 0.5:
            make = hard_game.FORMATIONS[self.rng.choice(sorted(hard_game.FORMATIONS))]
            self.obstacles += make(hard_game.WIDTH + 10, max(self.speed, 10.5))
            self._next_gap = self.rng.randint(int(self.speed * 30), int(self.speed * 55))
        else:
            Game._spawn_obstacle(self)


def env_class(game_class):
    class Env(DinoEnv):
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.game = game_class(1, seed=int(self.np_random.integers(0, 2**31)), colors=[(40, 120, 220)])
            return self._observe(), {}
    return Env


ATTEMPTS = [
    # name, start from lesson 4?, training game, entropy bonus, steps
    ("1. from scratch", False, HardGame, 0.03, 3_000_000),
    ("2. fine-tuning lesson 4", True, HardGame, 0.03, 1_000_000),
    ("3. + curriculum", True, PracticeGame, 0.03, 2_000_000),
    ("4. + more exploration", True, PracticeGame, 0.10, 3_000_000),
]


def main():
    os.makedirs(MODELS, exist_ok=True)
    if "--charts-only" in sys.argv:
        with open(os.path.join(OUTPUT, "rl_attempts.json")) as f:
            results = json.load(f)
        draw(results)
        return
    results = {}
    for name, from_lesson_4, game_class, entropy, steps in ATTEMPTS:
        start = time.time()
        env = make_vec_env(env_class(game_class), n_envs=8, seed=0)
        if from_lesson_4:
            model = PPO.load(LESSON_4_FILE, env=env, device="cpu", ent_coef=entropy, **PPO_SETTINGS)
        else:
            model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=[32, 32]), ent_coef=entropy,
                        seed=0, device="cpu", **PPO_SETTINGS)
        model.learn(steps)
        path = os.path.join(MODELS, name.split(".")[0] + ".zip")
        model.save(path)
        agent = Lesson4Agent(path)
        original = [play(agent, Game, s)[0] for s in SEEDS]
        hard = [play(agent, HardGame, s) for s in SEEDS]
        results[name] = {"steps": steps, "original game": original, "hard game": [h[0] for h in hard],
                         "crashed_into": [h[1] for h in hard]}
        print(f"{name:<26} {steps:>9,} steps   original game: mean {np.mean(original):5.0f}   "
              f"hard game: mean {np.mean([h[0] for h in hard]):5.0f}   [{time.time() - start:.0f} s]", flush=True)
    with open(os.path.join(OUTPUT, "rl_attempts.json"), "w") as f:
        json.dump(results, f, indent=1)
    draw(results)


def draw(results):
    # 1) the four attempts
    lesson4 = Lesson4Agent()
    rows = [("lesson 4, as it was", [play(lesson4, Game, s)[0] for s in SEEDS],
             [play(lesson4, HardGame, s)[0] for s in SEEDS])]
    rows += [(n, r["original game"], r["hard game"]) for n, r in results.items()]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, (label, color) in enumerate([("original game", "#adb5bd"), ("hard game", "#e8590c")]):
        means = [np.mean(r[1 + i]) for r in rows]
        bars = ax.bar(x + (i - 0.5) * 0.38, means, 0.38, color=color, label=label)
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, m + 60, f"{m:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, [r[0] for r in rows], fontsize=8)
    ax.set_ylabel("mean score in 10 matches")
    ax.set_ylim(0, 5600)
    ax.set_title("Reinforcement learning alone doesn't get through the new formations")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "02_rl_attempts.png"), dpi=110)
    plt.close(fig)

    # 2) why: follow the last attempt while it approaches the first staircase of a match
    agent = Lesson4Agent(os.path.join(MODELS, "4.zip"))
    game = HardGame(1, seed=SEEDS[0])
    dino = game.dinos[0]
    trace = []
    while dino.alive and dino.score < 5000:
        ahead = game.next_obstacle(dino)
        if ahead is not None and ahead.kind == "bird_staircase" and ahead.rect.left - dino.x < 250:
            trace.append((game.frame, *agent.probabilities(game), 250 - dino.feet))
        game.step([agent.act(game)])
    t = np.array(trace)
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for i, (label, color) in enumerate([("nothing", "#868e96"), ("jump", "#e03131"), ("duck", "#1971c2")]):
        axs[0].plot(t[:, 0], t[:, 1 + i] * 100, c=color, lw=2, label=label)
    axs[0].set_ylabel("chance (%)")
    axs[0].set_title("The agent approaching a staircase: the chance it gives to each key")
    axs[0].legend()
    axs[0].grid(alpha=0.3)
    axs[1].plot(t[:, 0], t[:, 4], c="k", lw=2)
    axs[1].set_ylabel("dino's height (pixels)")
    axs[1].set_xlabel("frame")
    axs[1].set_title("What it did: it ducked while the birds were far, then jumped when they got close "
                     "(its old habit for birds), and hit one")
    axs[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "03_why_rl_fails.png"), dpi=110)
    plt.close(fig)
    print(f"Highest chance given to 'duck' while approaching the staircase: {t[:, 3].max():.1%}")
    print(f"\nCharts in: {OUTPUT}")


if __name__ == "__main__":
    main()
