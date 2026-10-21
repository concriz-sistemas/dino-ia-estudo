"""
test_agents.py — The world changed. Who can still play?

Every agent of the study plays 10 matches of the ORIGINAL game and 10 of the
HARD one, with the same seeds. The planner (which doesn't learn, it searches)
plays too, as a reference. It's slow, so it plays 5 of each.

Run from the repository root (about 4 minutes):
    python lessons/08-hard-mode/test_agents.py
    # also test lesson 8's student (after train_student.py)
    python lessons/08-hard-mode/test_agents.py --student

Output: results/01_who_can_play.png, and results/test_agents.json
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FOLDER)
from agents import Lesson3Champion, Lesson4Agent, Lesson5Student, Planner, Student8, play
from hard_game import HardGame
from dino.game import Game

OUTPUT = os.path.join(FOLDER, "results")
SEEDS = [5000 + s for s in range(10)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student", action="store_true", help="also test lesson 8's student")
    args = p.parse_args()
    os.makedirs(OUTPUT, exist_ok=True)

    agents = []
    for make in [Lesson3Champion, Lesson4Agent, Lesson5Student]:
        try:
            agents.append((make(), SEEDS))
        except FileNotFoundError as e:
            print(f"Skipping {make.name}: {e}")
    if args.student:
        agents.append((Student8(), SEEDS))
    agents.append((Planner(), SEEDS[:5]))

    results = {}
    for agent, seeds in agents:
        row = {}
        for label, game_class in [("original game", Game), ("hard game", HardGame)]:
            games = [play(type(agent)() if isinstance(agent, Planner) else agent, game_class, s) for s in seeds]
            scores = [score for score, _ in games]
            hits = Counter(hit for _, hit in games if hit != "-")
            row[label] = {"scores": scores, "mean": float(np.mean(scores)), "crashed_into": dict(hits)}
            print(f"{agent.name:<42} {label:<14} mean {np.mean(scores):6.0f}   crashed into: {dict(hits) or 'nothing'}")
        results[agent.name] = row
    with open(os.path.join(OUTPUT, "test_agents.json"), "w") as f:
        json.dump(results, f, indent=1)

    names = list(results)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, (label, color) in enumerate([("original game", "#adb5bd"), ("hard game", "#e8590c")]):
        means = [results[n][label]["mean"] for n in names]
        bars = ax.bar(x + (i - 0.5) * 0.38, means, 0.38, color=color, label=label)
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, m + 60, f"{m:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, [n.replace(": ", ":\n") for n in names], fontsize=8)
    ax.set_ylabel("mean score")
    ax.set_ylim(0, 5600)
    ax.set_title("The same agents, before and after the world changed")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "01_who_can_play.png"), dpi=110)
    plt.close(fig)
    print(f"\nChart saved to: {os.path.join(OUTPUT, '01_who_can_play.png')}")


if __name__ == "__main__":
    main()
