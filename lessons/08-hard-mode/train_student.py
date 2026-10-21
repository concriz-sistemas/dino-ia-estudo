"""
train_student.py — A student learns the hard game by imitating a COACH, with DAgger.

The coach is lesson 4's agent, with the planner taking over only near the new
formations (agents.Coach). The student is a small network that sees the same
6 sensors and learns to press what the coach would press:

    round 0     — the coach plays 6 matches of the hard game; every decision is recorded
    round 1+    — DAgger (lesson 6): the STUDENT plays 4 matches; in every situation it
                  gets into, the coach says what it would press; those are added to the
                  data, and a new student is trained on everything recorded so far
After each round the student plays 6 hard matches and 3 original ones (never trained on).

Run from the repository root (about 20 minutes: the coach's planner is slow):
    python lessons/08-hard-mode/train_student.py
    # the first idea, which doesn't work: the planner alone as the teacher
    python lessons/08-hard-mode/train_student.py --teacher planner

Outputs (in results/):
    student.pt              — the trained student
    04_student_rounds.png   — the student on both games after each round
    train_student.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FOLDER)
from agents import ACTIONS, STUDENT_8_FILE, Coach, PlannerTeacher, StudentNet, Student8, play
from hard_game import HardGame
from dino.game import Game

OUTPUT = os.path.join(FOLDER, "results")
SEED = 0
COACH_MATCHES = 6
DAGGER_ROUNDS = 6
DAGGER_MATCHES = 4
EPOCHS = 60
TEST_HARD = [5000 + s for s in range(6)]
TEST_ORIGINAL = [5000 + s for s in range(3)]


def record(driver, coach, seed):
    """`driver` plays one hard match; at every decision, the COACH says what it would press."""
    game = HardGame(1, seed=seed)
    dino = game.dinos[0]
    sensors, keys = [], []
    while dino.alive and dino.score < 5000:
        if game.frame % 2 == 0:
            sensors.append(game.sensors(dino))
            keys.append(ACTIONS.index(coach.decide(game)))
        game.step([driver.act(game)])
    return sensors, keys


def train(sensors, keys):
    """Plain supervised learning, like lesson 5: cross-entropy against the coach's keys."""
    torch.manual_seed(SEED)
    net = StudentNet()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    X = torch.tensor(np.array(sensors), dtype=torch.float32)
    y = torch.tensor(keys)
    for _ in range(EPOCHS):
        order = torch.randperm(len(X))
        for i in range(0, len(X), 256):
            idx = order[i:i + 256]
            loss = nn.functional.cross_entropy(net(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", choices=["coach", "planner"], default="coach")
    p.add_argument("--rounds", type=int, default=DAGGER_ROUNDS)
    args = p.parse_args()
    os.makedirs(OUTPUT, exist_ok=True)
    start = time.time()
    coach = Coach() if args.teacher == "coach" else PlannerTeacher()
    tag = "" if args.teacher == "coach" else "_planner_teacher"
    sensors, keys = [], []
    print(f"Round 0: the teacher ({coach.name}) plays {COACH_MATCHES} hard matches...", flush=True)
    for m in range(COACH_MATCHES):
        s, k = record(coach, coach, 1000 + m)
        sensors += s
        keys += k

    history = []
    counts = np.bincount(keys, minlength=3) / len(keys)
    print(f"  the teacher pressed: nothing {counts[0]:.0%}, jump {counts[1]:.0%}, duck {counts[2]:.0%}", flush=True)
    for rnd in range(args.rounds + 1):
        student = Student8(train(sensors, keys))
        hard = [play(student, HardGame, s) for s in TEST_HARD]
        original = [play(student, Game, s)[0] for s in TEST_ORIGINAL]
        history.append({"round": rnd, "examples": len(keys), "hard": [h[0] for h in hard],
                        "crashed_into": [h[1] for h in hard], "original": original})
        print(f"  round {rnd}: {len(keys):6d} examples   hard game: mean {np.mean([h[0] for h in hard]):5.0f}   "
              f"original game: {original}   [{time.time() - start:.0f} s]", flush=True)
        if rnd == args.rounds:
            break
        for m in range(DAGGER_MATCHES):
            s, k = record(student, coach, 2000 + 10 * rnd + m)
            sensors += s
            keys += k

    with open(os.path.join(OUTPUT, f"train_student{tag}.json"), "w") as f:
        json.dump(history, f, indent=1)
    if tag:
        print("\n(Trained only to compare: the saved student and the chart are left as they were.)")
        return
    torch.save(student.net.state_dict(), STUDENT_8_FILE)

    rounds = [h["round"] for h in history]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(rounds, [np.mean(h["hard"]) for h in history], "o-", c="#e8590c", lw=2, label="hard game (6 matches)")
    ax.plot(rounds, [np.mean(h["original"]) for h in history], "s--", c="#868e96", lw=2,
            label="original game (3 matches)")
    for h in history:
        ax.scatter([h["round"]] * len(h["hard"]), h["hard"], c="#ffc078", s=14, zorder=1)
    ax.set_xticks(rounds)
    ax.set_xlabel("round (0 = only the coach's matches; 1+ = DAgger)")
    ax.set_ylabel("mean score")
    ax.set_ylim(0, 5400)
    ax.set_title("The student of the coach, round by round")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "04_student_rounds.png"), dpi=110)
    plt.close(fig)
    print(f"\nStudent saved to: {STUDENT_8_FILE}")
    print("Test it against the others:")
    print("    python lessons/08-hard-mode/test_agents.py --student")


if __name__ == "__main__":
    main()
