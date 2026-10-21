"""
Lesson 5 — Trains a STUDENT network to imitate the demonstrations.

Run from the repository root:
    # the example demonstrations: lesson 4's agent, 10 matches
    python lessons/05-imitation/train.py
    # your own demonstrations, recorded with record.py
    python lessons/05-imitation/train.py --demos lessons/05-imitation/results/demos/human_20261021_120000.csv
    # only the first match of a file (how much data does the student really need?)
    python lessons/05-imitation/train.py --matches 1

What it does:
    1. loads the demonstrations (sensors + the action that was taken);
    2. keeps the last match aside to check the student on situations it never saw;
    3. trains the student by gradient descent (lesson 1, now with 3 answers);
    4. lets the student play 10 new matches on its own;
    5. saves the student's weights and the charts.

Outputs (in results/<name of the demonstrations>/):
    01_actions.png    — what the teacher did, and the "always do nothing" trap
    02_training.png   — loss and accuracy during training
    03_decisions.png  — what the student decides for each distance and speed
and the student itself in results/models/student_<name>.npz
"""
import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FOLDER)
from imitation import OUTPUT, DEMOS, MODELS, ACTION_NAMES, Student, load_demos, play

DEFAULT_DEMOS = os.path.join(DEMOS, "teacher_02000k.csv")
N_TEST_MATCHES = 10
COLORS = ["#adb5bd", "#e03131", "#1971c2"]  # nothing, jump, duck

# Situations to ask the student about: an obstacle at many distances and speeds
DISTANCES = np.linspace(-0.05, 0.6, 66)
SPEEDS = np.linspace(0.4, 1.0, 31)
CACTUS = (20 / 60, 40 / 60, 0.0)
BIRD = (40 / 60, 25 / 60, 35 / 60)   # at head height


def ask(student, obstacle):
    width, height, elev = obstacle
    D, S = np.meshgrid(DISTANCES, SPEEDS)
    X = np.column_stack([D.ravel(), np.full(D.size, width), np.full(D.size, height),
                         np.full(D.size, elev), np.zeros(D.size), S.ravel()])
    return student.probabilities(X)[0].reshape(*D.shape, 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demos", nargs="+", default=[DEFAULT_DEMOS], help="one or more CSV files from record.py")
    p.add_argument("--matches", type=int, default=0, help="use only the first N matches (0 = all)")
    args = p.parse_args()

    X, y, match = load_demos(args.demos)
    if args.matches:
        keep = match < args.matches
        X, y, match = X[keep], y[keep], match[keep]
    name = "+".join(os.path.splitext(os.path.basename(d))[0] for d in args.demos)
    if args.matches:
        name += f"_{args.matches}match"
    out = os.path.join(OUTPUT, name)
    os.makedirs(out, exist_ok=True)

    counts = np.bincount(y, minlength=3)
    print(f"{len(y)} examples from {len(set(match))} match(es)")
    for n, c in zip(ACTION_NAMES, counts):
        print(f"  {n:<8} {c:6d}  ({c / len(y):.0%})")
    trap = counts.max() / len(y)
    print(f"A student that always answers '{ACTION_NAMES[counts.argmax()]}' would already be "
          f"{trap:.0%} accurate, and would crash at the first cactus.")

    # Keep the last match aside: it checks the student on situations it never trained on
    if len(set(match)) > 1:
        val = match == match.max()
        X_train, y_train, X_val, y_val = X[~val], y[~val], X[val], y[val]
        aside = "the match kept aside"
    else:
        cut = int(len(y) * 0.8)
        X_train, y_train, X_val, y_val = X[:cut], y[:cut], X[cut:], y[cut:]
        aside = "the last 20% of the match, kept aside"

    print("\nTraining the student (6 -> 16 -> 3)...")
    student = Student()
    history = np.array(student.train(X_train, y_train, X_val=X_val, y_val=y_val), dtype=float)
    print(f"Accuracy: {history[-1, 2]:.1%} on the training examples, "
          f"{history[-1, 3]:.1%} on {aside}")

    print(f"\nThe student plays {N_TEST_MATCHES} new matches on its own:")
    scores = [play(student, seed=5000 + s) for s in range(N_TEST_MATCHES)]
    print("  " + ", ".join(map(str, scores)) + f"   (mean {np.mean(scores):.0f})")

    model_path = os.path.join(MODELS, f"student_{name}.npz")
    student.save(model_path)

    # 1) what the demonstrations contain
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(ACTION_NAMES, counts / len(y) * 100, color=COLORS)
    ax.axhline(trap * 100, c="k", ls="--", lw=1)
    ax.text(2.45, trap * 100 + 1.5, f"'always {ACTION_NAMES[counts.argmax()]}' = {trap:.0%} accurate",
            ha="right", fontsize=8)
    ax.set_ylabel("% of the frames")
    ax.set_title(f"What was pressed in the demonstrations ({len(y)} frames)")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "01_actions.png"), dpi=110)
    plt.close(fig)

    # 2) training curves
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    axs[0].plot(history[:, 0], history[:, 1], c="#1c7ed6")
    axs[0].set_title("Loss (how far the student is from the teacher)")
    axs[1].plot(history[:, 0], history[:, 2] * 100, c="#2f9e44", label="training examples")
    axs[1].plot(history[:, 0], history[:, 3] * 100, c="#e8590c", label=aside)
    axs[1].axhline(trap * 100, c="k", ls="--", lw=1, label=f"'always {ACTION_NAMES[counts.argmax()]}'")
    axs[1].set_title("Accuracy: how often the student presses what the teacher pressed")
    axs[1].set_ylim(0, 100)
    axs[1].legend(loc="lower right", fontsize=8)
    for ax in axs:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "02_training.png"), dpi=110)
    plt.close(fig)

    # 3) decision map, comparable with lesson 4's 04_decisions.png
    extent = (DISTANCES[0], DISTANCES[-1], SPEEDS[0], SPEEDS[-1])
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    im = axs[0].imshow(ask(student, CACTUS)[..., 1], origin="lower", aspect="auto", cmap="RdBu_r",
                       vmin=0, vmax=1, extent=extent)
    axs[0].set_title("Chance of jumping (cactus)")
    axs[0].set_xlabel("distance to the cactus")
    fig.colorbar(im, ax=axs[0], label="probability")
    axs[1].imshow(ask(student, BIRD).argmax(-1), origin="lower", aspect="auto", cmap=ListedColormap(COLORS),
                  vmin=0, vmax=2, extent=extent, interpolation="nearest")
    axs[1].set_title("Chosen action (bird at head height)")
    axs[1].set_xlabel("distance to the bird")
    axs[1].legend(handles=[Patch(color=c, label=n) for c, n in zip(COLORS, ACTION_NAMES)],
                  loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    for ax in axs:
        ax.set_ylabel("game speed")
    fig.suptitle("What the student learned to do")
    fig.savefig(os.path.join(out, "03_decisions.png"), dpi=110)
    plt.close(fig)

    print(f"\nStudent saved to: {model_path}")
    print(f"Charts in: {out}")
    print("Watch the student play:")
    print(f"    python lessons/05-imitation/watch.py --student {os.path.relpath(model_path)}")


if __name__ == "__main__":
    main()
