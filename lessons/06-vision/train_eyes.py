"""
Lesson 6 — Teaches the EYES (a CNN) to read the game from the screen.

Run from the repository root (about 20-25 minutes on a recent Mac; longer on a computer
without a graphics card that PyTorch can use — trained eyes are already in results/):
    python lessons/06-vision/train_eyes.py

The training happens in rounds:
    round 0  — the brain plays with the TRUE sensors; the eyes learn from those scenes
    round 1+ — "DAgger": eyes + brain play TOGETHER; the scenes they get into are
               recorded with the TRUE sensors as the answer key, and the eyes keep
               training on everything recorded so far
After each round, eyes + brain play 10 new matches (never used for training).

Outputs (in results/):
    eyes.pt                 — the trained eyes (the learning, in one file)
    history.json            — scores and errors per round
    01_rounds.png           — how eyes + brain play after each round
    02_estimates.png        — true vs estimated distance during one match
    03_filters.png          — the 16 filters of the first layer
    04_what_layers_see.png  — what each layer "sees" in one scene
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

FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FOLDER)
from vision import OUTPUT, EYES_FILE, DEVICE, SENSOR_NAMES, Eyes, Brain, play

SEED = 0
RECORD_EVERY = 2        # neighboring frames are almost identical: keep 1 in 2
TEACHER_MATCHES = 8     # round 0: matches played by the brain with the true sensors
DAGGER_MATCHES = 6      # each later round: matches played by eyes + brain
FIRST_EPOCHS = 8
EPOCHS_PER_ROUND = 4
TEST_SEEDS = [5000 + s for s in range(10)]
NEAR = 0.5              # "near" obstacle: distance below this. Those are the ones that matter.


def train(eyes, images, answers, epochs):
    """Gradient descent (Adam, a smarter variant) with a cosine learning rate schedule."""
    X = np.array(images)
    S = torch.tensor(np.array(answers, dtype=np.float32))
    # Errors about near obstacles count 10× more: far ones don't change any decision
    weight = torch.where(S[:, 0] < NEAR, 3.0, 0.3)
    target = (S - eyes.mean.cpu()) / eyes.std.cpu()
    opt = torch.optim.Adam(eyes.parameters(), 1e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    eyes.train()
    for _ in range(epochs):
        order = torch.randperm(len(X))
        for i in range(0, len(X), 256):
            idx = order[i:i + 256]
            batch = torch.as_tensor(X[idx.numpy()], device=DEVICE)
            error = (eyes(batch) - target[idx].to(DEVICE)) ** 2
            loss = (error.mean(1) * weight[idx].to(DEVICE)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        schedule.step()
    eyes.eval()


def evaluate(eyes, brain):
    """Eyes + brain play the test matches. Returns the scores and the eyes' error near obstacles."""
    scores, errors = [], []
    for seed in TEST_SEEDS:
        score, images, answers = play(brain, seed, eyes=eyes, record_every=RECORD_EVERY)
        scores.append(score)
        for pair, true in zip(images, answers):
            if true[0] < NEAR:
                errors.append(np.abs(eyes.estimate(pair) - np.array(true)))
    return scores, np.mean(errors, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=6, help="DAgger rounds after round 0")
    args = p.parse_args()
    torch.manual_seed(SEED)
    os.makedirs(OUTPUT, exist_ok=True)
    start = time.time()
    print(f"Training on: {DEVICE}")

    brain = Brain()
    eyes = Eyes().to(DEVICE)
    images, answers = [], []
    history = []

    for rnd in range(args.rounds + 1):
        if rnd == 0:
            print(f"\nRound 0: the brain plays {TEACHER_MATCHES} matches with the TRUE sensors...")
            for m in range(TEACHER_MATCHES):
                _, im, an = play(brain, 1000 + m, record_every=RECORD_EVERY)
                images += im
                answers += an
            s = np.array(answers, dtype=np.float32)
            eyes.mean.copy_(torch.tensor(s.mean(0)))
            eyes.std.copy_(torch.tensor(s.std(0) + 1e-6))
            epochs = FIRST_EPOCHS
        else:
            print(f"\nRound {rnd} (DAgger): eyes + brain play {DAGGER_MATCHES} matches; "
                  f"their scenes are recorded with the true sensors...")
            for m in range(DAGGER_MATCHES):
                _, im, an = play(brain, 2000 + 10 * rnd + m, eyes=eyes, record_every=RECORD_EVERY)
                images += im
                answers += an
            epochs = EPOCHS_PER_ROUND
        print(f"  training the eyes on {len(images)} scenes for {epochs} epochs...")
        train(eyes, images, answers, epochs)

        scores, error = evaluate(eyes, brain)
        history.append({"round": rnd, "scenes": len(images), "scores": scores,
                        "mean_score": float(np.mean(scores)),
                        "near_error": {n: round(float(e), 4) for n, e in zip(SENSOR_NAMES, error)}})
        print(f"  eyes + brain on 10 new matches: {scores}  (mean {np.mean(scores):.0f})")
        print(f"  distance error near obstacles: {error[0]:.4f}  ({error[0] * 800:.0f} pixels)"
              f"   [{time.time() - start:.0f} s]")

    torch.save(eyes.state_dict(), EYES_FILE)
    with open(os.path.join(OUTPUT, "history.json"), "w") as f:
        json.dump(history, f, indent=1)

    # 1) the rounds
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    rounds = [h["round"] for h in history]
    for h in history:
        axs[0].scatter([h["round"]] * len(h["scores"]), h["scores"], c="#adb5bd", s=14, zorder=2)
    axs[0].plot(rounds, [h["mean_score"] for h in history], "o-", c="#1c7ed6", lw=2, zorder=3, label="mean")
    axs[0].set_title("Eyes + brain on 10 new matches (dots = each match)")
    axs[0].set_ylabel("score")
    axs[0].legend()
    axs[1].plot(rounds, [h["near_error"]["distance"] * 800 for h in history], "o-", c="#e8590c", lw=2)
    axs[1].set_title("The eyes' distance error near obstacles")
    axs[1].set_ylabel("average error (pixels)")
    axs[1].set_ylim(bottom=0)
    for ax in axs:
        ax.set_xlabel("round (0 = only the teacher's scenes; 1+ = DAgger)")
        ax.set_xticks(rounds)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "01_rounds.png"), dpi=110)
    plt.close(fig)

    # 2) one match: true vs estimated distance
    trace = []
    play(brain, TEST_SEEDS[0], eyes=eyes,
         on_frame=lambda game, pair, true, seen, action: trace.append((game.frame, true[0], seen[0])))
    trace = np.array(trace[:900])
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(trace[:, 0], trace[:, 1] * 800, c="k", lw=1.5, label="true distance (from the game)")
    ax.plot(trace[:, 0], trace[:, 2] * 800, c="#e8590c", lw=1.2, label="distance estimated by the eyes")
    ax.set_xlabel("frame")
    ax.set_ylabel("distance to the next obstacle (pixels)")
    ax.set_title("Each sawtooth is one obstacle coming closer; the eyes have to follow it from the pixels alone")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "02_estimates.png"), dpi=110)
    plt.close(fig)

    # 3) the first layer's filters (the "now" image channel)
    w = eyes.conv[0].weight.detach().cpu().numpy()[:, 1]
    fig, axs = plt.subplots(2, 8, figsize=(11, 3))
    for i, ax in enumerate(axs.ravel()):
        lim = np.abs(w[i]).max()
        ax.imshow(w[i], cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(f"filter {i + 1}", fontsize=8)
        ax.axis("off")
    fig.suptitle("The 16 filters of the first layer (5×5 pixels each). Red = 'bright here', blue = 'dark here'.")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "03_filters.png"), dpi=110)
    plt.close(fig)

    # 4) what each layer sees in one scene with a cactus close by
    scene = next(img for img, ans in zip(images, answers) if 0.15 < ans[0] < 0.3 and ans[4] == 0)
    with torch.no_grad():
        x = torch.as_tensor(scene[None], device=DEVICE).float() / 255
        x = torch.cat([x, eyes.columns.expand(1, -1, -1, -1)], dim=1)
        maps = []
        for layer in eyes.conv:
            x = layer(x)
            if isinstance(layer, torch.nn.ReLU):
                maps.append(x[0].cpu().numpy())
    fig, axs = plt.subplots(4, 4, figsize=(13, 6.5))
    axs[0, 0].imshow(scene[1], cmap="gray")
    axs[0, 0].set_title("what the eyes receive (now)", fontsize=9)
    axs[0, 1].imshow(scene[1].astype(int) - scene[0].astype(int), cmap="RdBu_r", vmin=-120, vmax=120)
    axs[0, 1].set_title("now minus 3 frames ago (movement)", fontsize=9)
    for ax in axs[0, 2:]:
        ax.axis("off")
    for row, (fm, name) in enumerate(zip(maps, ["layer 1", "layer 2", "layer 3"]), start=1):
        strongest = np.argsort(fm.reshape(len(fm), -1).max(1))[::-1][:4]
        for col, ch in enumerate(strongest):
            axs[row, col].imshow(fm[ch], cmap="magma")
            axs[row, col].set_title(f"{name}, channel {ch + 1}", fontsize=8)
    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Deeper layers see less detail but bigger things: edges → shapes → 'something is here'")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "04_what_layers_see.png"), dpi=110)
    plt.close(fig)

    print(f"\nEyes saved to: {EYES_FILE}")
    print(f"Charts in: {OUTPUT}")
    print("Watch eyes + brain play:")
    print("    python lessons/06-vision/watch.py")


if __name__ == "__main__":
    main()
