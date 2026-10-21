"""
pixels_to_keys.py — The first idea, which does NOT work well: one network from pixels straight to keys.

Like lesson 5, but the student sees the screen instead of the 6 sensors:
    two images  ->  CNN  ->  nothing / jump / duck
It learns to copy lesson 4's agent from 6 recorded matches.

Run from the repository root (a few minutes):
    python lessons/06-vision/pixels_to_keys.py

It prints how often the student presses the same key as the teacher, and then
how it scores when it plays on its own. Compare both numbers.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FOLDER)
from vision import DEVICE, IMG_H, IMG_W, ACTIONS, SCORE_LIMIT, BRAIN_EVERY, Brain, Camera, Game

SEED = 0
MATCHES = 6
EPOCHS = 8


class PixelsToKeys(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, 5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, 2, IMG_H, IMG_W)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, images):
        return self.head(self.conv(images.float() / 255))


def record(brain, seed):
    """The teacher plays with the true sensors; every frame keeps (images, key it pressed)."""
    game = Game(1, seed=seed, colors=[(83, 83, 83)])
    dino, camera = game.dinos[0], Camera()
    images, keys, action = [], [], 0
    while dino.alive and dino.score < SCORE_LIMIT:
        pair = camera.look(game)
        if game.frame % BRAIN_EVERY == 0:
            action = brain.decide(game.sensors(dino))
        images.append(pair)
        keys.append(action)
        game.step([ACTIONS[action]])
    return images, keys


def play(net, seed):
    game = Game(1, seed=seed, colors=[(83, 83, 83)])
    dino, camera = game.dinos[0], Camera()
    with torch.no_grad():
        while dino.alive and dino.score < SCORE_LIMIT:
            pair = torch.as_tensor(camera.look(game)[None], device=DEVICE)
            game.step([ACTIONS[int(net(pair).argmax())]])
    return dino.score


def main():
    torch.manual_seed(SEED)
    brain = Brain()
    images, keys = [], []
    print(f"Recording {MATCHES} matches of the teacher...")
    for m in range(MATCHES):
        im, k = record(brain, 1000 + m)
        images += im
        keys += k
    X, y = np.array(images), torch.tensor(keys)
    split = int(len(X) * 0.9)

    net = PixelsToKeys().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    loss_fn = nn.CrossEntropyLoss()
    print(f"Training on {split} frames for {EPOCHS} epochs (on {DEVICE})...")
    for _ in range(EPOCHS):
        order = torch.randperm(split)
        for i in range(0, split, 256):
            idx = order[i:i + 256]
            out = net(torch.as_tensor(X[idx.numpy()], device=DEVICE))
            loss = loss_fn(out, y[idx].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        guesses = torch.cat([net(torch.as_tensor(X[i:i + 512], device=DEVICE)).argmax(1).cpu()
                             for i in range(split, len(X), 512)])
    agreement = float((guesses == y[split:]).float().mean())
    print(f"\nPresses the same key as the teacher in {agreement:.1%} of the frames it never trained on.")
    scores = [play(net, 5000 + s) for s in range(10)]
    print(f"Playing on its own, 10 new matches: {scores}  (mean {np.mean(scores):.0f})")
    print("For comparison, eyes + brain: python lessons/06-vision/watch.py")


if __name__ == "__main__":
    main()
