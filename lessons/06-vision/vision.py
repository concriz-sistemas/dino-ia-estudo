"""
vision.py — The pieces shared by the scripts of lesson 6.

- What the network sees: the game screen, cropped, shrunk and turned to grayscale.
- The EYES: a convolutional network (CNN) that looks at two of those images and
  estimates the same 6 numbers the game's sensors give (distance to the next
  obstacle, its width, height and elevation, the dino's height and the speed).
- The BRAIN: lesson 4's agent, unchanged. It decides what to press from the
  6 numbers, without knowing whether they came from the game or from the eyes.
"""
import os
import sys

import numpy as np

FOLDER = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FOLDER, "..", ".."))
LESSON_4 = os.path.join(ROOT, "lessons", "04-reinforcement")
sys.path.insert(0, ROOT)
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame
import torch
import torch.nn as nn

from dino.game import Game, Dino, WIDTH, HEIGHT, NOTHING, JUMP, DUCK, SENSOR_NAMES

OUTPUT = os.path.join(FOLDER, "results")
EYES_FILE = os.path.join(OUTPUT, "eyes.pt")

ACTIONS = [NOTHING, JUMP, DUCK]
SCORE_LIMIT = 5000
BRAIN_EVERY = 2        # lesson 4's agent decides every 2 frames

# ---------------------------------------------------------------------------
# What the network sees
# ---------------------------------------------------------------------------
# Only the part of the screen that matters: from the top of a jump to just
# below the ground, and up to 600 pixels to the right (farther obstacles don't
# matter yet). Then every 3×5 block of pixels becomes one: 200 × 44 pixels.
TOP, BOTTOM, RIGHT = 40, 260, 600
SHRINK_X, SHRINK_Y = 3, 5
IMG_W, IMG_H = RIGHT // SHRINK_X, (BOTTOM - TOP) // SHRINK_Y
# One image doesn't show movement. The network gets the current image and the
# one from FRAME_GAP frames ago, so it can tell how fast things move.
FRAME_GAP = 3

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


class Camera:
    """Draws the game off-screen and turns it into the small image the eyes receive."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.Surface((WIDTH, HEIGHT))
        self.history = []

    def snapshot(self, game):
        # No text and no red "sensor line": the eyes get only the scene, like a person would
        game.draw(self.screen, None, [], highlight=None)
        pixels = pygame.surfarray.array3d(self.screen)[:RIGHT, TOP:BOTTOM].mean(axis=2).T
        small = pixels.reshape(IMG_H, SHRINK_Y, IMG_W, SHRINK_X).mean(axis=(1, 3))
        return (255 - small).astype(np.uint8)  # dark things (dino, cacti) become bright

    def look(self, game):
        """The pair of images for the current frame: FRAME_GAP frames ago, and now."""
        now = self.snapshot(game)
        if not self.history:
            self.history = [now] * FRAME_GAP
        pair = np.stack([self.history[0], now])
        self.history = self.history[1:] + [now]
        return pair


# ---------------------------------------------------------------------------
# The eyes: a convolutional network
# ---------------------------------------------------------------------------
class Eyes(nn.Module):
    """
    3 convolutional layers slide small filters over the image looking for
    patterns (edges, then shapes, then "a cactus here"). A final dense layer
    turns what they found into the 6 sensor numbers.
    """

    def __init__(self):
        super().__init__()
        # An extra input channel that says "how far to the right is this pixel".
        # Convolutions are good at WHAT; this helps them say WHERE ("CoordConv").
        self.register_buffer("columns", torch.linspace(0, 1, IMG_W).repeat(IMG_H, 1)[None, None])
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=(2, 1), padding=2), nn.ReLU(), nn.MaxPool2d((1, 2)),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, 3, IMG_H, IMG_W)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, 6))
        # The 6 numbers have very different ranges; the network predicts them standardized
        self.register_buffer("mean", torch.zeros(6))
        self.register_buffer("std", torch.ones(6))

    def forward(self, images):
        """images: a batch of pairs, uint8 0..255. Returns standardized sensor estimates."""
        x = images.float() / 255
        x = torch.cat([x, self.columns.expand(len(x), -1, -1, -1)], dim=1)
        return self.head(self.conv(x))

    def estimate(self, pair):
        """One pair of images -> the 6 estimated sensor numbers."""
        with torch.no_grad():
            out = self(torch.as_tensor(pair[None], device=self.mean.device))[0]
            return (out * self.std + self.mean).cpu().numpy()


def load_eyes(path=EYES_FILE):
    eyes = Eyes().to(DEVICE)
    eyes.load_state_dict(torch.load(path, map_location=DEVICE))
    eyes.eval()
    return eyes


# ---------------------------------------------------------------------------
# The brain: lesson 4's agent
# ---------------------------------------------------------------------------
class Brain:
    def __init__(self, thousand_steps=2000):
        from stable_baselines3 import PPO

        path = os.path.join(LESSON_4, "results", "models", f"ppo_{thousand_steps:05d}k.zip")
        if not os.path.exists(path):
            sys.exit(f"Lesson 4 model not found: {path}\nTrain lesson 4 first.")
        self.model = PPO.load(path, device="cpu")

    def decide(self, sensors):
        action, _ = self.model.predict(np.array(sensors, dtype=np.float32), deterministic=True)
        return int(action)


# ---------------------------------------------------------------------------
# Playing and recording
# ---------------------------------------------------------------------------
def play(brain, seed, eyes=None, record_every=0, on_frame=None):
    """
    One match. The brain decides from the TRUE sensors (eyes=None) or from what
    the EYES estimate. If record_every > 0, every record_every-th frame is kept
    as a training example: the image pair and the TRUE sensors (the answer key).
    Returns the score and the recorded examples.
    """
    game = Game(1, seed=seed, colors=[(83, 83, 83)])
    dino = game.dinos[0]
    camera = Camera()
    images, answers = [], []
    action = 0
    while dino.alive and dino.score < SCORE_LIMIT:
        pair = camera.look(game)
        true = game.sensors(dino)
        seen = true if eyes is None else eyes.estimate(pair)
        if record_every and game.frame % record_every == 0:
            images.append(pair)
            answers.append(true)
        if game.frame % BRAIN_EVERY == 0:
            action = brain.decide(seen)
        if on_frame:
            on_frame(game, pair, true, seen, action)
        game.step([ACTIONS[action]])
    return dino.score, images, answers
