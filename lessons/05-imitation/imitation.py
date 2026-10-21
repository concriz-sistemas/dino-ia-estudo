"""
imitation.py — The pieces shared by the scripts of lesson 5.

- The TEACHER: the PPO agent trained in lesson 4, loaded from its weights file.
- The STUDENT: a small neural network written with numpy only (like lesson 1)
  that learns to copy whatever the teacher did in each situation.
- Demonstrations: CSV files with one row per game frame — the 6 sensors and
  the action that was taken.

The student never sees a score or a reward. It only sees "in this situation,
the teacher pressed this key". That is supervised learning, exactly like
lesson 1, with the teacher's actions as the answer key.
"""
import csv
import os
import sys

import numpy as np

FOLDER = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FOLDER, "..", ".."))
LESSON_4 = os.path.join(ROOT, "lessons", "04-reinforcement")
sys.path.insert(0, ROOT)

from dino.game import Game, NOTHING, JUMP, DUCK, SENSOR_NAMES

OUTPUT = os.path.join(FOLDER, "results")
DEMOS = os.path.join(OUTPUT, "demos")
MODELS = os.path.join(OUTPUT, "models")

ACTIONS = [NOTHING, JUMP, DUCK]
ACTION_NAMES = ["nothing", "jump", "duck"]
SCORE_LIMIT = 5000      # same cap as lesson 4
TEACHER_EVERY = 2       # the lesson 4 agent decides every 2 frames (its ACTION_REPEAT)


# ---------------------------------------------------------------------------
# The teacher: lesson 4's agent
# ---------------------------------------------------------------------------
class Teacher:
    def __init__(self, thousand_steps=2000, delay=0):
        from stable_baselines3 import PPO

        path = os.path.join(LESSON_4, "results", "models", f"ppo_{thousand_steps:05d}k.zip")
        if not os.path.exists(path):
            sys.exit(f"Teacher model not found: {path}\nTrain lesson 4 first.")
        self.model = PPO.load(path, device="cpu")
        self.name = f"lesson 4 agent, {thousand_steps}k steps" + (f", {delay} frames late" if delay else "")
        # A delay simulates reaction time: the teacher presses the key `delay` frames
        # after seeing the situation, like a person would.
        self.delay = delay

    def play(self, seed):
        """Plays one match. Returns the demonstration rows and the final score."""
        game = Game(1, seed=seed, colors=[(40, 120, 220)])
        dino = game.dinos[0]
        queue = [0] * self.delay
        action = 0
        rows = []
        while dino.alive and dino.score < SCORE_LIMIT:
            sensors = game.sensors(dino)
            if game.frame % TEACHER_EVERY == 0:
                decision, _ = self.model.predict(np.array(sensors, dtype=np.float32), deterministic=True)
                action = int(decision)
            queue.append(action)
            pressed = queue.pop(0)
            rows.append(sensors + [pressed])
            game.step([ACTIONS[pressed]])
        return rows, dino.score


# ---------------------------------------------------------------------------
# Demonstration files
# ---------------------------------------------------------------------------
def save_demos(path, matches):
    """matches: a list of lists of rows (6 sensors + action). Each match gets a number."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["match"] + [n.replace(" ", "_") for n in SENSOR_NAMES] + ["action"])
        for m, rows in enumerate(matches):
            for r in rows:
                w.writerow([m] + [f"{v:.4f}" for v in r[:6]] + [r[6]])


def load_demos(paths):
    """Returns X (sensors), y (actions) and the match number of each row."""
    X, y, match = [], [], []
    offset = 0
    for path in paths:
        with open(path) as f:
            rows = list(csv.reader(f))[1:]
        for r in rows:
            match.append(int(r[0]) + offset)
            X.append([float(v) for v in r[1:7]])
            y.append(int(r[7]))
        offset = max(match) + 1 if match else 0
    return np.array(X), np.array(y), np.array(match)


# ---------------------------------------------------------------------------
# The student: 6 sensors -> 16 hidden neurons (tanh) -> 3 actions (softmax)
# ---------------------------------------------------------------------------
class Student:
    def __init__(self, n_hidden=16, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.5, (6, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, 0.5, (n_hidden, 3))
        self.b2 = np.zeros(3)

    def probabilities(self, X):
        """The chance the student gives to each action (softmax turns scores into chances)."""
        h = np.tanh(X @ self.W1 + self.b1)
        z = h @ self.W2 + self.b2
        z = z - z.max(axis=-1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=-1, keepdims=True), h

    def act(self, sensors):
        return int(self.probabilities(np.array(sensors))[0].argmax())

    def train(self, X, y, epochs=3000, rate=0.5, X_val=None, y_val=None, log_every=50):
        """
        Plain gradient descent on cross-entropy, like lesson 1 but with 3 answers
        instead of 2. Returns the history: epoch, loss, train accuracy, validation accuracy.
        """
        answers = np.eye(3)[y]
        history = []
        for epoch in range(epochs + 1):
            p, h = self.probabilities(X)
            if epoch % log_every == 0 or epoch == epochs:
                loss = float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-9)))
                acc = float(np.mean(p.argmax(1) == y))
                val = float(np.mean(self.probabilities(X_val)[0].argmax(1) == y_val)) if X_val is not None else None
                history.append([epoch, loss, acc, val])
            if epoch == epochs:
                break
            # backpropagation, as in lesson 1
            dz = (p - answers) / len(y)
            dW2, db2 = h.T @ dz, dz.sum(0)
            dh = dz @ self.W2.T * (1 - h ** 2)
            dW1, db1 = X.T @ dh, dh.sum(0)
            self.W1 -= rate * dW1
            self.b1 -= rate * db1
            self.W2 -= rate * dW2
            self.b2 -= rate * db2
        return history

    def save(self, path):
        """The whole student fits in one small file: 6×16 + 16 + 16×3 + 3 = 163 numbers."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @classmethod
    def load(cls, path):
        s = cls()
        data = np.load(path)
        s.W1, s.b1, s.W2, s.b2 = data["W1"], data["b1"], data["W2"], data["b2"]
        return s


def play(student, seed):
    """The student plays one match on its own. Returns the score."""
    game = Game(1, seed=seed, colors=[(40, 120, 220)])
    dino = game.dinos[0]
    while dino.alive and dino.score < SCORE_LIMIT:
        game.step([ACTIONS[student.act(game.sensors(dino))]])
    return dino.score
