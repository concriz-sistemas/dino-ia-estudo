"""
agents.py — Every "player" of lesson 8 behind the same interface: agent.act(game) -> key.

    Lesson3Champion  — the last champion of lesson 3 (neuroevolution)
    Lesson4Agent     — lesson 4's PPO agent (reinforcement learning), or any PPO file
    Lesson5Student   — lesson 5's student (imitation of lesson 4's agent)
    Planner          — the search of planner.py (no learning)
    Coach            — lesson 4's agent, with the planner taking over in the new formations
    Student8         — lesson 8's student, trained to imitate the coach (train_student.py)

They all decide every 2 frames and hold the key in between, like in lessons 4 and 5.
"""
import os
import pickle
import sys

import numpy as np

FOLDER = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FOLDER, "..", ".."))
LESSONS = os.path.join(ROOT, "lessons")
sys.path.insert(0, ROOT)
sys.path.insert(0, FOLDER)

import torch
import torch.nn as nn

from dino.game import NOTHING, JUMP, DUCK
import planner

ACTIONS = [NOTHING, JUMP, DUCK]
NEW_FORMATIONS = {"bird_tunnel", "bird_staircase", "bird_pair"}
LESSON_4_FILE = os.path.join(LESSONS, "04-reinforcement", "results", "models", "ppo_02000k.zip")
STUDENT_8_FILE = os.path.join(FOLDER, "results", "student.pt")


class EveryTwoFrames:
    """Decides on even frames and holds the key on odd ones."""

    def __init__(self):
        self.key = NOTHING

    def act(self, game):
        if game.frame % 2 == 0:
            self.key = self.decide(game)
        return self.key


class Lesson3Champion(EveryTwoFrames):
    name = "lesson 3: neuroevolution champion"

    def __init__(self):
        super().__init__()
        import neat

        path = os.path.join(LESSONS, "03-neuroevolution", "results", "champions.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError("Run lesson 3 first: python lessons/03-neuroevolution/train.py --fast")
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                             neat.DefaultStagnation, os.path.join(LESSONS, "03-neuroevolution", "neat-config.txt"))
        with open(path, "rb") as f:
            self.net = neat.nn.FeedForwardNetwork.create(pickle.load(f)[-1], config)

    def decide(self, game):
        out = self.net.activate(game.sensors(game.dinos[0]))
        return ACTIONS[out.index(max(out))]


class Lesson4Agent(EveryTwoFrames):
    name = "lesson 4: reinforcement learning (PPO)"

    def __init__(self, path=LESSON_4_FILE):
        super().__init__()
        from stable_baselines3 import PPO

        self.model = PPO.load(path, device="cpu")

    def probabilities(self, game):
        obs = torch.tensor(game.sensors(game.dinos[0]), dtype=torch.float32)[None]
        with torch.no_grad():
            return self.model.policy.get_distribution(obs).distribution.probs[0].numpy()

    def decide(self, game):
        action, _ = self.model.predict(np.array(game.sensors(game.dinos[0]), dtype=np.float32), deterministic=True)
        return ACTIONS[int(action)]


class Lesson5Student(EveryTwoFrames):
    name = "lesson 5: imitation student"

    def __init__(self):
        super().__init__()
        sys.path.insert(0, os.path.join(LESSONS, "05-imitation"))
        from imitation import Student

        self.student = Student.load(os.path.join(LESSONS, "05-imitation", "results", "models",
                                                 "student_teacher_02000k.npz"))

    def decide(self, game):
        return ACTIONS[self.student.act(game.sensors(game.dinos[0]))]


class Planner(EveryTwoFrames):
    name = "a planner (search, no learning)"

    def __init__(self):
        super().__init__()
        self.inner = planner.Planner()

    def act(self, game):  # the planner plans frame by frame
        return self.inner.act(game)


class PlannerTeacher(EveryTwoFrames):
    """The planner alone as a teacher: at each decision, the first key of its plan."""
    name = "the planner alone, as a teacher"

    def decide(self, game):
        dino = game.dinos[0]
        if not any(o.rect.right > dino.x and o.rect.left - dino.x < 500 for o in game.obstacles):
            return NOTHING  # nothing nearby: nothing to decide
        plan = planner.search(game, dino)
        return plan[0] if plan else JUMP


class Coach(EveryTwoFrames):
    """
    The teacher of lesson 8: lesson 4's agent plays as usual, but when one of the
    new formations is close, the planner takes over (preferring to duck early).
    """
    name = "the coach: lesson 4's agent + the planner in the new formations"

    def __init__(self):
        super().__init__()
        self.agent = Lesson4Agent()

    @staticmethod
    def formation_near(game):
        dino = game.dinos[0]
        return any(o.kind in NEW_FORMATIONS and o.rect.left - dino.x < 350 and o.rect.right > dino.x - 10
                   for o in game.obstacles)

    def decide(self, game):
        if self.formation_near(game):
            plan = planner.search(game, game.dinos[0], preference=[DUCK, NOTHING, JUMP])
            if plan:
                return plan[0]
        return self.agent.decide(game)


class StudentNet(nn.Module):
    """6 sensors -> 64 -> 64 -> 3 actions. A bit bigger than lesson 5's student."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 3))

    def forward(self, x):
        return self.net(x)


class Student8(EveryTwoFrames):
    name = "lesson 8: student of the coach"

    def __init__(self, net=None, path=STUDENT_8_FILE):
        super().__init__()
        if net is None:
            net = StudentNet()
            net.load_state_dict(torch.load(path, map_location="cpu"))
        self.net = net.eval()

    def probabilities(self, game):
        with torch.no_grad():
            x = torch.tensor(game.sensors(game.dinos[0]), dtype=torch.float32)[None]
            return torch.softmax(self.net(x), dim=1)[0].numpy()

    def decide(self, game):
        return ACTIONS[int(self.probabilities(game).argmax())]


def play(agent, game_class, seed):
    """One match. Returns the score and what the dino hit ('-' if it reached the score limit)."""
    game = game_class(1, seed=seed)
    dino = game.dinos[0]
    while dino.alive and dino.score < 5000:
        game.step([agent.act(game)])
    hit = game.next_obstacle(dino)
    return dino.score, (hit.kind if hit is not None and not dino.alive else "-")
