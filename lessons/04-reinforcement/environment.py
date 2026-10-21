"""
environment.py — The dino game in the standard reinforcement learning format (Gymnasium).

Every RL environment answers two questions:
    reset()       -> starts a new match and returns the first OBSERVATION
    step(action)  -> applies the ACTION and returns (observation, REWARD, done?, ...)

It is the same game from lesson 2, with a single dino. Any RL algorithm that
"speaks Gymnasium" (PPO, DQN, A2C...) can play it without knowing anything about dinos.
"""
import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dino.game import Game, WIDTH, HEIGHT, NOTHING, JUMP, DUCK

ACTIONS = [NOTHING, JUMP, DUCK]
ACTION_NAMES = ["nothing", "jump", "duck"]
SCORE_LIMIT = 5000              # the match ends here, even without crashing

# The REWARD is RL's only "teacher". Change it and see what happens!
REWARD_ALIVE = 0.1              # per frame survived
REWARD_CRASH = -1.0             # when crashing
JUMP_COST = 0.0                 # try 0.01: the dino learns not to jump for no reason

# Each decision of the agent lasts 2 game frames (DeepMind used 4 on Atari).
# Fewer decisions per match = easier to link the action to its consequence.
ACTION_REPEAT = 2


class DinoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}  # 30 steps x 2 frames = 60 frames/s

    def __init__(self, render_mode=None):
        # OBSERVATION: the same 6 sensors from the previous lessons
        self.observation_space = spaces.Box(low=-1.0, high=2.0, shape=(6,), dtype=np.float32)
        # ACTION: 0 = nothing, 1 = jump, 2 = duck
        self.action_space = spaces.Discrete(3)
        self.render_mode = render_mode
        self.game = None
        self.screen = None
        self.font = None
        self.info_lines = []

    def _observe(self):
        return np.array(self.game.sensors(self.game.dinos[0]), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Different obstacles every match: the agent has to GENERALIZE, not memorize
        game_seed = int(self.np_random.integers(0, 2**31))
        self.game = Game(1, seed=game_seed, colors=[(40, 120, 220)])
        return self._observe(), {}

    def step(self, action):
        dino = self.game.dinos[0]
        reward = 0.0
        for _ in range(ACTION_REPEAT):
            self.game.step([ACTIONS[int(action)]])
            if not dino.alive or dino.score >= SCORE_LIMIT:
                break
            reward += REWARD_ALIVE
        crashed = not dino.alive
        if crashed:
            reward += REWARD_CRASH
        if action == 1:
            reward -= JUMP_COST

        terminated = crashed                            # a "real" ending: it lost
        truncated = dino.score >= SCORE_LIMIT           # ended by the time limit
        info = {"score": dino.score}
        if self.render_mode == "human":
            self.render()
        return self._observe(), reward, terminated, truncated, info

    def render(self):
        import pygame

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
                pygame.display.set_caption("Dino AI — reinforcement learning")
            else:
                self.screen = pygame.Surface((WIDTH, HEIGHT))
            self.font = pygame.font.SysFont("monospace", 16)
            self.clock = pygame.time.Clock()
        dino = self.game.dinos[0]
        self.game.draw(self.screen, self.font,
                       [f"Score: {dino.score}", f"Speed: {self.game.speed:.1f}"] + self.info_lines,
                       highlight=dino)
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.quit()
            self.screen = None
