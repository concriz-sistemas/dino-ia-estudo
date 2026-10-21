"""
watch.py — Watch the hard game: lesson 4's agent and lesson 8's student, on the same match.

Run from the repository root:
    # lesson 4's agent on top, lesson 8's student below, same obstacles
    python lessons/08-hard-mode/watch.py
    # records results/before_after.gif instead of opening a window
    python lessons/08-hard-mode/watch.py --gif

Keys: R = new match, ESC = quit
"""
import argparse
import os
import random
import sys

FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FOLDER)

import pygame

from agents import Lesson4Agent, Student8
from hard_game import HardGame
from dino.game import WIDTH, HEIGHT


class Pair:
    """Two agents, each in its own copy of the same match."""

    def __init__(self, agents, seed):
        self.agents = agents
        self.games = [HardGame(1, seed=seed, colors=[c]) for c in [(232, 89, 12), (40, 120, 220)]]

    def step(self):
        for agent, game in zip(self.agents, self.games):
            dino = game.dinos[0]
            if dino.alive and dino.score < 5000:
                game.step([agent.act(game)])

    def draw(self, screen, font):
        for row, (agent, game) in enumerate(zip(self.agents, self.games)):
            dino = game.dinos[0]
            surf = pygame.Surface((WIDTH, HEIGHT))
            info = [agent.name, f"Score: {dino.score}"] + ([] if dino.alive else ["Crashed!"])
            game.draw(surf, font, info, highlight=dino)
            screen.blit(surf, (0, row * HEIGHT))
        pygame.draw.line(screen, (120, 120, 120), (0, HEIGHT), (WIDTH, HEIGHT), 2)

    def done(self):
        return all(not g.dinos[0].alive or g.dinos[0].score >= 5000 for g in self.games)


def record_gif(agents, path, seed=5000, start=650, end=1700):
    from PIL import Image

    pygame.init()
    screen = pygame.Surface((WIDTH, 2 * HEIGHT))
    font = pygame.font.SysFont("monospace", 16)
    pair = Pair(agents, seed)
    frames = []
    while pair.games[1].frame < end and not pair.done():
        pair.step()
        frame = max(g.frame for g in pair.games)
        if frame >= start and frame % 3 == 0:
            pair.draw(screen, font)
            img = Image.frombytes("RGB", screen.get_size(), pygame.image.tobytes(screen, "RGB"))
            frames.append(img.resize((img.width * 3 // 5, img.height * 3 // 5)).quantize(colors=48))
    frames += [frames[-1]] * 15
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=50, loop=0, optimize=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gif", action="store_true", help="record results/before_after.gif")
    args = p.parse_args()
    agents = [Lesson4Agent(), Student8()]

    if args.gif:
        path = os.path.join(FOLDER, "results", "before_after.gif")
        record_gif(agents, path)
        print(f"Animation saved to: {path}")
        return

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, 2 * HEIGHT))
    pygame.display.set_caption("Dino AI — hard mode: before and after")
    font = pygame.font.SysFont("monospace", 16)
    clock = pygame.time.Clock()
    pair = Pair(agents, random.randint(0, 10**9))
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                pair = Pair(agents, random.randint(0, 10**9))
        pair.step()
        pair.draw(screen, font)
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
