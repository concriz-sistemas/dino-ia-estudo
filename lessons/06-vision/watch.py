"""
watch.py — Eyes + brain playing from the screen, with what the eyes see below.

Run from the repository root:
    python lessons/06-vision/watch.py
    # records results/eyes.gif instead of opening a window
    python lessons/06-vision/watch.py --gif

Keys: R = new match, ESC = quit

On the game: the ORANGE line is where the eyes think the next obstacle starts.
Below the game:
    - the small image the eyes receive (shrunk, grayscale, dark things bright);
    - the 6 sensor numbers: the TRUE ones from the game and the ones the EYES estimated.
The brain only gets the eyes' numbers.
"""
import argparse
import os
import random
import sys

FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FOLDER)

import numpy as np
import pygame

from vision import (EYES_FILE, SENSOR_NAMES, SCORE_LIMIT, BRAIN_EVERY, ACTIONS, IMG_W, IMG_H,
                    Game, WIDTH, HEIGHT, Brain, Camera, Dino, load_eyes)

PANEL = 130
ORANGE = (232, 89, 12)


class Match:
    def __init__(self, eyes, brain, seed):
        self.eyes, self.brain = eyes, brain
        self.game = Game(1, seed=seed, colors=[(40, 120, 220)])
        self.camera = Camera()
        self.action = 0

    def step(self):
        dino = self.game.dinos[0]
        self.pair = self.camera.look(self.game)
        self.true = self.game.sensors(dino)
        self.seen = self.eyes.estimate(self.pair)
        if dino.alive and dino.score < SCORE_LIMIT:
            if self.game.frame % BRAIN_EVERY == 0:
                self.action = self.brain.decide(self.seen)
            self.game.step([ACTIONS[self.action]])

    def draw(self, screen, font):
        dino = self.game.dinos[0]
        info = ["Eyes + brain: playing from the screen", f"Score: {dino.score}"]
        if not dino.alive:
            info.append("Crashed!")
        self.game.draw(screen, font, info, highlight=None)
        # Where the eyes think the next obstacle starts (distance is measured from the dino's front)
        if self.seen[0] < 0.9:
            x = int(dino.x + Dino.STANDING_W + self.seen[0] * WIDTH)
            pygame.draw.line(screen, ORANGE, (x, 60), (x, 250), 2)

        pygame.draw.rect(screen, (235, 235, 240), (0, HEIGHT, WIDTH, PANEL))
        img = np.repeat(self.pair[1][:, :, None], 3, axis=2)
        surf = pygame.surfarray.make_surface(img.transpose(1, 0, 2))
        screen.blit(pygame.transform.scale(surf, (IMG_W * 2, IMG_H * 2)), (10, HEIGHT + 30))
        screen.blit(font.render("what the eyes receive", True, (40, 40, 40)), (10, HEIGHT + 8))
        x0 = IMG_W * 2 + 40
        screen.blit(font.render("sensor        true   eyes", True, (40, 40, 40)), (x0, HEIGHT + 8))
        for i, (name, t, s) in enumerate(zip(SENSOR_NAMES, self.true, self.seen)):
            color = ORANGE if abs(t - s) > 0.05 else (40, 40, 40)
            screen.blit(font.render(f"{name:<12} {t:5.2f}  {s:5.2f}", True, color), (x0, HEIGHT + 28 + i * 16))


def record_gif(eyes, brain, path):
    from PIL import Image

    pygame.init()
    screen = pygame.Surface((WIDTH, HEIGHT + PANEL))
    font = pygame.font.SysFont("monospace", 14)
    match = Match(eyes, brain, seed=7)
    frames = []
    while match.game.frame < 900:
        match.step()
        if match.game.frame % 4 == 0:
            match.draw(screen, font)
            img = Image.frombytes("RGB", screen.get_size(), pygame.image.tobytes(screen, "RGB"))
            frames.append(img.resize((img.width * 3 // 5, img.height * 3 // 5)).quantize(colors=64))
        if not match.game.dinos[0].alive:
            break
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=66, loop=0, optimize=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gif", action="store_true", help="record results/eyes.gif instead of opening a window")
    args = p.parse_args()
    if not os.path.exists(EYES_FILE):
        print(f"Eyes not found: {EYES_FILE}\nRun first: python lessons/06-vision/train_eyes.py")
        return
    eyes, brain = load_eyes(), Brain()

    if args.gif:
        path = os.path.join(FOLDER, "results", "eyes.gif")
        record_gif(eyes, brain, path)
        print(f"Animation saved to: {path}")
        return

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT + PANEL))
    pygame.display.set_caption("Dino AI — playing from the screen")
    font = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()
    match = Match(eyes, brain, random.randint(0, 10**9))
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                match = Match(eyes, brain, random.randint(0, 10**9))
        match.step()
        match.draw(screen, font)
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
