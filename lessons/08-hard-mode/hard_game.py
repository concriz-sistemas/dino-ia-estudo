"""
hard_game.py — A harder version of the game, for lesson 8.

Everything from dino/game.py stays the same (the original game is untouched;
lessons 1-7 still use it). Once birds start to appear, sometimes a FORMATION
shows up instead of a single obstacle:

    tunnel     — a row of birds at head height, too long to jump over at any
                 speed. The only way through: stay ducked all the way.
    staircase  — birds going down: high (75), middle (50), then three low ones (26).
                 Standing passes under the first two, the low ones need ducking,
                 and a jump anywhere in it hits a bird.
    pair       — two birds at head height, close together. Short enough to jump
                 over, low enough to duck under: TWO ways through.

All three were checked by search (verify_formations below), at every speed:
tunnel and staircase can only be passed by ducking; the pair, by ducking or by
jumping.

Run from the repository root to play it yourself:
    python lessons/08-hard-mode/hard_game.py
Controls: SPACE / ↑ = jump,  ↓ = duck,  R = restart
"""
import math
import os
import sys

FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(FOLDER, "..", "..")))

import pygame

from dino.game import (Game, Dino, Obstacle, GROUND_Y, WIDTH, HEIGHT, NOTHING, JUMP, DUCK,
                       SENSOR_NAMES)

FORMATION_CHANCE = 0.3     # once birds appear, the chance that the next obstacle is a formation
BIRD_W, BIRD_H = 40, 25
TUNNEL_ELEVATION = 30      # bird from 30 to 55 above the ground: hits a standing dino, not a ducking one
TUNNEL_SPACING = 50
STAIRCASE = [75, 50, 26, 26, 26]
STAIRCASE_SPACING = 5      # × speed, in pixels


class Bird(Obstacle):
    """A bird at any height (the original game only has 3 fixed heights)."""

    def __init__(self, kind, x, elevation):
        self.kind = kind
        self.rect = pygame.Rect(int(x), GROUND_Y - elevation - BIRD_H, BIRD_W, BIRD_H)
        self.x = float(x)


def tunnel(x, speed):
    # Long enough that crossing it takes about 30 frames: more than a jump stays high
    n = max(3, math.ceil(speed * 30 / TUNNEL_SPACING))
    return [Bird("bird_tunnel", x + i * TUNNEL_SPACING, TUNNEL_ELEVATION) for i in range(n)]


def staircase(x, speed):
    return [Bird("bird_staircase", x + i * int(speed * STAIRCASE_SPACING), e) for i, e in enumerate(STAIRCASE)]


def pair(x, speed):
    birds = [Bird("bird_pair", x + i * TUNNEL_SPACING, TUNNEL_ELEVATION) for i in range(2)]
    for b in birds:
        b.group = id(birds[0])  # which pair each bird belongs to (pair_choices.py uses it)
    return birds


FORMATIONS = {"tunnel": tunnel, "staircase": staircase, "pair": pair}


class HardGame(Game):
    def _spawn_obstacle(self):
        if self.speed > 10 and self.rng.random() < FORMATION_CHANCE:
            make = FORMATIONS[self.rng.choice(sorted(FORMATIONS))]
            self.obstacles += make(WIDTH + 10, self.speed)
            self._next_gap = self.rng.randint(int(self.speed * 30), int(self.speed * 55))
        else:
            super()._spawn_obstacle()


# ---------------------------------------------------------------------------
# Checking that every formation can be passed, and only by ducking
# ---------------------------------------------------------------------------
def can_pass(obstacles, speed, allowed=(NOTHING, JUMP, DUCK)):
    """
    Tries every sequence of keys (one key held for 2 frames at a time, like the
    agents) and says whether any of them gets the dino past all the obstacles.
    """
    boxes = [(o.rect.x, o.rect.y, o.rect.w, o.rect.h) for o in obstacles]
    end = max(x + w for x, _, w, _ in boxes)

    def crashes(dino, shift):
        hit = dino.rect.inflate(-8, -8)  # the same smaller hitbox as the game
        return any(hit.colliderect(pygame.Rect(int(x - shift), y, w, h)) for x, y, w, h in boxes)

    seen, frontier = set(), [(0, GROUND_Y, 0.0)]
    while frontier:
        new = []
        for frame, feet, vy in frontier:
            for action in allowed:
                dino = Dino()
                dino.feet, dino.vy = feet, vy
                f, alive = frame, True
                for _ in range(2):
                    dino.update(action)
                    f += 1
                    if crashes(dino, f * speed):
                        alive = False
                        break
                if not alive:
                    continue
                if end - f * speed < dino.x:
                    return True
                key = (f, round(dino.feet, 1), round(dino.vy, 1))
                if key not in seen:
                    seen.add(key)
                    new.append((f, dino.feet, dino.vy))
        frontier = new
    return False


def verify_formations(speeds=(10.5, 12, 14, 16, 18, 20)):
    print(f"  {'formation':<10} {'speed':>5}   can pass?  without ducking?  without jumping?")
    for name, make in FORMATIONS.items():
        for speed in speeds:
            obstacles = make(400, speed)
            yes = can_pass(obstacles, speed)
            no_duck = can_pass(obstacles, speed, allowed=(NOTHING, JUMP))
            no_jump = can_pass(obstacles, speed, allowed=(NOTHING, DUCK))
            print(f"  {name:<10} {speed:5}   {str(yes):<9}  {str(no_duck):<16}  {no_jump}")


def play_manual():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Dino — hard mode")
    font = pygame.font.SysFont("monospace", 16)
    clock = pygame.time.Clock()
    game = HardGame(1, colors=[(83, 83, 83)])
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                game = HardGame(1, colors=[(83, 83, 83)])
        keys = pygame.key.get_pressed()
        action = NOTHING
        if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
            action = JUMP
        elif keys[pygame.K_DOWN]:
            action = DUCK
        if game.any_alive():
            game.step([action])
        info = [f"Score: {game.dinos[0].score}", f"Speed: {game.speed:.1f}", "HARD MODE"]
        if not game.any_alive():
            info.append("GAME OVER — press R")
        game.draw(screen, font, info, highlight=game.dinos[0])
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify_formations()
    else:
        play_manual()
