"""
planner.py — A teacher that doesn't learn: it PLANS.

At each decision, it looks at the obstacles on the screen, imagines the next
frames with the game's own physics (gravity, jump force, speed) and searches
for a sequence of keys that gets the dino through without crashing. It then
presses the first key of that plan, and keeps following the plan until a new
obstacle shows up.

It never trains and has no weights. It is slow (it searches), but it can find
moves that a learning agent would only stumble on by luck, like ducking all
the way through a tunnel. That makes it a good teacher for imitation.

When several keys work, it prefers, in this order: do nothing, duck, jump.
"""
import os
import sys

FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(FOLDER, "..", "..")))

import pygame

from dino.game import Dino, NOTHING, JUMP, DUCK, ACCELERATION, MAX_SPEED

PREFERENCE = [NOTHING, DUCK, JUMP]
HOLD = 2           # each key is held for 2 frames, like the agents of lessons 4 and 5
HORIZON = 80       # frames imagined ahead


class Planner:
    def __init__(self):
        self.plan = []
        self.known = None

    def act(self, game):
        dino = game.dinos[0]
        # Replan when the obstacles in view change (a new one appeared) or the plan ran out
        visible = tuple(id(o) for o in game.obstacles)
        if not self.plan or visible != self.known:
            self.plan = search(game, dino) or [JUMP]
            self.known = visible
        return self.plan.pop(0)


def search(game, dino, preference=PREFERENCE):
    """
    Breadth-first search over key sequences. Returns the keys (one per frame) of a safe plan,
    the first one found in order of preference, or None if every sequence crashes.
    """
    boxes = [(o.x, o.rect.y, o.rect.w, o.rect.h) for o in game.obstacles]
    speeds, s = [], game.speed
    for _ in range(HORIZON + HOLD):
        speeds.append(s)
        s = min(MAX_SPEED, s + ACCELERATION)
    shifts = [0.0]
    for sp in speeds:
        shifts.append(shifts[-1] + sp)

    def crashes(d, frame):
        hit = d.rect.inflate(-8, -8)
        return any(hit.colliderect(pygame.Rect(int(x - shifts[frame]), y, w, h)) for x, y, w, h in boxes)

    frontier = [(dino.feet, dino.vy, [])]
    seen = set()
    for depth in range(0, HORIZON, HOLD):
        new = []
        for feet, vy, keys in frontier:
            for key in preference:
                d = Dino()
                d.feet, d.vy = feet, vy
                alive = True
                for k in range(HOLD):
                    d.update(key)
                    if crashes(d, depth + k + 1):
                        alive = False
                        break
                if not alive:
                    continue
                state = (depth, round(d.feet, 1), round(d.vy, 1))
                if state in seen:
                    continue
                seen.add(state)
                new.append((d.feet, d.vy, keys + [key] * HOLD))
        if not new:
            return None
        frontier = new
    return frontier[0][2]   # the first surviving plan found, in order of preference
