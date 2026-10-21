"""
pair_choices.py — Two ways through: does each agent jump over the pair, or duck under it?

The "pair" formation (two birds at head height, close together) can be passed
either way: the search in hard_game.py checked both at every speed. When an
agent gets through one, this script looks at what the dino was doing while the
birds were over it: in the air = jumped, ducking on the ground = ducked.

Run from the repository root, after train_student.py (a few minutes):
    python lessons/08-hard-mode/pair_choices.py
"""
import os
import sys
from collections import Counter

FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FOLDER)
from agents import Lesson4Agent, Coach, Student8
from hard_game import HardGame

SEEDS = [6000 + s for s in range(12)]


def pair_choices(agent):
    choices = Counter()
    for seed in SEEDS:
        game = HardGame(1, seed=seed)
        dino = game.dinos[0]
        watching = {}   # pair -> what the dino did while its birds were over it
        passed = set()  # pairs completely behind the dino while it was still alive
        while dino.alive and dino.score < 5000:
            game.step([agent.act(game)])
            for o in game.obstacles:
                if o.kind != "bird_pair":
                    continue
                if o.rect.left < dino.x + dino.rect.width and o.rect.right > dino.x:
                    watching.setdefault(o.group, set()).add("jumped" if not dino.on_ground else
                                                            ("ducked" if dino.ducking else "stood"))
                elif o.rect.right <= dino.x and dino.alive and o.group in watching:
                    passed.add(o.group)
        hit = game.next_obstacle(dino)
        if not dino.alive and hit is not None and hit.kind == "bird_pair":
            passed.discard(hit.group)  # it crashed into this pair: it didn't get through
        for group in passed:
            did = watching[group]
            choices["jumped" if "jumped" in did else "ducked" if "ducked" in did else "other"] += 1
    return choices


def main():
    for make in [Lesson4Agent, Coach, Student8]:
        agent = make()
        c = pair_choices(agent)
        total = c["jumped"] + c["ducked"]
        print(f"{agent.name:<62} pairs passed: {total:3d}   jumped: {c['jumped']:3d}   ducked: {c['ducked']:3d}")


if __name__ == "__main__":
    main()
