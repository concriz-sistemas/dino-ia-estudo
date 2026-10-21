"""
Lesson 5 — Records demonstrations: who plays, what they see and what they press.

Run from the repository root:
    # YOU play; every frame is recorded (sensors + the key you pressed)
    python lessons/05-imitation/record.py
    # lesson 4's agent plays 10 matches and is recorded
    python lessons/05-imitation/record.py --teacher
    # a weaker teacher: lesson 4's agent after only 500k training steps
    python lessons/05-imitation/record.py --teacher --model 500
    # a teacher with human-like reaction time: presses the key 8 frames late
    python lessons/05-imitation/record.py --teacher --delay 8

Keys when you play:  SPACE / ↑ = jump,  ↓ = duck,  R = next match,  ESC = save and quit

Each run writes one CSV to results/demos/, one row per game frame.
"""
import argparse
import os
import sys
import time

FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FOLDER)

from imitation import DEMOS, ACTIONS, ACTION_NAMES, SCORE_LIMIT, Teacher, save_demos
from dino.game import Game, WIDTH, HEIGHT, NOTHING, JUMP, DUCK


def record_human():
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Dino AI — recording your demonstrations")
    font = pygame.font.SysFont("monospace", 16)
    clock = pygame.time.Clock()

    matches, rows = [], []
    game = Game(1, colors=[(83, 83, 83)])
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                if rows:
                    matches.append(rows)
                pygame.quit()
                return matches
            if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                if rows:
                    matches.append(rows)
                rows = []
                game = Game(1, colors=[(83, 83, 83)])

        keys = pygame.key.get_pressed()
        action = NOTHING
        if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
            action = JUMP
        elif keys[pygame.K_DOWN]:
            action = DUCK

        dino = game.dinos[0]
        if dino.alive and dino.score < SCORE_LIMIT:
            # The row pairs what you SAW with what you PRESSED in the same frame
            rows.append(game.sensors(dino) + [ACTIONS.index(action)])
            game.step([action])

        total = sum(len(m) for m in matches) + len(rows)
        counts = [sum(r[6] == a for m in matches + [rows] for r in m) for a in range(3)]
        info = [f"Recording: {total} examples, {len(matches) + 1} match(es)",
                "  " + "  ".join(f"{n} {c / max(total, 1):.0%}" for n, c in zip(ACTION_NAMES, counts)),
                f"Score: {dino.score}"]
        if not dino.alive:
            info.append("Crashed! R = next match, ESC = save and quit")
        game.draw(screen, font, info, highlight=dino)
        pygame.display.flip()
        clock.tick(60)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", action="store_true", help="record lesson 4's agent instead of you")
    p.add_argument("--model", type=int, default=2000, help="which lesson 4 model, in thousands of steps")
    p.add_argument("--delay", type=int, default=0, help="teacher's reaction delay, in frames")
    p.add_argument("--matches", type=int, default=10, help="how many matches the teacher plays")
    args = p.parse_args()

    if args.teacher:
        teacher = Teacher(args.model, args.delay)
        print(f"Recording the teacher ({teacher.name}) for {args.matches} matches...")
        matches = []
        for m in range(args.matches):
            rows, score = teacher.play(seed=1000 + m)
            matches.append(rows)
            print(f"  match {m + 1}: {score} points")
        name = f"teacher_{args.model:05d}k" + (f"_delay{args.delay}" if args.delay else "")
    else:
        matches = record_human()
        name = "human_" + time.strftime("%Y%m%d_%H%M%S")

    if not matches:
        print("Nothing was recorded.")
        return
    path = os.path.join(DEMOS, name + ".csv")
    save_demos(path, matches)
    total = sum(len(m) for m in matches)
    print(f"\nSaved {total} examples from {len(matches)} match(es) to: {path}")
    print("Next step:")
    print(f"    python lessons/05-imitation/train.py --demos {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
