"""
watch.py — The student playing, with its "mind" beside it.

Run from the repository root, after train.py:
    # the student trained on the example demonstrations
    python lessons/05-imitation/watch.py
    # any other student
    python lessons/05-imitation/watch.py --student lessons/05-imitation/results/models/student_teacher_00500k.npz
    # records results/student.gif instead of opening a window
    python lessons/05-imitation/watch.py --gif

Keys: R = new match, ESC = quit

On the right panel:
    - the chance the student gives to each action (the biggest one is pressed);
    - what the TEACHER (lesson 4's final agent) would press in the same situation,
      and how often, so far, the student agreed with it.
"""
import argparse
import os
import random
import sys

FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FOLDER)

import numpy as np
import pygame

from imitation import MODELS, ACTIONS, ACTION_NAMES, SCORE_LIMIT, Student, Teacher
from dino.game import Game, WIDTH, HEIGHT

DEFAULT_STUDENT = os.path.join(MODELS, "student_teacher_02000k.npz")
PANEL = 300
COLORS = [(150, 150, 150), (224, 49, 49), (25, 113, 194)]


def draw_panel(screen, font, probs, teacher_action, agreement):
    x0 = WIDTH
    pygame.draw.rect(screen, (235, 235, 240), (x0, 0, PANEL, HEIGHT))
    screen.blit(font.render("Student: chance of each action", True, (40, 40, 40)), (x0 + 15, 12))
    choice = int(probs.argmax())
    for i, (name, p) in enumerate(zip(ACTION_NAMES, probs)):
        y = 40 + i * 30
        screen.blit(font.render(name, True, (40, 40, 40)), (x0 + 15, y))
        pygame.draw.rect(screen, (210, 210, 215), (x0 + 95, y, 150, 18))
        pygame.draw.rect(screen, COLORS[i], (x0 + 95, y, int(150 * p), 18))
        if i == choice:
            pygame.draw.rect(screen, (255, 190, 0), (x0 + 93, y - 2, 154, 22), 2)
        screen.blit(font.render(f"{p:4.0%}", True, (40, 40, 40)), (x0 + 252, y))
    screen.blit(font.render(f"Teacher would press: {ACTION_NAMES[teacher_action]}", True, (40, 40, 40)),
                (x0 + 15, 150))
    same = "same" if teacher_action == choice else "DIFFERENT"
    screen.blit(font.render(f"  -> {same}", True, (47, 158, 68) if same == "same" else (224, 49, 49)),
                (x0 + 15, 172))
    screen.blit(font.render(f"Agreement so far: {agreement:.0%}", True, (40, 40, 40)), (x0 + 15, 210))


class Match:
    """One match played by the student, with the teacher watching."""

    def __init__(self, student, teacher, seed):
        self.student, self.teacher = student, teacher
        self.game = Game(1, seed=seed, colors=[(40, 120, 220)])
        self.agree = self.total = 0
        self.teacher_action = 0

    def step(self):
        dino = self.game.dinos[0]
        sensors = self.game.sensors(dino)
        probs = self.student.probabilities(np.array(sensors))[0]
        if self.game.frame % 2 == 0:
            decision, _ = self.teacher.model.predict(np.array(sensors, dtype=np.float32), deterministic=True)
            self.teacher_action = int(decision)
        if dino.alive and dino.score < SCORE_LIMIT:
            self.agree += int(probs.argmax() == self.teacher_action)
            self.total += 1
            self.game.step([ACTIONS[int(probs.argmax())]])
        return probs

    def draw(self, screen, font, probs, title):
        dino = self.game.dinos[0]
        info = [title, f"Score: {dino.score}"] + ([] if dino.alive else ["Crashed!"])
        self.game.draw(screen, font, info, highlight=dino)
        draw_panel(screen, font, probs, self.teacher_action, self.agree / max(self.total, 1))


def record_gif(student, teacher, title, path):
    from PIL import Image

    pygame.init()
    screen = pygame.Surface((WIDTH + PANEL, HEIGHT))
    font = pygame.font.SysFont("monospace", 14)
    match = Match(student, teacher, seed=7)
    frames, after_crash = [], 0
    while match.game.frame < 700 and after_crash < 30:
        probs = match.step()
        if not match.game.dinos[0].alive:
            after_crash += 1
        if match.game.frame % 4 == 0 or after_crash:
            match.draw(screen, font, probs, title)
            img = Image.frombytes("RGB", screen.get_size(), pygame.image.tobytes(screen, "RGB"))
            frames.append(img.resize((img.width * 3 // 5, img.height * 3 // 5)).quantize(colors=64))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=66, loop=0, optimize=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student", default=DEFAULT_STUDENT, help="a .npz file saved by train.py")
    p.add_argument("--gif", action="store_true", help="record results/student.gif instead of opening a window")
    args = p.parse_args()

    if not os.path.exists(args.student):
        print(f"Student not found: {args.student}\nRun first: python lessons/05-imitation/train.py")
        return
    student = Student.load(args.student)
    teacher = Teacher(2000)
    title = "Student: " + os.path.basename(args.student)[len("student_"):-len(".npz")]

    if args.gif:
        path = os.path.join(FOLDER, "results", "student.gif")
        record_gif(student, teacher, title, path)
        print(f"Animation saved to: {path}")
        return

    pygame.init()
    screen = pygame.display.set_mode((WIDTH + PANEL, HEIGHT))
    pygame.display.set_caption("Dino AI — the student imitating its teacher")
    font = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()
    match = Match(student, teacher, random.randint(0, 10**9))
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                match = Match(student, teacher, random.randint(0, 10**9))
        probs = match.step()
        match.draw(screen, font, probs, title)
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
