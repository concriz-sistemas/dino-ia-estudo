"""
experiments.py — Is the student as good as its teacher? Four teachers, four students.

For each teacher, the script:
    1. records its demonstrations (in memory, nothing is saved to demos/);
    2. trains a student on them, exactly like train.py;
    3. lets the teacher AND the student play the same 10 new matches.

Run from the repository root (about 1 minute):
    python lessons/05-imitation/experiments.py

Output: results/experiments.png and a table in the terminal.
"""
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FOLDER)
from imitation import OUTPUT, Teacher, Student, play

N_TEST_MATCHES = 10

# (label, lesson 4 model in thousands of steps, reaction delay in frames, matches recorded)
EXPERIMENTS = [
    ("strong teacher,\n1 match recorded", 2000, 0, 1),
    ("weak teacher\n(500k steps)", 500, 0, 10),
    ("strong teacher,\n8 frames late", 2000, 8, 10),
    ("strong teacher,\n12 frames late", 2000, 12, 10),
]


def main():
    results = []
    for label, model, delay, n_matches in EXPERIMENTS:
        teacher = Teacher(model, delay)
        start = time.time()
        rows = [r for m in range(n_matches) for r in teacher.play(seed=1000 + m)[0]]
        data = np.array(rows, dtype=float)
        student = Student()
        student.train(data[:, :6], data[:, 6].astype(int))
        seconds = time.time() - start

        seeds = [5000 + s for s in range(N_TEST_MATCHES)]
        teacher_scores = [teacher.play(seed)[1] for seed in seeds]
        student_scores = [play(student, seed) for seed in seeds]
        results.append((label, np.mean(teacher_scores), np.mean(student_scores), len(rows), seconds))
        print(f"{label.replace(chr(10), ' '):<36} examples: {len(rows):6d}   "
              f"teacher: {np.mean(teacher_scores):6.0f}   student: {np.mean(student_scores):6.0f}   "
              f"({seconds:.0f} s to record and train)")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(results))
    ax.bar(x - 0.2, [r[1] for r in results], 0.4, color="#868e96", label="teacher")
    ax.bar(x + 0.2, [r[2] for r in results], 0.4, color="#1c7ed6", label="student")
    for i, r in enumerate(results):
        ax.text(i - 0.2, r[1] + 60, f"{r[1]:.0f}", ha="center", fontsize=8)
        ax.text(i + 0.2, r[2] + 60, f"{r[2]:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, [r[0] for r in results], fontsize=9)
    ax.set_ylabel(f"mean score in {N_TEST_MATCHES} new matches")
    ax.set_ylim(0, 5600)
    ax.set_title("The student copies its teacher: good, bad, and sometimes better")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(OUTPUT, "experiments.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"\nChart saved to: {path}")


if __name__ == "__main__":
    main()
