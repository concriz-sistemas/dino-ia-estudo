"""
Lesson 2 — Play yourself and see, in the corner of the screen, the 6 numbers
the AI will receive instead of eyes.

Run from the repository root:
    python lessons/02-the-environment/play.py

Controls: SPACE / ↑ = jump,  ↓ = duck,  R = restart
"""
import os
import sys

# Lets us import the `dino/` package that lives at the repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dino.game import play_manual

if __name__ == "__main__":
    play_manual()
