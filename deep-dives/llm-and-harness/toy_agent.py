"""
toy_agent.py — A toy agent: a harness, two tools and a fake model.

Run from the repository root (a few seconds):
    python deep-dives/llm-and-harness/toy_agent.py
    # your own question
    python deep-dives/llm-and-harness/toy_agent.py --question "how many letters in transformer?"
    # lesson 7's model in the place of the fake one (train lesson 7 first)
    python deep-dives/llm-and-harness/toy_agent.py --tiny

What it does:
    1. runs the fake model in the harness and prints every step;
    2. runs lesson 7's TinyGPT in the SAME harness, to show what happens
       when the model never learned the format;
    3. draws the three loops, and an animation of the agent loop.

Outputs (in charts/):
    00_three_loops.png   — the token loop inside the agent loop inside the session
    01_agent_loop.gif    — who does what, step by step
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from harness import OUTPUT, fake_model, lesson7_model, run

INK, MUTED = "#0b0b0b", "#898781"
COLORS = {"user": "#52514e", "model": "#2a78d6", "tool": "#1baf7a", "harness": "#eb6834"}
EXPLAIN = {
    "user": "The user asks. The harness starts the transcript with the question.",
    "model": "The model reads the WHOLE transcript and writes one line of text.",
    "tool": "The harness sees 'call ...' and runs the tool: ordinary Python code.",
    "harness": "The harness pastes the result into the transcript, and asks the model again.",
    "done": "The model wrote 'answer: ...', so the harness stops the loop.",
}


def show_run(model, name, question):
    print(f"\n=== {name} ===")
    print(f"  user:          'q: {question}'")
    transcript, steps = run(model, question, verbose=True)
    print("  stopped: " + ("the model answered" if steps[-1][0] == "model" else "out of turns"))
    return steps


def draw_three_loops(path):
    fig = plt.figure(figsize=(9, 5.6))
    ax = fig.add_axes([0, 0, 1, 0.93])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")
    boxes = [
        (2, 1, 96, 60, MUTED, "3. The session loop (harness)",
         "Many questions over hours. When the transcript gets too long, the harness\n"
         "summarizes it. Between sessions, memory is a text file it pastes back in."),
        (6, 5, 88, 38, COLORS["harness"], "2. The agent loop (harness)",
         "the model writes a line -> 'call ...'? the harness runs the tool\n"
         "-> pastes 'result: ...' -> asks the model again, until 'answer: ...'"),
        (10, 9, 80, 16, COLORS["model"], "1. The token loop (inside the model)",
         "next_probabilities -> draw one token -> append it -> again\n"
         "(lesson 7's generate)"),
    ]
    for x, y, w, h, color, title, text in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2",
                                    fc=color + "12", ec=color, lw=2))
        ax.text(x + 2.5, y + h - 2.5, title, fontsize=12, weight="bold", color=INK, va="top")
        ax.text(x + 2.5, y + h - 7, text, fontsize=9.5, color="#52514e", va="top", family="monospace",
                linespacing=1.5)
    fig.suptitle("Three loops, one inside the other", fontsize=12)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def draw_agent_gif(steps, path):
    """Left: the transcript as it grows. Right: who is acting right now."""
    frames = []                                  # (lines written so far, who is acting)
    lines = []
    for who, text in steps:
        if who != "tool":
            lines.append((who, text))
        frames.append((list(lines), who))
    frames.append((list(lines), "done"))

    fig = plt.figure(figsize=(10, 4.6))
    ax_text = fig.add_axes([0.02, 0.17, 0.5, 0.72])
    ax_who = fig.add_axes([0.55, 0.17, 0.43, 0.72])
    actors = {"model": (50, 85), "harness": (50, 45), "tool": (50, 8)}

    def draw(k):
        shown, acting = frames[k]
        for ax in (ax_text, ax_who):
            ax.clear()
            ax.axis("off")
        ax_text.set_xlim(0, 1)
        ax_text.set_ylim(0, 1)
        ax_text.set_title("The transcript: the only thing the model ever sees", fontsize=10)
        for i, (who, text) in enumerate(shown):
            ax_text.text(0.02, 0.9 - i * 0.13, text, family="monospace", fontsize=13,
                         color=COLORS[who], weight="bold" if i == len(shown) - 1 and acting != "done" else "normal")
        ax_who.set_xlim(0, 100)
        ax_who.set_ylim(0, 100)
        ax_who.set_title("Who is acting now", fontsize=10)
        for name, (x, y) in actors.items():
            on = acting == name or (acting == "user" and name == "harness")
            ax_who.add_patch(FancyBboxPatch((x - 22, y - 7), 44, 14, boxstyle="round,pad=0,rounding_size=3",
                                            fc=COLORS[name] if on else "white", ec=COLORS[name], lw=2))
            ax_who.text(x, y, {"model": "the model (LLM)", "harness": "the harness",
                               "tool": "a tool (Python)"}[name],
                        ha="center", va="center", fontsize=11, color="white" if on else COLORS[name], weight="bold")
        for (a, b) in [((50, 77), (50, 53)), ((50, 37), (50, 16))]:
            ax_who.add_patch(FancyArrowPatch(a, b, arrowstyle="<|-|>", mutation_scale=14, color=MUTED, lw=1.5))
        fig.texts.clear()
        fig.text(0.5, 0.06, EXPLAIN[acting], ha="center", fontsize=11, color=INK)

    order = [i for i in range(len(frames)) for _ in range(3)] + [len(frames) - 1] * 6
    FuncAnimation(fig, draw, frames=order).save(path, writer=PillowWriter(fps=2), dpi=80)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--question", default="what is 37+48?")
    p.add_argument("--tiny", action="store_true", help="only lesson 7's model")
    args = p.parse_args()

    if not args.tiny:
        steps = show_run(fake_model, "the fake model", args.question)
    show_run(lesson7_model(), "lesson 7's TinyGPT, in the same harness", args.question)

    if not args.tiny:
        os.makedirs(OUTPUT, exist_ok=True)
        draw_three_loops(os.path.join(OUTPUT, "00_three_loops.png"))
        draw_agent_gif(steps, os.path.join(OUTPUT, "01_agent_loop.gif"))
        print(f"\nSaved 00_three_loops.png and 01_agent_loop.gif in {OUTPUT}")


if __name__ == "__main__":
    main()
