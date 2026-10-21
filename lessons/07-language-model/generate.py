"""
generate.py — The tiny language model writes, one character at a time.

Run from the repository root, after train.py:
    python lessons/07-language-model/generate.py
    # your own start, and how "adventurous" the choices are
    python lessons/07-language-model/generate.py --prompt "the network " --temperature 0.5
    # the same start at three temperatures
    python lessons/07-language-model/generate.py --compare
    # shows the 5 most likely next characters before each choice
    python lessons/07-language-model/generate.py --explain
    # records results/generation.gif
    python lessons/07-language-model/generate.py --gif

How it writes: the model gives a chance to every possible next character, ONE is
drawn at random according to those chances, it's added to the text, and the
model is asked again. That's all an LLM does too, one token at a time.

Temperature divides the scores before they become chances:
    low  (0.3) — the likely characters get even more likely: safe, repetitive
    1.0        — the chances the model learned
    high (1.5) — the unlikely ones get a better shot: varied, then nonsense
"""
import argparse
import os
import sys

FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FOLDER)
from tiny_lm import OUTPUT, MODEL_FILE, load_model


def show(c):
    return {"\n": "⏎", " ": "␣"}.get(c, c)


def explain(model, vocab, prompt, n, temperature):
    import torch

    ids = vocab.encode(prompt)
    print(f"{'text so far':>34}   the 5 most likely next characters         drawn")
    for _ in range(n):
        probs = model.next_probabilities(ids, temperature)
        top = probs.topk(5)
        choices = "  ".join(f"{show(vocab.chars[i])} {p:4.0%}" for p, i in zip(top.values.tolist(), top.indices.tolist()))
        pick = int(torch.multinomial(probs, 1))
        print(f"{vocab.decode(ids)[-30:]!r:>34}   {choices}   -> {show(vocab.chars[pick])!r}")
        ids.append(pick)


def record_gif(model, vocab, prompt, temperature, path, n=70):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from matplotlib.animation import FuncAnimation, PillowWriter

    torch.manual_seed(1)
    ids = vocab.encode(prompt)
    steps = []
    for _ in range(n):
        probs = model.next_probabilities(ids, temperature)
        pick = int(torch.multinomial(probs, 1))
        top = probs.topk(8)
        steps.append((vocab.decode(ids), [(show(vocab.chars[i]), p) for p, i in
                                          zip(top.values.tolist(), top.indices.tolist())], show(vocab.chars[pick])))
        ids.append(pick)

    fig, (ax_text, ax_bars) = plt.subplots(2, 1, figsize=(9, 5), gridspec_kw={"height_ratios": [1, 1.6]})

    def draw(k):
        text, top, pick = steps[k]
        ax_text.clear()
        ax_text.axis("off")
        shown = text[-60:].replace("\n", " ⏎ ")
        ax_text.text(0.0, 0.5, shown + "▌", fontsize=13, family="monospace", va="center", wrap=True)
        ax_text.set_title("The text so far (the model writes one character at a time)", fontsize=10)
        ax_bars.clear()
        labels = [c for c, _ in top]
        colors = ["#e8590c" if c == pick else "#adb5bd" for c in labels]
        ax_bars.bar(range(len(top)), [p * 100 for _, p in top], color=colors)
        ax_bars.set_xticks(range(len(top)), [f"'{c}'" for c in labels], fontsize=12, family="monospace")
        ax_bars.set_ylim(0, 100)
        ax_bars.set_ylabel("chance (%)")
        ax_bars.set_title(f"The 8 most likely next characters (temperature {temperature}). "
                          f"Orange = the one drawn: '{pick}'", fontsize=10)
        fig.tight_layout()

    frames = list(range(len(steps))) + [len(steps) - 1] * 8
    FuncAnimation(fig, draw, frames=frames).save(path, writer=PillowWriter(fps=4), dpi=70)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="the dino ")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--length", type=int, default=300)
    p.add_argument("--compare", action="store_true", help="the same start at temperatures 0.3, 0.8 and 1.5")
    p.add_argument("--explain", action="store_true", help="show the 5 most likely next characters at each step")
    p.add_argument("--gif", action="store_true", help="record results/generation.gif")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not os.path.exists(MODEL_FILE):
        print(f"Model not found: {MODEL_FILE}\nRun first: python lessons/07-language-model/train.py")
        return
    model, vocab = load_model()
    prompt = args.prompt.lower()

    if args.gif:
        path = os.path.join(OUTPUT, "generation.gif")
        record_gif(model, vocab, prompt, args.temperature, path)
        print(f"Animation saved to: {path}")
    elif args.explain:
        explain(model, vocab, prompt, 25, args.temperature)
    elif args.compare:
        for t in [0.3, 0.8, 1.5]:
            text = vocab.decode(model.generate(vocab.encode(prompt), 200, t, args.seed))
            print(f"--- temperature {t} ---\n{text}\n")
    else:
        print(vocab.decode(model.generate(vocab.encode(prompt), args.length, args.temperature, args.seed)))


if __name__ == "__main__":
    main()
