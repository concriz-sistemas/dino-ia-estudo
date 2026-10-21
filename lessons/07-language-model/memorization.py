"""
memorization.py — Is the model writing new text, or copying what it read?

For every character the model writes, we look for the LONGEST piece of text
around it that appears, exactly, in the training text. Short pieces ("the ",
"ing") are everywhere and mean nothing. A copied piece of 20+ characters is
a sign of memorization.

As a fair reference, the same measure runs on the unseen 10% of the study's
text: real English, written by a person, never shown to the model. It shows
how much "copying by chance" new text naturally has.

Two models are trained for 12,000 steps (3× longer than train.py) and checked
every 1,000 steps:
    with dropout 0.1   — like train.py
    without dropout    — nothing stops it from memorizing

Run from the repository root (about 3 minutes on a recent Mac):
    python lessons/07-language-model/memorization.py

Outputs (in results/):
    05_memorization.png  — loss on unseen text, and how much of the written text is copied
    06_copied_text.png   — the text itself, colored by how long the copied piece around each character is
"""
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FOLDER)
from tiny_lm import OUTPUT, DEVICE, CONTEXT, Vocab, TinyGPT, load_text

SEED = 0
STEPS = 12000
CHECK_EVERY = 1000
BATCH = 32
LEARNING_RATE = 3e-3
COPY = 20             # a copied piece this long or longer counts as memorized
SAMPLE_CHARS = 2000   # characters written at each check
PROMPT = "the dino "


def match_lengths(text, reference):
    """
    For each position i: the length of the longest piece text[i:i+L] that appears in `reference`.
    (Each next answer is at least the previous one minus 1, so it's quick to compute.)
    """
    lengths, L = [], 0
    for i in range(len(text)):
        L = max(L - 1, 0)
        while i + L < len(text) and text[i:i + L + 1] in reference:
            L += 1
        lengths.append(L)
    return lengths


def copy_length_per_char(text, reference):
    """For each character: the longest copied piece (from `reference`) that covers it."""
    cover = np.zeros(len(text), dtype=int)
    for i, L in enumerate(match_lengths(text, reference)):
        cover[i:i + L] = np.maximum(cover[i:i + L], L)
    return cover


def copied_fraction(text, reference):
    return float((copy_length_per_char(text, reference) >= COPY).mean())


def train_and_watch(name, dropout, vocab, train_data, unseen_data, train_text):
    torch.manual_seed(SEED)
    model = TinyGPT(len(vocab), dropout=dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), LEARNING_RATE)

    def pieces(data):
        starts = torch.randint(len(data) - CONTEXT - 1, (BATCH,))
        return (torch.stack([data[i:i + CONTEXT] for i in starts]).to(DEVICE),
                torch.stack([data[i + 1:i + CONTEXT + 1] for i in starts]).to(DEVICE))

    history, texts = [], {}
    start = time.time()
    for step in range(STEPS + 1):
        if step % CHECK_EVERY == 0:
            model.eval()
            with torch.no_grad():
                unseen = np.mean([F.cross_entropy(model(x).reshape(-1, len(vocab)), y.reshape(-1)).item()
                                  for x, y in [pieces(unseen_data) for _ in range(40)]])
            text = vocab.decode(model.generate(vocab.encode(PROMPT), SAMPLE_CHARS, 0.8, seed=1))[len(PROMPT):]
            copied = copied_fraction(text, train_text)
            history.append((step, unseen, copied))
            texts[step] = text
            model.train()
            print(f"  {name:<16} step {step:6d}: loss on unseen text {unseen:.2f}, "
                  f"copied {copied:5.1%}   [{time.time() - start:.0f} s]")
        if step == STEPS:
            break
        x, y = pieces(train_data)
        loss = F.cross_entropy(model(x).reshape(-1, len(vocab)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return np.array(history), texts


def draw_colored(ax, text, cover, title, width=78, lines=9):
    """Draws text one character at a time, its background colored by the copied piece around it."""
    ax.axis("off")
    ax.set_xlim(0, width)
    ax.set_ylim(lines, -1)
    ax.set_title(title, fontsize=10, loc="left")
    col, row = 0, 0
    for c, n in zip(text, cover):
        if c == "\n":
            col, row = 0, row + 1
            continue
        if col >= width:
            col, row = 0, row + 1
        if row >= lines:
            break
        color = "#ffc9c9" if n >= COPY else ("#fff3bf" if n >= 10 else "#d3f9d8")
        ax.add_patch(plt.Rectangle((col, row - 0.5), 1, 1, color=color, lw=0))
        ax.text(col + 0.5, row, c, ha="center", va="center", family="monospace", fontsize=8)
        col += 1


def main():
    os.makedirs(OUTPUT, exist_ok=True)
    text = load_text()
    vocab = Vocab(text)
    data = torch.tensor(vocab.encode(text))
    cut = int(len(data) * 0.9)
    train_text, unseen_text = text[:cut], text[cut:]

    baseline = copied_fraction(unseen_text[:SAMPLE_CHARS], train_text)
    print(f"Reference: {baseline:.1%} of real, unseen English text is in copied pieces of {COPY}+ characters.\n")

    runs = {}
    for name, dropout in [("with dropout", 0.1), ("without dropout", 0.0)]:
        print(f"Training {name}...")
        runs[name] = train_and_watch(name, dropout, vocab, data[:cut], data[cut:], train_text)

    # 1) loss and copying over training
    colors = {"with dropout": "#1c7ed6", "without dropout": "#e03131"}
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, (h, _) in runs.items():
        axs[0].plot(h[:, 0], h[:, 1], "o-", c=colors[name], label=name)
        axs[1].plot(h[:, 0], h[:, 2] * 100, "o-", c=colors[name], label=name)
    axs[1].axhline(baseline * 100, c="k", ls="--", lw=1, label="real unseen English text")
    axs[0].set_title("Loss on unseen text")
    axs[1].set_title(f"% of the written text inside copied pieces of {COPY}+ characters")
    for ax in axs:
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "05_memorization.png"), dpi=110)
    plt.close(fig)

    # 2) the text, colored
    h_drop, texts_drop = runs["with dropout"]
    best_step = int(h_drop[h_drop[:, 1].argmin(), 0])
    h_raw, texts_raw = runs["without dropout"]
    worst_step = int(h_raw[h_raw[:, 2].argmax(), 0])
    panels = [
        (texts_drop[best_step], f"With dropout, step {best_step} (its best on unseen text): "
                                f"{h_drop[h_drop[:, 0] == best_step, 2][0]:.0%} copied"),
        (texts_raw[worst_step], f"Without dropout, step {worst_step}: "
                                f"{h_raw[h_raw[:, 0] == worst_step, 2][0]:.0%} copied"),
        (unseen_text[:SAMPLE_CHARS], f"Reference: real unseen English text from the study: {baseline:.0%} copied"),
    ]
    fig, axs = plt.subplots(len(panels), 1, figsize=(13, 10))
    for ax, (t, title) in zip(axs, panels):
        draw_colored(ax, t, copy_length_per_char(t, train_text), title)
    fig.suptitle(f"Green: new combination  ·  yellow: inside a copied piece of 10-{COPY - 1} characters  ·  "
                 f"red: inside a copied piece of {COPY}+ characters", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "06_copied_text.png"), dpi=110)
    plt.close(fig)
    print(f"\nCharts in: {OUTPUT}")


if __name__ == "__main__":
    main()
