"""
inside.py — Looks inside the trained tiny language model.

Run from the repository root, after train.py:
    python lessons/07-language-model/inside.py

Outputs (in results/):
    02_cosine.png      — cosine similarity between the vectors (embeddings) of the characters
    03_embeddings.png  — the same vectors squeezed from 64 numbers to 2, to draw them on a map
    04_attention.png   — which earlier characters each attention head looks at, in one sentence
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FOLDER)
from tiny_lm import OUTPUT, MODEL_FILE, DEVICE, load_model

SENTENCE = "the dino jumps over the cactus."
GROUPS = [
    ("vowels", "aeiou", "#e8590c"),
    ("consonants", "bcdfghjklmnpqrstvwxyz", "#1c7ed6"),
    ("digits", "0123456789", "#2f9e44"),
    ("punctuation and space", " .,;:'\"!?()-%+=/\n", "#868e96"),
]


def show(c):
    return {"\n": "⏎", " ": "␣"}.get(c, c)


def main():
    if not os.path.exists(MODEL_FILE):
        print(f"Model not found: {MODEL_FILE}\nRun first: python lessons/07-language-model/train.py")
        return
    model, vocab = load_model()

    # Each character is a vector of 64 numbers. Cosine similarity: do two vectors point the same way?
    vectors = model.token_embedding.weight.detach().cpu()
    unit = vectors / vectors.norm(dim=1, keepdim=True)
    cosine = (unit @ unit.T).numpy()

    order, colors, bounds = [], [], []
    for name, members, color in GROUPS:
        ids = [vocab.ids[c] for c in members if c in vocab.ids]
        order += ids
        colors += [color] * len(ids)
        bounds.append((name, len(order)))

    print("Average cosine similarity inside and between groups:")
    group_ids = {name: [vocab.ids[c] for c in members if c in vocab.ids] for name, members, _ in GROUPS}

    def mean_cos(a, b):
        return np.mean([cosine[i, j] for i in a for j in b if i != j])

    for name, ids in group_ids.items():
        others = [i for n, g in group_ids.items() if n != name for i in g]
        print(f"  {name:<22} with each other: {mean_cos(ids, ids):+.2f}   with the rest: {mean_cos(ids, others):+.2f}")
    print("\nNearest neighbors (highest cosine similarity):")
    for c in "a3 .e":
        i = vocab.ids[c]
        near = [j for j in np.argsort(-cosine[i]) if j != i][:4]
        print(f"  {show(c)!r}: " + ", ".join(f"{show(vocab.chars[j])!r} {cosine[i, j]:+.2f}" for j in near))

    # 1) the cosine similarity matrix, ordered by group
    fig, ax = plt.subplots(figsize=(9, 8))
    m = cosine[np.ix_(order, order)]
    np.fill_diagonal(m, np.nan)
    im = ax.imshow(m, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ticks = [show(vocab.chars[i]) for i in order]
    ax.set_xticks(range(len(order)), ticks, fontsize=7, family="monospace")
    ax.set_yticks(range(len(order)), ticks, fontsize=7, family="monospace")
    start = 0
    for name, end in bounds:
        ax.add_patch(plt.Rectangle((start - 0.5, start - 0.5), end - start, end - start, fill=False, lw=1.5))
        start = end
    fig.colorbar(im, ax=ax, shrink=0.7, label="cosine similarity")
    ax.set_title("Cosine similarity between character vectors (squares: vowels, consonants, digits, punctuation)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "02_cosine.png"), dpi=110)
    plt.close(fig)

    # 2) a 2D map of the vectors (PCA: the 2 directions where they differ the most)
    centered = vectors - vectors.mean(0)
    _, _, v = torch.linalg.svd(centered, full_matrices=False)
    xy = (centered @ v[:2].T).numpy()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for name, members, color in GROUPS:
        ids = [vocab.ids[c] for c in members if c in vocab.ids]
        ax.scatter(xy[ids, 0], xy[ids, 1], c=color, s=30, label=name)
        for i in ids:
            ax.annotate(show(vocab.chars[i]), xy[i], fontsize=10, family="monospace",
                        xytext=(3, 3), textcoords="offset points")
    ax.legend(fontsize=8)
    ax.set_title("Each character's 64-number vector, squeezed to 2 numbers to fit on a map")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "03_embeddings.png"), dpi=110)
    plt.close(fig)

    # 3) attention in one sentence: the second (last) layer, its 4 heads
    ids = torch.tensor([vocab.encode(SENTENCE)], device=DEVICE)
    with torch.no_grad():
        model(ids, keep_weights=True)
    weights = model.blocks[-1].last_weights[0].cpu().numpy()  # heads × positions × positions
    labels = [show(c) for c in SENTENCE]
    fig, axs = plt.subplots(1, len(weights), figsize=(17, 4.8))
    for h, ax in enumerate(axs):
        ax.imshow(weights[h], cmap="Purples", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)), labels, fontsize=7, family="monospace")
        ax.set_yticks(range(len(labels)), labels, fontsize=7, family="monospace")
        ax.set_title(f"head {h + 1}", fontsize=10)
        ax.set_xlabel("looks at this character")
    axs[0].set_ylabel("while predicting what comes after this one")
    fig.suptitle("Attention in the last layer: each row is one position, and the dark cells are the earlier "
                 "characters it looks at. Nothing above the diagonal: the future is hidden.", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "04_attention.png"), dpi=110)
    plt.close(fig)
    print(f"\nCharts in: {OUTPUT}")


if __name__ == "__main__":
    main()
