"""
quantization.py — What happens to a model when its weights are rounded to fewer bits?

Takes the tiny language model of lesson 7 and stores its weights with fewer
and fewer bits: 8, 6, 4, 3 and 2 bits per weight instead of 32. Then measures
how good it still is (loss on unseen text) and what it writes.

Two ways to round, both very simple:
    one ruler per table  — all the weights of a matrix share one scale
    one ruler per row    — each row of the matrix gets its own scale
                           (rows with small weights don't waste the fine steps)

Run from the repository root, after lesson 7's train.py (about 1 minute):
    python deep-dives/hardware/quantization.py

Outputs (in charts/):
    01_rounding.png      — real weights of the model, and the few values 2-bit and 4-bit can store
    02_quality.png       — loss on unseen text for each number of bits, both ways of rounding
    03_colors.png        — the same idea with colors: a chart drawn with fewer and fewer colors
    quantization.json
"""
import copy
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

FOLDER = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FOLDER, "..", ".."))
OUTPUT = os.path.join(FOLDER, "charts")
os.makedirs(OUTPUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "lessons", "07-language-model"))
from tiny_lm import load_model, load_text, CONTEXT

BITS = [8, 6, 4, 3, 2]
PROMPT = "the dino "


def quantize(w, bits, per_row):
    """
    Symmetric rounding to `bits` bits: the weights are divided by a scale, rounded
    to the nearest whole number between -(2^(bits-1) - 1) and +(2^(bits-1) - 1),
    then multiplied back. Only those few whole numbers (and the scale) need to be stored.
    """
    levels = 2 ** (bits - 1) - 1
    if per_row and w.dim() == 2:
        scale = w.abs().amax(dim=1, keepdim=True) / levels
    else:
        scale = w.abs().max() / levels
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return torch.round(w / scale).clamp(-levels, levels) * scale


def quantized_copy(model, bits, per_row):
    """Rounds the big tables (embeddings and layer weights). Small things (biases, norms) stay as they are."""
    m = copy.deepcopy(model)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.dim() == 2:
                p.copy_(quantize(p, bits, per_row))
            elif "in_proj_weight" in name:
                p.copy_(quantize(p, bits, per_row))
    return m


def unseen_loss(model, data, vocab_size, seed=0, n=60, batch=32):
    g = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    total = 0.0
    model.eval()
    with torch.no_grad():
        for _ in range(n):
            starts = torch.randint(len(data) - CONTEXT - 1, (batch,), generator=g)
            x = torch.stack([data[i:i + CONTEXT] for i in starts]).to(device)
            y = torch.stack([data[i + 1:i + CONTEXT + 1] for i in starts]).to(device)
            total += F.cross_entropy(model(x).reshape(-1, vocab_size), y.reshape(-1)).item()
    return total / n


def main():
    model, vocab = load_model()
    model = model.cpu()
    text = load_text()
    data = torch.tensor(vocab.encode(text))
    unseen = data[int(len(data) * 0.9):]
    n_big = sum(p.numel() for p in model.parameters() if p.dim() == 2)
    n_all = sum(p.numel() for p in model.parameters())

    base = unseen_loss(model, unseen, len(vocab))
    sample = vocab.decode(model.generate(vocab.encode(PROMPT), 110, 0.8, seed=0))
    print(f"Original (32-bit floats): loss on unseen text {base:.3f}")
    print(f"  writes: {sample!r}\n")
    results = {"32": {"loss": base, "sample": sample, "kilobytes": n_all * 4 / 1024}}
    for per_row in [False, True]:
        way = "one ruler per row" if per_row else "one ruler per table"
        for bits in BITS:
            q = quantized_copy(model, bits, per_row)
            loss = unseen_loss(q, unseen, len(vocab))
            text_q = vocab.decode(q.generate(vocab.encode(PROMPT), 110, 0.8, seed=0))
            kb = (n_big * bits / 8 + (n_all - n_big) * 4) / 1024
            results[f"{bits}-{'row' if per_row else 'table'}"] = {"loss": loss, "sample": text_q, "kilobytes": kb}
            print(f"{bits} bits, {way:<20} loss {loss:.3f}   size {kb:6.0f} KB   writes: {text_q!r}")
    with open(os.path.join(OUTPUT, "quantization.json"), "w") as f:
        json.dump(results, f, indent=1)

    # 1) what rounding does to the weights of one table
    w = model.blocks[0].mlp[0].weight.detach().flatten()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    for ax, bits in zip(axs, [4, 2]):
        levels = 2 ** (bits - 1) - 1
        scale = w.abs().max() / levels
        ax.hist(w.numpy(), bins=120, color="#adb5bd")
        for k in range(-levels, levels + 1):
            ax.axvline(k * scale, c="#e8590c", lw=1.5)
        ax.set_title(f"{bits} bits: every weight becomes one of the {2 * levels + 1} orange values", fontsize=10)
        ax.set_xlabel("weight value")
    axs[0].set_ylabel("how many weights")
    fig.suptitle("The real weights of one table of the lesson 7 model (gray), and where rounding sends them",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "01_rounding.png"), dpi=110)
    plt.close(fig)

    # 2) quality per number of bits
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for per_row, color in [(False, "#868e96"), (True, "#1c7ed6")]:
        key = "row" if per_row else "table"
        ax.plot(BITS, [results[f"{b}-{key}"]["loss"] for b in BITS], "o-", c=color, lw=2,
                label="one ruler per row" if per_row else "one ruler per table")
    ax.axhline(base, c="#2f9e44", ls="--", label="original, 32 bits")
    ax.set_xticks(BITS)
    ax.invert_xaxis()
    ax.set_xlabel("bits per weight (fewer to the right)")
    ax.set_ylabel("loss on unseen text (lower = better)")
    ax.set_title("Rounding the weights of lesson 7's model to fewer bits")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "02_quality.png"), dpi=110)
    plt.close(fig)

    # 3) the same idea with colors
    from PIL import Image

    img = Image.open(os.path.join(ROOT, "lessons", "01-neuron-and-gradient", "results", "04_boundaries.png"))
    img = img.convert("RGB").crop((0, int(img.height * 0.1), img.width // 4, img.height))  # one panel
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, colors in zip(axs, [256, 16, 4, 2]):
        q = img.quantize(colors=colors).convert("RGB")
        ax.imshow(q)
        ax.set_title(f"{colors} colors", fontsize=10)
        ax.axis("off")
    fig.suptitle("Quantization with colors: the same picture, stored with fewer and fewer colors", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "03_colors.png"), dpi=110)
    plt.close(fig)
    print(f"\nCharts in: {OUTPUT}")


if __name__ == "__main__":
    main()
