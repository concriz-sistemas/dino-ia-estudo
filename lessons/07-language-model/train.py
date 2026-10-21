"""
Lesson 7 — Trains a tiny language model to predict the next character.

Run from the repository root (about 30 seconds on a recent Mac):
    python lessons/07-language-model/train.py
    # on any other text file (a book, your own notes...)
    python lessons/07-language-model/train.py --text my_book.txt
    # rebuild data/study.txt from the current READMEs
    python lessons/07-language-model/train.py --rebuild-text

What it does:
    1. splits the text: the first 90% to train on, the last 10% to check
       how the model does on text it has never seen;
    2. Model 1, the bigram: just counts which character follows which;
    3. Model 2, the tiny transformer: learns by gradient descent, like lesson 1;
    4. saves the model at the moment it was best on the unseen text.

Outputs (in results/):
    tiny_gpt.pt   — the trained model (weights + the list of characters)
    01_loss.png   — the loss on training text vs unseen text, step by step
    samples.md    — text written by the model at different moments of training
"""
import argparse
import copy
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FOLDER)
from tiny_lm import (OUTPUT, MODEL_FILE, DEVICE, CONTEXT, Vocab, Bigram, TinyGPT,
                     load_text, build_study_text, save_model)

SEED = 0
STEPS = 4000
BATCH = 32              # pieces of text per gradient step
LEARNING_RATE = 3e-3
EVAL_EVERY = 100
SAMPLE_AT = [0, 250, 500, 1000, 2000, 4000]
PROMPT = "the dino "


def random_pieces(data):
    """BATCH random pieces of CONTEXT characters, and the same pieces shifted by one: the answers."""
    starts = torch.randint(len(data) - CONTEXT - 1, (BATCH,))
    x = torch.stack([data[i:i + CONTEXT] for i in starts])
    y = torch.stack([data[i + 1:i + CONTEXT + 1] for i in starts])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def average_loss(model, data, n=40):
    model.eval()
    total = 0.0
    for _ in range(n):
        x, y = random_pieces(data)
        total += F.cross_entropy(model(x).reshape(-1, model.out.out_features), y.reshape(-1)).item()
    model.train()
    return total / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", help="train on this text file instead of the study's READMEs")
    p.add_argument("--rebuild-text", action="store_true", help="rebuild data/study.txt from the READMEs")
    p.add_argument("--steps", type=int, default=STEPS)
    args = p.parse_args()
    torch.manual_seed(SEED)
    os.makedirs(OUTPUT, exist_ok=True)

    text = build_study_text() if args.rebuild_text else load_text(args.text)
    vocab = Vocab(text)
    data = torch.tensor(vocab.encode(text))
    cut = int(len(data) * 0.9)
    train_data, unseen_data = data[:cut], data[cut:]
    print(f"{len(text):,} characters, {len(vocab)} different ones. "
          f"Training on {cut:,}, keeping {len(data) - cut:,} unseen.")
    print(f"A model that guesses at random has a loss of {torch.log(torch.tensor(len(vocab))):.2f}.")

    bigram = Bigram(train_data, len(vocab))
    bigram_loss = bigram.loss(unseen_data)
    print(f"\nModel 1, counting pairs (bigram): loss {bigram_loss:.2f} on unseen text")
    print("  writes: " + repr(vocab.decode(bigram.generate(vocab.encode(PROMPT), 120))))

    model = TinyGPT(len(vocab)).to(DEVICE)
    n_weights = sum(p.numel() for p in model.parameters())
    print(f"\nModel 2, a tiny transformer ({n_weights:,} weights), training on {DEVICE}...")
    opt = torch.optim.AdamW(model.parameters(), LEARNING_RATE)
    history, samples = [], []
    best = (float("inf"), None, 0)
    start = time.time()
    for step in range(args.steps + 1):
        if step % EVAL_EVERY == 0:
            tr, un = average_loss(model, train_data), average_loss(model, unseen_data)
            history.append((step, tr, un))
            if un < best[0]:
                best = (un, copy.deepcopy(model.state_dict()), step)
        if step in SAMPLE_AT:
            sample = vocab.decode(model.generate(vocab.encode(PROMPT), 160, temperature=0.8))
            samples.append((step, history[-1][1], history[-1][2], sample))
            print(f"  step {step:5d}  loss: training {history[-1][1]:.2f}, unseen {history[-1][2]:.2f}"
                  f"   [{time.time() - start:.0f} s]")
            print("     writes: " + repr(sample[:110]))
        if step == args.steps:
            break
        x, y = random_pieces(train_data)
        loss = F.cross_entropy(model(x).reshape(-1, len(vocab)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.load_state_dict(best[1])
    save_model(model, vocab)
    print(f"\nBest on unseen text at step {best[2]} (loss {best[0]:.2f}). Saved to: {MODEL_FILE}")

    # 1) the loss curves
    h = torch.tensor(history)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(h[:, 0], h[:, 1], c="#2f9e44", label="training text")
    ax.plot(h[:, 0], h[:, 2], c="#e8590c", label="unseen text")
    ax.axhline(bigram_loss, c="#868e96", ls="--", lw=1, label=f"bigram (counting pairs): {bigram_loss:.2f}")
    ax.axvline(best[2], c="k", ls=":", lw=1)
    ax.text(best[2], ax.get_ylim()[1] * 0.95, f"  best on unseen text (step {best[2]})", fontsize=8, va="top")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (lower = better guesses)")
    ax.set_title("The model gets better at the text it trains on; on unseen text it stops improving")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "01_loss.png"), dpi=110)
    plt.close(fig)

    # 2) what the model writes as it trains
    with open(os.path.join(OUTPUT, "samples.md"), "w", encoding="utf-8") as f:
        f.write("# What the tiny transformer writes during training\n\n")
        f.write(f"Each sample starts with `{PROMPT.strip()}` and uses temperature 0.8.\n\n")
        f.write(f"Bigram (counting pairs), for comparison:\n\n```\n"
                f"{vocab.decode(bigram.generate(vocab.encode(PROMPT), 160))}\n```\n\n")
        for step, tr, un, sample in samples:
            f.write(f"## Step {step} — loss {tr:.2f} on training text, {un:.2f} on unseen text\n\n"
                    f"```\n{sample}\n```\n\n")
    print(f"Charts and samples in: {OUTPUT}")
    print("Make it write:")
    print("    python lessons/07-language-model/generate.py")


if __name__ == "__main__":
    main()
