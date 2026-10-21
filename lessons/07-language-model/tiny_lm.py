"""
tiny_lm.py — The pieces shared by the scripts of lesson 7: a tiny language model.

- The text: the English READMEs of lessons 1-6 and the study home, cleaned up
  (no tables, no code), in lowercase, frozen in data/study.txt. About 47,000
  characters. You can train on any other text file.
- Tokens: here every CHARACTER is a token (53 of them). Real LLMs use pieces of
  words instead, with vocabularies of 50,000 to 200,000 tokens.
- Two models that do the same job — given the text so far, give the chance of
  each possible next character:
    Bigram   — counts which character follows which. No training, no network.
    TinyGPT  — a transformer, the same design as an LLM, just tiny: 2 layers,
               4 attention heads, vectors of 64 numbers. About 110,000 weights.
"""
import glob
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

FOLDER = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FOLDER, "..", ".."))
OUTPUT = os.path.join(FOLDER, "results")
MODEL_FILE = os.path.join(OUTPUT, "tiny_gpt.pt")
# The text is frozen in a file, so the example results don't change every time a README does
TEXT_FILE = os.path.join(FOLDER, "data", "study.txt")

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
ALLOWED = set("abcdefghijklmnopqrstuvwxyz0123456789 .,;:'\"!?()-%+=/\n")

# The size of the model (the "hyperparameters" of a transformer)
CONTEXT = 64      # how many previous characters the model can look at
EMBED = 64        # how many numbers describe each token (the size of its vector)
HEADS = 4         # attention heads per layer
LAYERS = 2        # transformer blocks stacked
DROPOUT = 0.1     # randomly switches off 10% of the signals in training: harder to memorize


# ---------------------------------------------------------------------------
# The text
# ---------------------------------------------------------------------------
def clean_markdown(md):
    """Keeps only the prose of a README: no code blocks, tables, images, headings or links."""
    md = re.sub(r"```.*?```", "", md, flags=re.S)
    lines = []
    for line in md.split("\n"):
        s = line.strip()
        if not s or s.startswith(("|", "!", "#", ">", "**English", "[English")) or set(s) <= set("-"):
            continue
        s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)    # [text](link) -> text
        s = re.sub(r"[`*_]", "", s)                        # markdown emphasis
        s = re.sub(r"^\d+\.\s+|^-\s+", "", s)              # list markers
        lines.append(s)
    text = "\n".join(lines).lower()
    for a, b in [("—", "-"), ("–", "-"), ("−", "-"), ("…", "..."), ("×", "x")]:
        text = text.replace(a, b)
    return "".join(c for c in text if c in ALLOWED)


def build_study_text():
    """Cleans the English READMEs of the study and saves the result to data/study.txt."""
    files = [os.path.join(ROOT, "README.md")] + sorted(glob.glob(os.path.join(ROOT, "lessons", "*", "README.md")))
    text = "\n".join(clean_markdown(open(f, encoding="utf-8").read()) for f in files)
    os.makedirs(os.path.dirname(TEXT_FILE), exist_ok=True)
    with open(TEXT_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def load_text(path=None):
    """The training text: any text file you choose, or the frozen study text (built on first use)."""
    if path:
        return "".join(c for c in open(path, encoding="utf-8").read().lower() if c in ALLOWED)
    if not os.path.exists(TEXT_FILE):
        return build_study_text()
    return open(TEXT_FILE, encoding="utf-8").read()


class Vocab:
    """Turns characters into numbers (token ids) and back."""

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.ids = {c: i for i, c in enumerate(self.chars)}

    def __len__(self):
        return len(self.chars)

    def encode(self, text):
        return [self.ids[c] for c in text if c in self.ids]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)


# ---------------------------------------------------------------------------
# Model 1: counting pairs
# ---------------------------------------------------------------------------
class Bigram:
    def __init__(self, data, n_tokens):
        counts = torch.ones(n_tokens, n_tokens)  # start at 1: nothing is ever impossible
        for a, b in zip(data[:-1].tolist(), data[1:].tolist()):
            counts[a, b] += 1
        self.probs = counts / counts.sum(1, keepdim=True)

    def loss(self, data):
        return float(-torch.log(self.probs[data[:-1], data[1:]]).mean())

    def generate(self, start_ids, n, seed=0):
        g = torch.Generator().manual_seed(seed)
        ids = list(start_ids)
        for _ in range(n):
            ids.append(int(torch.multinomial(self.probs[ids[-1]], 1, generator=g)))
        return ids


# ---------------------------------------------------------------------------
# Model 2: a tiny transformer
# ---------------------------------------------------------------------------
class Block(nn.Module):
    """
    One transformer block:
      1. ATTENTION — each position looks back at the previous characters and
         decides which ones matter for what comes next;
      2. a small network (MLP) that processes what attention gathered.
    """

    def __init__(self, dropout=DROPOUT):
        super().__init__()
        self.norm1 = nn.LayerNorm(EMBED)
        self.attention = nn.MultiheadAttention(EMBED, HEADS, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(EMBED)
        self.mlp = nn.Sequential(nn.Linear(EMBED, 4 * EMBED), nn.GELU(), nn.Linear(4 * EMBED, EMBED))
        self.drop = nn.Dropout(dropout)
        self.last_weights = None  # filled when asked to, so we can draw what attention looks at

    def forward(self, x, mask, keep_weights=False):
        h = self.norm1(x)
        a, w = self.attention(h, h, h, attn_mask=mask, need_weights=keep_weights, average_attn_weights=False)
        if keep_weights:
            self.last_weights = w
        x = x + self.drop(a)
        return x + self.drop(self.mlp(self.norm2(x)))


class TinyGPT(nn.Module):
    def __init__(self, n_tokens, dropout=DROPOUT):
        super().__init__()
        self.token_embedding = nn.Embedding(n_tokens, EMBED)     # one vector of 64 numbers per character
        self.position_embedding = nn.Embedding(CONTEXT, EMBED)   # one vector per position (1st, 2nd, ...)
        self.blocks = nn.ModuleList([Block(dropout) for _ in range(LAYERS)])
        self.norm = nn.LayerNorm(EMBED)
        self.out = nn.Linear(EMBED, n_tokens)                    # a score for each possible next character
        # The mask stops each position from looking at the FUTURE: no peeking at the answer
        self.register_buffer("mask", torch.triu(torch.ones(CONTEXT, CONTEXT, dtype=torch.bool), 1))

    def forward(self, ids, keep_weights=False):
        t = ids.shape[1]
        x = self.token_embedding(ids) + self.position_embedding(torch.arange(t, device=ids.device))
        for block in self.blocks:
            x = block(x, self.mask[:t, :t], keep_weights)
        return self.out(self.norm(x))

    @torch.no_grad()
    def next_probabilities(self, ids, temperature=1.0):
        """The chance of each possible next character, given the text so far."""
        self.eval()
        context = torch.tensor([ids[-CONTEXT:]], device=self.mask.device)
        scores = self(context)[0, -1] / temperature
        return F.softmax(scores, dim=-1).cpu()

    def generate(self, start_ids, n, temperature=1.0, seed=0):
        g = torch.Generator().manual_seed(seed)
        ids = list(start_ids)
        for _ in range(n):
            ids.append(int(torch.multinomial(self.next_probabilities(ids, temperature), 1, generator=g)))
        return ids


def save_model(model, vocab, path=MODEL_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"weights": model.state_dict(), "chars": vocab.chars}, path)


def load_model(path=MODEL_FILE):
    saved = torch.load(path, map_location=DEVICE)
    vocab = Vocab("".join(saved["chars"]))
    model = TinyGPT(len(vocab)).to(DEVICE)
    model.load_state_dict(saved["weights"])
    model.eval()
    return model, vocab
