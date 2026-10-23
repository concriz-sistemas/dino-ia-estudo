"""
harness.py — The pieces shared by the scripts of this deep dive: the tools,
the models and the harness (the agent loop).

The whole conversation is ONE piece of text, the transcript. Each line starts
with who wrote it:

    q: what is 37+48?          <- the user
    call calc: 37+48           <- the model asks for a tool
    result: 85                 <- the harness ran the tool and pasted the result
    answer: 85                 <- the model answers; the loop stops

Everything is lowercase and uses only the 53 characters lesson 7's model knows,
so the same harness can drive the scripted fake model and lesson 7's TinyGPT.
"""
import os
import random
import re
import sys

import torch

FOLDER = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FOLDER, "..", ".."))
OUTPUT = os.path.join(FOLDER, "charts")
LESSON_7 = os.path.join(ROOT, "lessons", "07-language-model")
sys.path.insert(0, LESSON_7)
from tiny_lm import CONTEXT, load_model, load_text  # noqa: E402

MAX_TURNS = 4


# ---------------------------------------------------------------------------
# The tools: ordinary Python functions. The harness runs them, never the model.
# ---------------------------------------------------------------------------
def calc(expression):
    """Adds or subtracts two whole numbers: '37+48' -> '85'."""
    m = re.fullmatch(r"\s*(-?\d+)\s*([+-])\s*(\d+)\s*", expression)
    if not m:
        return "error: write a+b or a-b"
    a, op, b = int(m[1]), m[2], int(m[3])
    return str(a + b if op == "+" else a - b)


def letters(word):
    """Counts the letters of a word: 'gradient' -> '8'."""
    word = word.strip()
    if not word.isalpha():
        return "error: write one word"
    return str(len(word))


TOOLS = {"calc": calc, "letters": letters}


# ---------------------------------------------------------------------------
# The questions
# ---------------------------------------------------------------------------
def study_words():
    """The different words of lesson 7's text, 3 to 12 letters long."""
    return sorted({w for w in re.findall(r"[a-z]+", load_text()) if 3 <= len(w) <= 12})


def random_question(rng, words):
    """Half arithmetic, half counting letters. Returns (question, tool, input, right answer)."""
    if rng.random() < 0.5:
        a, b, op = rng.randint(10, 99), rng.randint(10, 99), rng.choice("+-")
        expr = f"{a}{op}{b}"
        return f"what is {expr}?", "calc", expr, calc(expr)
    w = rng.choice(words)
    return f"how many letters in {w}?", "letters", w, letters(w)


# ---------------------------------------------------------------------------
# The models: text in, one line of text out
# ---------------------------------------------------------------------------
def fake_model(transcript):
    """A scripted model. Like a real LLM, it sees nothing but the transcript."""
    lines = transcript.strip().split("\n")
    if lines[-1].startswith("result: "):
        return "answer: " + lines[-1][len("result: "):]
    q = lines[-1][len("q: "):]
    m = re.fullmatch(r"what is (.+)\?", q)
    if m:
        return f"call calc: {m[1]}"
    m = re.fullmatch(r"how many letters in (\w+)\?", q)
    if m:
        return f"call letters: {m[1]}"
    return "answer: i don't know"


class TinyModel:
    """A TinyGPT used as the model: it writes one character at a time until the end of the line."""

    def __init__(self, model, vocab, temperature=0.0, seed=0):
        self.model, self.vocab, self.temperature = model, vocab, temperature
        self.g = torch.Generator().manual_seed(seed)

    def __call__(self, transcript, max_chars=40):
        ids = self.vocab.encode(transcript)
        line = ""
        for _ in range(max_chars):
            probs = self.model.next_probabilities(ids[-CONTEXT:], self.temperature or 1.0)
            if self.temperature == 0:
                pick = int(probs.argmax())                      # always the most likely character
            else:
                pick = int(torch.multinomial(probs, 1, generator=self.g))
            c = self.vocab.chars[pick]
            if c == "\n":
                break
            line += c
            ids.append(pick)
        return line


def lesson7_model(**kw):
    model, vocab = load_model()
    return TinyModel(model, vocab, **kw)


# ---------------------------------------------------------------------------
# The harness: the agent loop
# ---------------------------------------------------------------------------
def run(model, question, max_turns=MAX_TURNS, verbose=False):
    """
    Asks the model, runs the tool it asks for, pastes the result, and asks again,
    until the model answers or runs out of turns.
    Returns the transcript and the list of steps [(who, text), ...].
    """
    transcript = f"q: {question}\n"
    steps = [("user", f"q: {question}")]
    for _ in range(max_turns):
        reply = model(transcript)
        transcript += reply + "\n"
        steps.append(("model", reply))
        if verbose:
            print(f"  model wrote:   {reply!r}")

        call = re.fullmatch(r"call (\w+): (.+)", reply)
        if call:
            name, arg = call[1], call[2]
            result = TOOLS[name](arg) if name in TOOLS else f"error: no tool called {name}"
            steps.append(("tool", f"{name}({arg!r}) -> {result}"))
        elif reply.startswith("answer: "):
            break
        else:
            result = "error: write call or answer"
        transcript += f"result: {result}\n"
        steps.append(("harness", f"result: {result}"))
        if verbose:
            print(f"  harness wrote: 'result: {result}'")
    return transcript, steps


def final_answer(transcript):
    m = re.search(r"^answer: (.*)$", transcript, re.M)
    return m[1].strip() if m else None
