"""
teach_tools.py — Can lesson 7's tiny model learn to use tools?

Run from the repository root, after training lesson 7 (about 1 minute on a recent Mac):
    python deep-dives/llm-and-harness/teach_tools.py

Lesson 7's model only ever read the study's READMEs, so it has no idea what
"call calc: 37+48" means. Here we fine-tune it, the same way lesson 5 taught
the dino by imitation: the TEACHER is the scripted fake model, and its
transcripts are the examples. Three students start from the same lesson 7 weights:

    1. answer directly       q: what is 37+48?  ->  answer: 85
    2. call tools            q: what is 37+48?  ->  call calc: 37+48
                             result: 85         ->  answer: 85
       (two-digit numbers and words of the study, 3,000 steps)
    3. call tools, varied    the same format, but numbers of 1 to 3 digits and
       made-up words like "qzwkt", trained much longer (25,000 steps)

Then all four models (lesson 7 as is, and the three students) run in the SAME
harness, on questions none of them saw in training:
    arithmetic        two-digit numbers
    counting letters  real words that were kept out of training
    bigger numbers    four-digit numbers: longer than anything in training

Outputs (in charts/):
    02_teach_tools.png      — how often each model gets the right answer
    03_copy_emerges.png     — student 3: when copying the input starts to work
    samples.md              — a few transcripts of each model
"""
import os
import random
import string
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from harness import (OUTPUT, CONTEXT, calc, fake_model, run, final_answer, random_question,
                     study_words, load_model, TinyModel)
from tiny_lm import DEVICE, TinyGPT

SEED = 0
EPISODES = 6000         # teacher transcripts to learn from
STEPS = 3000            # students 1 and 2
LONG_STEPS = 25000      # student 3
CHECK_EVERY = 2500
BATCH = 32
LEARNING_RATE = 1e-3
TESTS = 150             # questions of each kind


def teacher_text(questions, with_tools):
    """The teacher's transcripts, glued into one long text, like lesson 7's study.txt."""
    parts = []
    for q, _, _, right in questions:
        if with_tools:
            parts.append(run(fake_model, q)[0])
        else:
            parts.append(f"q: {q}\nanswer: {right}\n")
    return "".join(parts)


def fine_tune(text, vocab, label, steps=STEPS, check=None):
    """Lesson 7's training loop, starting from lesson 7's weights instead of random ones."""
    torch.manual_seed(SEED)
    base, _ = load_model()
    model = TinyGPT(len(vocab)).to(DEVICE)
    model.load_state_dict(base.state_dict())
    model.train()
    data = torch.tensor(vocab.encode(text))
    opt = torch.optim.AdamW(model.parameters(), LEARNING_RATE)
    start = time.time()
    for step in range(steps + 1):
        if check and step % CHECK_EVERY == 0:
            check(model, step)
            model.train()
        if step == steps:
            break
        starts = torch.randint(len(data) - CONTEXT - 1, (BATCH,))
        x = torch.stack([data[i:i + CONTEXT] for i in starts]).to(DEVICE)
        y = torch.stack([data[i + 1:i + CONTEXT + 1] for i in starts]).to(DEVICE)
        loss = F.cross_entropy(model(x).reshape(-1, len(vocab)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % (steps // 3) == 0:
            print(f"  {label}: step {step:4d}  loss {loss.item():.3f}   [{time.time() - start:.0f} s]")
    model.eval()
    return model


def varied_question(rng):
    """Student 3's examples: numbers of 1 to 3 digits, and made-up words."""
    if rng.random() < 0.5:
        a, b = (rng.randint(0, 10 ** rng.randint(1, 3) - 1) for _ in range(2))
        expr = f"{a}{rng.choice('+-')}{b}"
        return f"what is {expr}?", "calc", expr, calc(expr)
    w = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 12)))
    return f"how many letters in {w}?", "letters", w, str(len(w))


def test_questions(rng, test_words, seen):
    """TESTS questions of each kind, none of them seen in training."""
    kinds = {"arithmetic": [], "counting letters": [], "bigger numbers": []}
    while len(kinds["arithmetic"]) < TESTS:
        a, b, op = rng.randint(10, 99), rng.randint(10, 99), rng.choice("+-")
        q = f"what is {a}{op}{b}?"
        if q not in seen:
            kinds["arithmetic"].append((q, "calc", f"{a}{op}{b}", calc(f"{a}{op}{b}")))
    for w in rng.sample(test_words, TESTS):
        kinds["counting letters"].append((f"how many letters in {w}?", "letters", w, str(len(w))))
    for _ in range(TESTS):
        a, b, op = rng.randint(1000, 9999), rng.randint(1000, 9999), rng.choice("+-")
        kinds["bigger numbers"].append((f"what is {a}{op}{b}?", "calc", f"{a}{op}{b}", calc(f"{a}{op}{b}")))
    return kinds


def evaluate(model, kinds):
    """Runs every question through the harness and checks the answer and the tool call."""
    scores, examples = {}, {}
    for kind, questions in kinds.items():
        right = called = 0
        for q, tool, arg, answer in questions:
            transcript, _ = run(model, q)
            right += final_answer(transcript) == answer
            called += f"\ncall {tool}: {arg}\n" in transcript
            examples.setdefault(kind, transcript)
        scores[kind] = (right / len(questions), called / len(questions))
    return scores, examples


def main():
    rng = random.Random(SEED)
    words = study_words()
    rng.shuffle(words)
    cut = int(len(words) * 0.8)
    train_words, test_words = words[:cut], words[cut:]

    train_questions = [random_question(rng, train_words) for _ in range(EPISODES)]
    varied_questions = [varied_question(rng) for _ in range(4 * EPISODES)]
    seen = {q for q, *_ in train_questions + varied_questions}
    kinds = test_questions(rng, test_words, seen)
    print(f"{EPISODES} teacher transcripts, {len(train_words)} training words, "
          f"{len(test_words)} words kept out. Training on {DEVICE}.")

    base, vocab = load_model()
    direct = fine_tune(teacher_text(train_questions, with_tools=False), vocab, "1. answer directly")
    tools = fine_tune(teacher_text(train_questions, with_tools=True), vocab, "2. call tools")

    # Student 3: every CHECK_EVERY steps, how often does it copy the input right?
    progress = []
    few = {kind: kinds[kind][:50] for kind in ("arithmetic", "counting letters")}

    def check(model, step):
        scores, _ = evaluate(TinyModel(model, vocab), few)
        progress.append((step, {kind: called for kind, (_, called) in scores.items()}))

    varied = fine_tune(teacher_text(varied_questions, with_tools=True), vocab, "3. call tools, varied",
                       steps=LONG_STEPS, check=check)

    models = {"lesson 7\nas is": base, "1. answer\ndirectly": direct,
              "2. call tools": tools, "3. call tools,\nvaried data": varied}
    results, samples = {}, ["# Transcripts on questions never seen in training\n"]
    for name, m in models.items():
        scores, examples = evaluate(TinyModel(m, vocab), kinds)
        results[name] = scores
        print(f"\n{name.replace(chr(10), ' ')}")
        for kind, (right, called) in scores.items():
            print(f"  {kind:17s} right answer {right:5.0%}   called the right tool with the right input {called:5.0%}")
        samples.append(f"## {name.replace(chr(10), ' ')}\n")
        for kind, t in examples.items():
            samples.append(f"{kind}:\n\n```\n{t}```\n")

    os.makedirs(OUTPUT, exist_ok=True)
    with open(os.path.join(OUTPUT, "samples.md"), "w") as f:
        f.write("\n".join(samples))

    # Chart 1: right answers, by model and kind of question
    colors = {"arithmetic": "#2a78d6", "counting letters": "#eb6834", "bigger numbers": "#1baf7a"}
    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.26
    for k, kind in enumerate(kinds):
        for i, name in enumerate(models):
            value = results[name][kind][0] * 100
            x = i + (k - 1) * (width + 0.02)
            ax.bar(x, max(value, 0.4), width, color=colors[kind], label=kind if i == 0 else None)
            ax.text(x, value + 2, f"{value:.0f}%", ha="center", fontsize=9, color="#52514e")
    ax.set_xticks(range(len(models)), list(models))
    ax.set_ylim(0, 112)
    ax.set_ylabel("right answers (%)")
    ax.set_title("The same harness, four models, questions never seen in training", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e2", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "02_teach_tools.png"), dpi=110)

    # Chart 2: student 3 learning to copy
    fig, ax = plt.subplots(figsize=(9, 4.2))
    steps = [s for s, _ in progress]
    for kind in few:
        ys = [p[kind] * 100 for _, p in progress]
        ax.plot(steps, ys, "o-", lw=2, ms=6, c=colors[kind], label=kind)
        ax.annotate(kind, (steps[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color="#52514e")
    ax.set_xlim(0, LONG_STEPS * 1.2)
    ax.set_ylim(-3, 105)
    ax.set_xlabel("training steps")
    ax.set_ylabel("copied the input exactly (%)")
    ax.set_title("Student 3: copying the numbers comes fast, copying new words comes late", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e2", lw=0.8)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "03_copy_emerges.png"), dpi=110)
    print(f"\nSaved the charts and samples.md in {OUTPUT}")


if __name__ == "__main__":
    main()
