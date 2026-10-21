"""
concepts.py — The charts of the deep dive: weight, bias, activation, loss and gradient,
plus an animation of turning the weight and bias "knobs" (charts/knobs.gif).

Run from the repository root (a few seconds):
    python lessons/01-neuron-and-gradient/concepts/concepts.py

Every chart uses the same tiny neuron, with ONE input, so everything fits on a
flat picture:
    input  x = distance to the cactus (0 = touching it, 1 = far away)
    output   = how much the neuron "wants to jump"

It also prints the worked examples used in the README, so the numbers there
can be checked.
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOLDER = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(FOLDER, "charts")
os.makedirs(OUTPUT, exist_ok=True)

x = np.linspace(0, 1, 300)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, name), dpi=110)
    plt.close(fig)


def style(ax, title, xlabel="distance to the cactus (x)", ylabel=None):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.axhline(0, c="k", lw=0.6)
    ax.grid(alpha=0.3)


# ---------------------------------------------------------------------------
# 1) Weight: how much (and in which direction) the input matters
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
for w, color in [(2, "#2f9e44"), (1, "#69db7c"), (0, "#868e96"), (-1, "#ffa94d"), (-2, "#e8590c")]:
    ax.plot(x, w * x, c=color, lw=2, label=f"weight = {w:+d}")
style(ax, "sum = weight × x  —  the weight is the slope", ylabel="the neuron's sum")
ax.legend(fontsize=8)
save(fig, "c1_weight.png")

print("Weight, worked example (two inputs):")
for dist, speed in [(0.2, 0.5), (0.8, 0.5)]:
    s = -3 * dist + 1 * speed
    print(f"  distance {dist}, speed {speed}:  -3 × {dist} + 1 × {speed} = {s:+.1f}")

# ---------------------------------------------------------------------------
# 2) Bias: where the neuron "turns on"
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
w = -10
for b, color in [(1, "#a5d8ff"), (3, "#4dabf7"), (5, "#1c7ed6"), (7, "#0b4f9c")]:
    ax.plot(x, sigmoid(w * x + b), c=color, lw=2, label=f"bias = {b}  (turns on below x = {-b / w:.1f})")
    ax.axvline(-b / w, c=color, ls=":", lw=1)
ax.axhline(0.5, c="k", ls="--", lw=0.8)
style(ax, "sigmoid(-10 × x + bias)  —  the bias moves the switch point", ylabel="chance of jumping")
ax.legend(fontsize=8)
save(fig, "c2_bias.png")

# ---------------------------------------------------------------------------
# 3) Activation: the curve applied to the sum
# ---------------------------------------------------------------------------
z = np.linspace(-5, 5, 300)
fig, axs = plt.subplots(1, 4, figsize=(14, 3.4))
for ax, (name, f) in zip(axs, [("none (linear)", lambda v: v), ("sigmoid", sigmoid),
                               ("tanh", np.tanh), ("ReLU", lambda v: np.maximum(0, v))]):
    ax.plot(z, f(z), c="#7048e8", lw=2)
    style(ax, name, xlabel="the neuron's sum (z)")
    ax.axvline(0, c="k", lw=0.6)
axs[0].set_ylabel("output")
fig.suptitle("Four common activations. Sigmoid squeezes into 0..1 (a chance), tanh into -1..1, ReLU cuts off negatives.",
             fontsize=10)
save(fig, "c3_activation.png")

# Why stacking linear neurons is useless
w1, b1, w2, b2 = 2.0, 1.0, -3.0, 0.5
print("\nActivation, worked example (two linear neurons in a row):")
print(f"  first:  2x + 1;  second: -3 × (2x + 1) + 0.5 = {w2 * w1:+.0f}x {w2 * b1 + b2:+.1f}  -> still a straight line")

# ---------------------------------------------------------------------------
# 4) How two neurons make a band (the trick inside lesson 1)
# ---------------------------------------------------------------------------
n1 = np.tanh(25 * (x - 0.2))     # "the cactus is at least 0.2 away"
n2 = np.tanh(25 * (x - 0.5))     # "the cactus is at least 0.5 away"
band = sigmoid(4 * n1 - 4 * n2 - 3)
fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
axs[0].plot(x, n1, c="#1c7ed6", lw=2, label="neuron A: tanh(25 × (x - 0.2))")
axs[0].plot(x, n2, c="#e8590c", lw=2, label="neuron B: tanh(25 × (x - 0.5))")
style(axs[0], "Two hidden neurons, each one a 'switch'", ylabel="output")
axs[0].legend(fontsize=7, loc="lower right")
axs[1].plot(x, 4 * n1 - 4 * n2 - 3, c="#2f9e44", lw=2)
style(axs[1], "Output neuron's sum: 4 × A - 4 × B - 3")
axs[2].plot(x, band, c="#e03131", lw=2)
axs[2].axvspan(0.2, 0.5, color="#ffc9c9", alpha=0.5, label="jump zone")
axs[2].set_ylim(-0.05, 1.05)
style(axs[2], "After the sigmoid: a band", ylabel="chance of jumping")
axs[2].legend(fontsize=8)
save(fig, "c4_band.png")

print("\nThe band, worked example (output = sigmoid(4A - 4B - 3)):")
for d in [0.1, 0.35, 0.8]:
    a, bb = np.tanh(25 * (d - 0.2)), np.tanh(25 * (d - 0.5))
    print(f"  x = {d}:  A = {a:+.2f}, B = {bb:+.2f},  sum = {4 * a - 4 * bb - 3:+.2f},  chance = {sigmoid(4 * a - 4 * bb - 3):.2f}")

# ---------------------------------------------------------------------------
# 5) Loss: how much each answer costs
# ---------------------------------------------------------------------------
p = np.linspace(0.001, 0.999, 400)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
axs[0].plot(p, -np.log(p), c="#e03131", lw=2, label="right answer: JUMP  →  loss = -log(p)")
axs[0].plot(p, -np.log(1 - p), c="#1c7ed6", lw=2, label="right answer: DON'T jump  →  loss = -log(1 - p)")
axs[0].set_ylim(0, 5)
style(axs[0], "Cross-entropy: the cost of one answer", xlabel="chance of jumping the network gave (p)",
      ylabel="loss")
axs[0].legend(fontsize=8)
axs[1].plot(p, (1 - p) ** 2, c="#e03131", lw=2, ls="--", label="squared error, right answer JUMP")
axs[1].plot(p, -np.log(p), c="#e03131", lw=2, label="cross-entropy, right answer JUMP")
axs[1].set_ylim(0, 5)
style(axs[1], "Being confidently wrong: squared error barely notices, cross-entropy explodes",
      xlabel="chance of jumping the network gave (p)")
axs[1].legend(fontsize=8)
save(fig, "c5_loss.png")

print("\nLoss, worked example (three examples):")
examples = [("jump", 0.9), ("don't jump", 0.2), ("jump", 0.05)]
costs = []
for answer, chance in examples:
    c = -np.log(chance) if answer == "jump" else -np.log(1 - chance)
    costs.append(c)
    print(f"  right answer {answer:<10}, network said {chance:.0%} jump  ->  loss {c:.2f}")
print(f"  average (the loss of the whole batch): {np.mean(costs):.2f}")

# ---------------------------------------------------------------------------
# 6) Gradient: the slope of the loss, one weight at a time
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
dist = rng.uniform(0, 1, 60)
jump = (dist < 0.35).astype(float)
flip = rng.choice(60, 6, replace=False)
jump[flip] = 1 - jump[flip]  # a little noise: real data is never perfect


def loss_for_weight(w, b=3.0):
    pr = sigmoid(w * dist + b)
    return -np.mean(jump * np.log(pr + 1e-9) + (1 - jump) * np.log(1 - pr + 1e-9))


ws = np.linspace(-20, 4, 300)
L = np.array([loss_for_weight(w) for w in ws])
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(ws, L, c="#1c7ed6", lw=2)
w = 2.0
rate = 12
print("\nGradient, worked example (one weight, learning rate 12):")
for step in range(10):
    eps = 1e-4
    g = (loss_for_weight(w + eps) - loss_for_weight(w - eps)) / (2 * eps)  # the slope, measured
    lw_ = loss_for_weight(w)
    if step in (0, 1, 2, 9):
        t = np.linspace(w - 2.5, w + 2.5, 2)
        ax.plot(t, lw_ + g * (t - w), c="#e8590c", lw=1)
        offset = (-150, -28) if step == 9 else (8, 10)  # the last one goes below, out of the way
        ax.annotate(f"step {step}: slope {g:+.3f}", (w, lw_), xytext=offset, textcoords="offset points",
                    fontsize=8, color="#e8590c")
        print(f"  step {step}: weight {w:+.2f}, loss {lw_:.3f}, slope {g:+.4f} -> new weight {w - rate * g:+.2f}")
    ax.scatter([w], [lw_], c="#e8590c", s=18, zorder=5)
    w = w - rate * g
style(ax, "Loss as a function of ONE weight. The orange lines are the slope (the gradient) at each step.",
      xlabel="weight of the distance (w)", ylabel="loss")
save(fig, "c6_gradient.png")
# ---------------------------------------------------------------------------
# 7) Turning the knobs: an animation of the weight and the bias
# ---------------------------------------------------------------------------
from matplotlib.animation import FuncAnimation, PillowWriter

frames = ([("w", v, 5.0) for v in np.linspace(2, -20, 34)] + [("w", -20.0, 5.0)] * 6
          + [("b", -10.0, v) for v in np.linspace(1, 9, 34)] + [("b", -10.0, 9.0)] * 6)
fig, (ax_knobs, ax_curve) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 2.4]})


def draw(k):
    which, w, b = frames[k]
    ax_knobs.clear()
    ax_knobs.set_xlim(-1, 1)
    ax_knobs.set_ylim(0, 3)
    ax_knobs.axis("off")
    for row, (name, value, lo, hi, active) in enumerate([("weight", w, -20, 2, which == "w"),
                                                         ("bias", b, 0, 10, which == "b")]):
        y = 2.2 - row * 1.3
        ax_knobs.text(-1, y + 0.35, f"{name} = {value:+.1f}", fontsize=12,
                      color="#e8590c" if active else "#495057", weight="bold" if active else "normal")
        ax_knobs.plot([-1, 1], [y, y], c="#ced4da", lw=6, solid_capstyle="round")
        pos = -1 + 2 * (value - lo) / (hi - lo)
        ax_knobs.scatter([pos], [y], s=160, c="#e8590c" if active else "#868e96", zorder=3)
    ax_curve.clear()
    ax_curve.plot(x, sigmoid(w * x + b), c="#1c7ed6", lw=2.5)
    ax_curve.axhline(0.5, c="k", ls="--", lw=0.8)
    if w != 0 and 0 <= -b / w <= 1:
        ax_curve.axvline(-b / w, c="#e8590c", ls=":", lw=1.5)
        ax_curve.text(-b / w, 1.02, " switch point", fontsize=8, color="#e8590c")
    ax_curve.set_ylim(-0.05, 1.1)
    ax_curve.set_xlabel("distance to the cactus (x)")
    ax_curve.set_ylabel("chance of jumping")
    ax_curve.set_title("sigmoid(weight × x + bias)", fontsize=10)
    ax_curve.grid(alpha=0.3)
    hint = ("the WEIGHT changes how steep and which way the curve goes" if which == "w"
            else "the BIAS slides the switch point left and right")
    fig.suptitle(hint, fontsize=11)
    fig.tight_layout()


FuncAnimation(fig, draw, frames=len(frames)).save(os.path.join(OUTPUT, "knobs.gif"),
                                                  writer=PillowWriter(fps=8), dpi=70)
plt.close(fig)
print(f"\nCharts saved in: {OUTPUT}")
