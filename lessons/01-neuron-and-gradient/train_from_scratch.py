"""
Lesson 1 — A neural network from scratch, using only numpy.

Goal: see with your own eyes what "training with gradients" means.
No AI libraries here: every calculation is written by hand.

The problem (inspired by the dino):
    Inputs: distance to the cactus (0..1) and game speed (0..1)
    Output: probability that "now is the time to jump"

    The rule we want the network to DISCOVER on its own:
    it's time to jump if the cactus is within a BAND of distances —
    not too close (too late), not too far (too early).
    And the faster the game, the farther away that band is.

Run from the repository root:
    # train and save the charts
    python lessons/01-neuron-and-gradient/train_from_scratch.py
    # train, then play the training back in a window, like a video
    python lessons/01-neuron-and-gradient/train_from_scratch.py --live
    # record the same replay to results/training.gif
    python lessons/01-neuron-and-gradient/train_from_scratch.py --gif

In the replay:  SPACE = pause/resume,  ← → = back/forward one epoch,
                ↑ ↓ = faster/slower,  R = restart

Outputs (in results/):
    weight_history.csv    — every weight, the loss and the accuracy at each epoch
    01_data.png           — the training examples
    02_loss.png           — the learning curve
    03_weights.png        — how each weight changed over time
    04_boundaries.png     — what the network "thinks" at different moments of training
    05_single_neuron.png  — why we need a hidden layer
    06_hidden_lines.png   — the straight line drawn by each hidden neuron
"""
import csv
import os
import sys
import warnings

import numpy as np
import matplotlib

LIVE = "--live" in sys.argv
GIF = "--gif" in sys.argv
if not LIVE:
    matplotlib.use("Agg")  # makes PNGs without needing a window
import matplotlib.pyplot as plt

FOLDER = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(FOLDER, "results")
os.makedirs(OUTPUT, exist_ok=True)

# ---------------------------------------------------------------------------
# HYPERPARAMETERS — the "knobs" of training. Change them and run again!
# ---------------------------------------------------------------------------
SEED = 42               # so the result is reproducible
N_EXAMPLES = 600        # how many training examples
N_HIDDEN = 6            # neurons in the hidden layer
LEARNING_RATE = 0.5     # size of the step in the direction opposite to the gradient
EPOCHS = 4000           # how many times the network sees all the examples

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# 1) DATA
# ---------------------------------------------------------------------------
def true_rule(dist, speed):
    """The 'right answer'. The network never sees this function — only the examples."""
    start = 0.10 + 0.25 * speed
    end = 0.30 + 0.35 * speed
    return ((dist > start) & (dist < end)).astype(float)


def make_data(n):
    dist = rng.uniform(0, 1, n)
    speed = rng.uniform(0, 1, n)
    X = np.column_stack([dist, speed])
    y = true_rule(dist, speed).reshape(-1, 1)
    return X, y


def normalize(X):
    """Maps 0..1 to -1..1. Networks train better with inputs centered on zero."""
    return X * 2 - 1


# ---------------------------------------------------------------------------
# 2) THE NETWORK: input(2) -> hidden(N, tanh) -> output(1, sigmoid)
# ---------------------------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def init_weights(n_inputs, n_hidden):
    # Small, random weights. If they all started out equal (e.g. zero),
    # every hidden neuron would learn exactly the same thing.
    return {
        "W1": rng.normal(0, 1, (n_inputs, n_hidden)),
        "b1": np.zeros((1, n_hidden)),
        "W2": rng.normal(0, 1, (n_hidden, 1)),
        "b2": np.zeros((1, 1)),
    }


def forward(p, X):
    """Forward pass: from the inputs all the way to the prediction."""
    z1 = X @ p["W1"] + p["b1"]     # weighted sum in the hidden layer
    h = np.tanh(z1)                # activation (adds the non-linearity)
    z2 = h @ p["W2"] + p["b2"]     # weighted sum in the output
    y_hat = sigmoid(z2)            # becomes a probability 0..1
    return y_hat, h


def loss(y_hat, y):
    """Binary cross-entropy: punishes hard whoever is wrong with confidence."""
    eps = 1e-9
    return float(-np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)))


def backward(p, X, y, y_hat, h):
    """
    Backpropagation: the chain rule, from the output back to the input.
    It answers: 'if I nudge this weight a tiny bit, how much does the loss change?'
    That is the GRADIENT.
    """
    n = X.shape[0]
    dz2 = (y_hat - y) / n                  # derivative of the loss with respect to z2
    dW2 = h.T @ dz2
    db2 = dz2.sum(axis=0, keepdims=True)
    dh = dz2 @ p["W2"].T                   # the error "flows back" to the hidden layer
    dz1 = dh * (1 - h ** 2)                # derivative of tanh
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0, keepdims=True)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def weight_names(p):
    names = []
    for k in ["W1", "b1", "W2", "b2"]:
        for idx in np.ndindex(p[k].shape):
            names.append(f"{k}[{','.join(map(str, idx))}]")
    return names


def flatten(p):
    return np.concatenate([p[k].ravel() for k in ["W1", "b1", "W2", "b2"]])


# ---------------------------------------------------------------------------
# 3) CHECKING THE GRADIENT (gradient check)
# ---------------------------------------------------------------------------
def check_gradient(p, X, y):
    """
    Compares the gradient from the formula (backward) with a 'brute force' estimate:
    nudge the weight by +eps and -eps and measure the difference in the loss. If they
    match, backpropagation is correct. This is how people debug backprop in real life.
    """
    y_hat, h = forward(p, X)
    g = backward(p, X, y, y_hat, h)
    eps = 1e-5
    print("\n[Checking the gradient]  formula  vs  brute force")
    for k, idx in [("W1", (0, 0)), ("W1", (1, 2)), ("W2", (3, 0)), ("b2", (0, 0))]:
        original = p[k][idx]
        p[k][idx] = original + eps
        loss_plus = loss(forward(p, X)[0], y)
        p[k][idx] = original - eps
        loss_minus = loss(forward(p, X)[0], y)
        p[k][idx] = original
        numerical = (loss_plus - loss_minus) / (2 * eps)
        print(f"  {k}{list(idx)}:  {g[k][idx]:+.6f}  vs  {numerical:+.6f}")


# ---------------------------------------------------------------------------
# 4) TRAINING
# ---------------------------------------------------------------------------
def train(X, y, n_hidden, epochs, rate, record=True, explain=True):
    p = init_weights(X.shape[1], n_hidden)
    history = []
    snapshots = {}  # copies of the weights at a few moments, to draw later
    moments = {0, 50, 300, epochs}

    for epoch in range(epochs + 1):
        y_hat, h = forward(p, X)
        L = loss(y_hat, y)
        acc = float(np.mean((y_hat > 0.5) == y))
        g = backward(p, X, y, y_hat, h)
        grad_norm = float(np.sqrt(sum((g[k] ** 2).sum() for k in g)))

        if record:
            history.append([epoch, L, acc, grad_norm] + flatten(p).tolist())
        if epoch in moments:
            snapshots[epoch] = {k: v.copy() for k, v in p.items()}

        if explain and epoch < 3:
            w, gw = p["W2"][0, 0], g["W2"][0, 0]
            print(f"\nEpoch {epoch}: loss={L:.4f}  accuracy={acc:.0%}")
            print(f"  example with one weight, W2[0,0]:")
            print(f"    current value = {w:+.5f}")
            print(f"    gradient      = {gw:+.5f}   (+ sign => increasing the weight would increase the loss)")
            print(f"    new value     = {w:+.5f} - {rate} × {gw:+.5f} = {w - rate * gw:+.5f}")
        elif explain and epoch % 500 == 0:
            print(f"Epoch {epoch:5d}: loss={L:.4f}  accuracy={acc:.1%}  |gradient|={grad_norm:.4f}")

        if epoch == epochs:
            break
        # THE GRADIENT DESCENT STEP: walk a little bit "downhill"
        for k in p:
            p[k] -= rate * g[k]

    return p, history, snapshots


# ---------------------------------------------------------------------------
# 5) CHARTS
# ---------------------------------------------------------------------------
def prediction_grid(p, resolution=200):
    d = np.linspace(0, 1, resolution)
    s = np.linspace(0, 1, resolution)
    D, S = np.meshgrid(d, s)
    Xg = normalize(np.column_stack([D.ravel(), S.ravel()]))
    prob = forward(p, Xg)[0].reshape(D.shape)
    return D, S, prob


def plot_data(X, y):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[y[:, 0] == 0, 0], X[y[:, 0] == 0, 1], s=10, c="#9aa5b1", label="don't jump")
    ax.scatter(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], s=10, c="#e8590c", label="jump")
    ax.set_xlabel("distance to the cactus")
    ax.set_ylabel("game speed")
    ax.set_title("Training examples: when is it time to jump?")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "01_data.png"), dpi=110)
    plt.close(fig)


def plot_loss(hist):
    h = np.array(hist)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].plot(h[:, 0], h[:, 1], c="#1c7ed6")
    axs[0].set_title("Loss (error) — lower is better")
    axs[0].set_xlabel("epoch")
    axs[1].plot(h[:, 0], h[:, 2] * 100, c="#2f9e44")
    axs[1].set_title("Accuracy (%)")
    axs[1].set_xlabel("epoch")
    axs[1].set_ylim(0, 100)
    axs[2].plot(h[:, 0], h[:, 3], c="#ae3ec9")
    axs[2].set_yscale("log")
    axs[2].set_title("Size of the gradient (log scale)")
    axs[2].set_xlabel("epoch")
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "02_loss.png"), dpi=110)
    plt.close(fig)


def plot_weights(hist, names):
    h = np.array(hist)
    weights = h[:, 4:]
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
    for i, name in enumerate(names):
        ax = axs[0] if name.startswith(("W1", "b1")) else axs[1]
        ax.plot(h[:, 0], weights[:, i], lw=1.2, label=name)
    axs[0].set_title("Hidden layer (W1 and b1)")
    axs[1].set_title("Output layer (W2 and b2)")
    for ax in axs:
        ax.set_xlabel("epoch")
        ax.set_ylabel("weight value")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Each line is one weight. Notice: they change fast at first, then settle down.")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "03_weights.png"), dpi=110)
    plt.close(fig)


def hidden_lines(p):
    """
    Each hidden neuron computes w1·distance + w2·speed + b — the formula of a
    linear regression. Where that sum is zero, the neuron "flips" from negative
    to positive. Those points form a STRAIGHT LINE. Returns one line per neuron,
    in the chart's coordinates (0..1), plus how much each neuron matters to the output.
    """
    d = np.linspace(-0.5, 1.5, 2)
    lines = []
    for j in range(p["W1"].shape[1]):
        w_dist, w_speed, b = p["W1"][0, j], p["W1"][1, j], p["b1"][0, j]
        # The inputs are normalized (x * 2 - 1), so we solve the line in normalized units
        if abs(w_speed) < 1e-9:
            xs, ys = np.full(2, (-b / w_dist + 1) / 2), np.array([-0.5, 1.5])
        else:
            xs, ys = d, ((-b - w_dist * (d * 2 - 1)) / w_speed + 1) / 2
        lines.append((xs, ys))
    importance = np.abs(p["W2"][:, 0])
    return lines, importance / importance.max()


def plot_hidden_lines(snapshots, X, y):
    epochs = sorted(snapshots)
    fig, axs = plt.subplots(1, len(epochs), figsize=(4.2 * len(epochs), 4))
    for ax, ep in zip(axs, epochs):
        D, S, prob = prediction_grid(snapshots[ep])
        ax.contourf(D, S, prob, levels=20, cmap="RdBu_r", vmin=0, vmax=1, alpha=0.35)
        ax.scatter(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], s=4, c="k", alpha=0.3)
        lines, importance = hidden_lines(snapshots[ep])
        for j, ((xs, ys), imp) in enumerate(zip(lines, importance)):
            ax.plot(xs, ys, c="#5f3dc4", lw=0.8 + 3 * imp, alpha=0.9,
                    label="one line per hidden neuron" if j == 0 else None)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"epoch {ep}")
        ax.set_xlabel("distance")
    axs[0].set_ylabel("speed")
    axs[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Each hidden neuron draws one straight line (thicker = matters more to the output). "
                 "The output layer combines them into the band.")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "06_hidden_lines.png"), dpi=110)
    plt.close(fig)


def plot_boundaries(snapshots, X, y):
    epochs = sorted(snapshots)
    fig, axs = plt.subplots(1, len(epochs), figsize=(4.2 * len(epochs), 4))
    for ax, ep in zip(axs, epochs):
        D, S, prob = prediction_grid(snapshots[ep])
        ax.contourf(D, S, prob, levels=20, cmap="RdBu_r", vmin=0, vmax=1, alpha=0.8)
        ax.contour(D, S, prob, levels=[0.5], colors="k", linewidths=1.5)
        ax.scatter(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], s=4, c="k", alpha=0.4, label="answer key: 'jump' examples")
        ax.set_title(f"epoch {ep}")
        ax.set_xlabel("distance")
    axs[0].set_ylabel("speed")
    axs[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Background = what the network thinks (red = 'jump!', blue = 'wait'). "
                 "Dots = answer key, fixed from the start.")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "04_boundaries.png"), dpi=110)
    plt.close(fig)


def plot_single_neuron(X, y, Xn):
    """A single neuron draws only ONE straight line. The band needs two — it can't be done."""
    # a 'network' with no hidden layer = logistic regression
    w = rng.normal(0, 1, (2, 1))
    b = np.zeros((1, 1))
    for _ in range(EPOCHS):
        y_hat = sigmoid(Xn @ w + b)
        dz = (y_hat - y) / len(y)
        w -= LEARNING_RATE * Xn.T @ dz
        b -= LEARNING_RATE * dz.sum(axis=0, keepdims=True)
    acc = float(np.mean((sigmoid(Xn @ w + b) > 0.5) == y))

    d = np.linspace(0, 1, 200)
    D, S = np.meshgrid(d, d)
    prob = sigmoid(normalize(np.column_stack([D.ravel(), S.ravel()])) @ w + b).reshape(D.shape)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.contourf(D, S, prob, levels=20, cmap="RdBu_r", vmin=0, vmax=1, alpha=0.8)
    ax.contour(D, S, prob, levels=[0.5], colors="k")
    ax.scatter(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], s=4, c="k", alpha=0.4, label="answer key: 'jump' examples")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"Single neuron: accuracy {acc:.0%}\n(it can only draw one straight line)")
    ax.set_xlabel("distance")
    ax.set_ylabel("speed")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "05_single_neuron.png"), dpi=110)
    plt.close(fig)
    return acc


# ---------------------------------------------------------------------------
# 6) REPLAY — the recorded training, played back like a video
# ---------------------------------------------------------------------------
SPEEDS = [1, 2, 5, 10, 25, 50, 100]  # epochs advanced per frame


def unflatten(v, template):
    """The opposite of flatten(): turns the list of weights back into the network's shape."""
    p, i = {}, 0
    for k in ["W1", "b1", "W2", "b2"]:
        n = template[k].size
        p[k] = v[i:i + n].reshape(template[k].shape)
        i += n
    return p


class Replay:
    """Draws the state of training at any epoch recorded in the history."""

    def __init__(self, hist, template, X, y, Xn):
        self.h = np.array(hist)
        self.template = template
        self.y, self.Xn = y, Xn
        self.last = int(self.h[-1, 0])
        names = weight_names(template)

        self.fig, axs = plt.subplots(2, 2, figsize=(13, 8))
        self.ax_net, self.ax_loss, self.ax_weights, self.ax_grad = axs.ravel()

        self.D, self.S, prob = prediction_grid(self.weights(0), resolution=80)
        self.image = self.ax_net.imshow(prob, origin="lower", extent=(0, 1, 0, 1), aspect="auto",
                                        cmap="RdBu_r", vmin=0, vmax=1, alpha=0.85)
        self.contour = None
        # The straight line of each hidden neuron, redrawn every frame
        self.neuron_lines = [self.ax_net.plot([], [], c="#5f3dc4", alpha=0.9,
                                              label="hidden neuron lines" if j == 0 else None)[0]
                             for j in range(template["W1"].shape[1])]
        self.ax_net.set_xlim(0, 1)
        self.ax_net.set_ylim(0, 1)
        self.ax_net.scatter(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], s=4, c="k", alpha=0.4, label="answer key: 'jump' examples")
        self.ax_net.legend(loc="lower right", fontsize=8)
        self.ax_net.set_xlabel("distance")
        self.ax_net.set_ylabel("speed")
        self.ax_net.set_title("What the network thinks: red = 'jump!', blue = 'wait'")

        self.loss_line, = self.ax_loss.plot([], [], c="#1c7ed6", label="loss")
        self.acc_line, = self.ax_loss.plot([], [], c="#2f9e44", label="accuracy")
        self.ax_loss.set_xlim(0, self.last)
        self.ax_loss.set_ylim(0, max(1.05, self.h[:, 1].max() * 1.05))
        self.ax_loss.set_xlabel("epoch")
        self.ax_loss.set_title("Loss going down, accuracy going up")
        self.ax_loss.legend(loc="center right")
        self.ax_loss.grid(alpha=0.3)

        weights = self.h[:, 4:]
        self.weight_lines = [self.ax_weights.plot([], [], lw=1)[0] for _ in names]
        self.ax_weights.set_xlim(0, self.last)
        self.ax_weights.set_ylim(weights.min() - 0.5, weights.max() + 0.5)
        self.ax_weights.set_xlabel("epoch")
        self.ax_weights.set_title(f"The {len(names)} weights adjusting (each line is one weight)")
        self.ax_weights.grid(alpha=0.3)
        self.markers = [self.ax_loss.axvline(0, c="k", lw=0.8, alpha=0.4),
                        self.ax_weights.axvline(0, c="k", lw=0.8, alpha=0.4)]

        self.bars = self.ax_grad.bar(range(len(names)), np.zeros(len(names)))
        self.ax_grad.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
        self.ax_grad.axhline(0, c="k", lw=0.8)

        self.fig.text(0.5, 0.005, "SPACE pause  ·  ← → one epoch  ·  ↑ ↓ speed  ·  R restart",
                      ha="center", fontsize=9, color="#555")
        self.fig.tight_layout(rect=(0, 0.02, 1, 0.96))

    def weights(self, epoch):
        return unflatten(self.h[epoch, 4:], self.template)

    def draw(self, epoch, status=""):
        h = self.h
        p = self.weights(epoch)

        _, _, prob = prediction_grid(p, resolution=80)
        self.image.set_data(prob)
        if self.contour is not None:
            self.contour.remove()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # no boundary yet (all blue) raises a warning
            self.contour = self.ax_net.contour(self.D, self.S, prob, levels=[0.5],
                                               colors="k", linewidths=1.5)

        lines, importance = hidden_lines(p)
        for line, (xs, ys), imp in zip(self.neuron_lines, lines, importance):
            line.set_data(xs, ys)
            line.set_linewidth(0.8 + 3 * imp)

        upto = slice(0, epoch + 1)
        self.loss_line.set_data(h[upto, 0], h[upto, 1])
        self.acc_line.set_data(h[upto, 0], h[upto, 2])
        for i, line in enumerate(self.weight_lines):
            line.set_data(h[upto, 0], h[upto, 4 + i])
        for marker in self.markers:
            marker.set_xdata([epoch, epoch])

        # The gradient isn't stored in the history, but we can just recompute it from that epoch's weights.
        y_hat, hidden = forward(p, self.Xn)
        grad = flatten(backward(p, self.Xn, self.y, y_hat, hidden))
        for bar, g in zip(self.bars, grad):
            bar.set_height(g)
            bar.set_color("#e03131" if g > 0 else "#1971c2")
        lim = max(np.abs(grad).max() * 1.1, 1e-9)
        self.ax_grad.set_ylim(-lim, lim)
        self.ax_grad.set_title(f"Gradient of each weight (the weight moves the OPPOSITE way)\n"
                               f"total size = {h[epoch, 3]:.5f} — scale adjusted every frame",
                               fontsize=10)

        self.fig.suptitle(f"Epoch {epoch}/{self.last}   loss={h[epoch, 1]:.4f}   "
                          f"accuracy={h[epoch, 2]:.1%}   {status}")


def play_replay(replay):
    for k in ("keymap.back", "keymap.forward", "keymap.home"):
        plt.rcParams[k] = []  # frees ← → R, which matplotlib uses to navigate the chart
    state = {"epoch": 0, "speed": 1, "paused": False}

    def on_key(event):
        if event.key == " ":
            state["paused"] = not state["paused"]
        elif event.key == "right":
            state["epoch"] = min(state["epoch"] + 1, replay.last)
            state["paused"] = True
        elif event.key == "left":
            state["epoch"] = max(state["epoch"] - 1, 0)
            state["paused"] = True
        elif event.key in ("up", "+", "="):
            state["speed"] = min(state["speed"] + 1, len(SPEEDS) - 1)
        elif event.key in ("down", "-"):
            state["speed"] = max(state["speed"] - 1, 0)
        elif event.key == "r":
            state["epoch"], state["paused"] = 0, False

    replay.fig.canvas.mpl_connect("key_press_event", on_key)
    while plt.fignum_exists(replay.fig.number):
        speed = SPEEDS[state["speed"]]
        status = "[PAUSED]" if state["paused"] else f"[{speed} epoch(s) per frame]"
        replay.draw(state["epoch"], status)
        plt.pause(1 / 30)
        if not state["paused"]:
            if state["epoch"] >= replay.last:
                state["paused"] = True
            else:
                state["epoch"] = min(state["epoch"] + speed, replay.last)


def record_gif(replay, path):
    from matplotlib.animation import FuncAnimation, PillowWriter

    # Slow at the start, where everything happens; fast at the end, where almost nothing changes.
    frames = (list(range(0, 60, 2)) + list(range(60, 400, 10))
              + list(range(400, replay.last, 100)) + [replay.last] * 15)
    anim = FuncAnimation(replay.fig, replay.draw, frames=frames)
    anim.save(path, writer=PillowWriter(fps=12), dpi=55)


# ---------------------------------------------------------------------------
def main():
    X, y = make_data(N_EXAMPLES)
    Xn = normalize(X)
    print(f"{N_EXAMPLES} examples, {int(y.sum())} of them are 'jump'.")

    p_test = init_weights(2, N_HIDDEN)
    check_gradient(p_test, Xn, y)

    print("\n[Training the network 2 -> %d -> 1]" % N_HIDDEN)
    p, hist, snapshots = train(Xn, y, N_HIDDEN, EPOCHS, LEARNING_RATE)

    names = weight_names(p)
    with open(os.path.join(OUTPUT, "weight_history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss", "accuracy", "gradient_norm"] + names)
        for row in hist:
            w.writerow([row[0]] + [f"{v:.6f}" for v in row[1:]])

    # Test on NEW data the network has never seen
    Xt, yt = make_data(2000)
    test_acc = float(np.mean((forward(p, normalize(Xt))[0] > 0.5) == yt))

    plot_data(X, y)
    plot_loss(hist)
    plot_weights(hist, names)
    plot_boundaries(snapshots, X, y)
    plot_hidden_lines(snapshots, X, y)
    single_acc = plot_single_neuron(X, y, Xn)

    print("\n=== Results ===")
    print(f"Accuracy on training data:      {hist[-1][2]:.1%}")
    print(f"Accuracy on new data:           {test_acc:.1%}")
    print(f"Single neuron (for comparison): {single_acc:.1%}")
    print(f"Charts and CSV in: {OUTPUT}")
    if GIF:
        path = os.path.join(OUTPUT, "training.gif")
        print("Recording the animation...")
        record_gif(Replay(hist, p, X, y, Xn), path)
        print(f"Animation saved to: {path}")
    if LIVE:
        print("Opening the training replay. Close the window to exit.")
        play_replay(Replay(hist, p, X, y, Xn))
    elif not GIF:
        print("To watch the training like a video, run again with --live")


if __name__ == "__main__":
    main()
