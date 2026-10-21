"""
net_utils.py — Helper functions used by the training, the report and the viewer.

- genome_to_dict(): turns a NEAT genome into a plain dictionary
  (nodes with their biases, connections with their weights). This is what goes
  into the history file — you can open it and read it in a text editor.
- draw_network(): draws a network with matplotlib.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from dino.game import SENSOR_NAMES

INPUT_NAMES = list(SENSOR_NAMES)
OUTPUT_NAMES = ["nothing", "jump", "duck"]


def node_name(key):
    """In NEAT, inputs have negative keys (-1..-6), outputs 0..2, hidden nodes 3+."""
    if key < 0:
        return INPUT_NAMES[-key - 1]
    if key < len(OUTPUT_NAMES):
        return OUTPUT_NAMES[key]
    return f"hidden {key}"


def genome_to_dict(genome):
    return {
        "id": genome.key,
        "fitness": genome.fitness,
        "nodes": {str(k): round(n.bias, 4) for k, n in genome.nodes.items()},
        "connections": [
            [a, b, round(c.weight, 4), bool(c.enabled)]
            for (a, b), c in sorted(genome.connections.items())
        ],
    }


def draw_network(ax, g, title=""):
    """Draws a genome (in the genome_to_dict format) on a matplotlib axis."""
    inputs = list(range(-1, -len(INPUT_NAMES) - 1, -1))
    outputs = list(range(len(OUTPUT_NAMES)))
    hidden = sorted(int(k) for k in g["nodes"] if int(k) not in outputs)

    pos = {}
    for i, k in enumerate(inputs):
        pos[k] = (0.0, 1 - (i + 0.5) / len(inputs))
    for i, k in enumerate(hidden):
        pos[k] = (0.5, 1 - (i + 0.5) / max(1, len(hidden)))
    for i, k in enumerate(outputs):
        pos[k] = (1.0, 1 - (i + 0.5) / len(outputs))

    for a, b, w, enabled in g["connections"]:
        if not enabled or a not in pos or b not in pos:
            continue
        color = "#2f9e44" if w > 0 else "#e03131"
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color=color, lw=min(5, 0.4 + abs(w)), alpha=0.7, zorder=1)

    for k in inputs:
        ax.scatter(*pos[k], s=220, c="#d0ebff", edgecolors="#1c7ed6", zorder=2)
        ax.text(pos[k][0] - 0.06, pos[k][1], node_name(k), ha="right", va="center", fontsize=8)
    for k in hidden:
        ax.scatter(*pos[k], s=180, c="#e5dbff", edgecolors="#7048e8", zorder=2)
    for k in outputs:
        ax.scatter(*pos[k], s=220, c="#fff3bf", edgecolors="#f08c00", zorder=2)
        ax.text(pos[k][0] + 0.06, pos[k][1], node_name(k), ha="left", va="center", fontsize=8)

    ax.set_xlim(-0.45, 1.4)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")
    ax.set_title(title, fontsize=10)
