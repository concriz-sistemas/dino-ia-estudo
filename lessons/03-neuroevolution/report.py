"""
report.py — Reads results/history.json and generates charts + REPORT.md.

Run from the repository root, after training:
    python lessons/03-neuroevolution/report.py

REPORT.md is saved in results/ and opens directly on GitHub, images included.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FOLDER)
from net_utils import draw_network, node_name

OUTPUT = os.path.join(FOLDER, "results")

# An explanation of each parameter worth understanding
EXPLANATIONS = {
    ("NEAT", "pop_size"): "How many dinos (networks) exist in each generation.",
    ("NEAT", "fitness_threshold"): "Score considered 'goal reached'.",
    ("NEAT", "no_fitness_termination"): "If True, runs every requested generation even after reaching the goal.",
    ("DefaultGenome", "num_inputs"): "How many sensors the network receives.",
    ("DefaultGenome", "num_outputs"): "How many actions the network can choose from.",
    ("DefaultGenome", "num_hidden"): "Hidden neurons at the start (NEAT adds more if needed).",
    ("DefaultGenome", "initial_connection"): "'full' = every input starts connected to every output.",
    ("DefaultGenome", "activation_default"): "The neurons' activation function.",
    ("DefaultGenome", "weight_init_stdev"): "How spread out the initial random weights are.",
    ("DefaultGenome", "weight_mutate_rate"): "Chance of each weight being mutated every generation.",
    ("DefaultGenome", "weight_mutate_power"): "Typical size of a mutation (a random 'nudge' to the weight).",
    ("DefaultGenome", "weight_replace_rate"): "Chance of a weight being replaced by a brand-new value.",
    ("DefaultGenome", "bias_mutate_rate"): "Chance of each bias being mutated.",
    ("DefaultGenome", "conn_add_prob"): "Chance of a new connection appearing.",
    ("DefaultGenome", "conn_delete_prob"): "Chance of a connection being removed.",
    ("DefaultGenome", "node_add_prob"): "Chance of a new neuron appearing (in the middle of a connection).",
    ("DefaultGenome", "node_delete_prob"): "Chance of a neuron being removed.",
    ("DefaultSpeciesSet", "compatibility_threshold"): "How different two networks must be to count as separate species.",
    ("DefaultStagnation", "max_stagnation"): "Generations without improvement before a species goes extinct.",
    ("DefaultReproduction", "elitism"): "How many of the best in each species pass unchanged to the next generation.",
    ("DefaultReproduction", "survival_threshold"): "Fraction of each species allowed to have 'children'.",
}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, name), dpi=110)
    plt.close(fig)


def plot_fitness(G, limit):
    x = [g["generation"] for g in G]
    best = np.array([g["best"] for g in G])
    average = np.array([g["average"] for g in G])
    stdev = np.array([g["stdev"] for g in G])
    median = np.array([g["median"] for g in G])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(x, np.maximum(0, average - stdev), average + stdev, color="#74c0fc", alpha=0.25,
                    label="average ± std. dev.")
    ax.plot(x, best, "o-", c="#e8590c", label="best dino")
    ax.plot(x, average, "o-", c="#1c7ed6", label="population average")
    ax.plot(x, median, "--", c="#1c7ed6", alpha=0.6, label="median")
    ax.axhline(limit, c="gray", ls=":", label=f"match limit ({limit})")
    ax.set_xlabel("generation")
    ax.set_ylabel("score (frames survived)")
    ax.set_title("Score (fitness) over the generations")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, "01_fitness.png")


def plot_species(G):
    ids = sorted({s["id"] for g in G for s in g["species"]})
    x = [g["generation"] for g in G]
    mat = np.zeros((len(ids), len(G)))
    for j, g in enumerate(G):
        for s in g["species"]:
            mat[ids.index(s["id"]), j] = s["size"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stackplot(x, mat, labels=[f"species {i}" for i in ids], alpha=0.85)
    ax.set_xlabel("generation")
    ax.set_ylabel("dinos")
    ax.set_title("Species: groups of similar networks that compete among themselves")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.set_xlim(x[0], x[-1])
    save(fig, "02_species.png")


def plot_complexity(G):
    x = [g["generation"] for g in G]
    hidden = [sum(1 for k in g["champion"]["nodes"] if int(k) >= 3) for g in G]
    connections = [sum(1 for c in g["champion"]["connections"] if c[3]) for g in G]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(x, connections, "o-", c="#2f9e44", label="enabled connections")
    ax.plot(x, hidden, "s-", c="#7048e8", label="hidden neurons")
    ax.set_xlabel("generation")
    ax.set_title("Network size of each generation's champion")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, "03_complexity.png")


def plot_champion_weights(G):
    """For each connection of the final champion, its weight in each generation's champion."""
    final = [(a, b) for a, b, w, enabled in G[-1]["champion"]["connections"] if enabled]
    x = [g["generation"] for g in G]
    fig, ax = plt.subplots(figsize=(11, 5))
    for a, b in final:
        ys = []
        for g in G:
            weights = {(c[0], c[1]): c[2] for c in g["champion"]["connections"] if c[3]}
            ys.append(weights.get((a, b), np.nan))
        ax.plot(x, ys, "o-", ms=3, lw=1.2, label=f"{node_name(a)} → {node_name(b)}")
    ax.axhline(0, c="k", lw=0.6)
    ax.set_xlabel("generation")
    ax.set_ylabel("weight")
    ax.set_title("Weights of the final champion's connections, across the generations\n"
                 "(gap in a line = that connection did not exist in that generation's champion)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    save(fig, "04_champion_weights.png")


def plot_distribution(G):
    x = [g["generation"] for g in G]
    data = [g["population_weights"] for g in G]
    fig, ax = plt.subplots(figsize=(11, 4))
    parts = ax.violinplot(data, positions=x, widths=0.8, showmedians=True)
    for body in parts["bodies"]:
        body.set_facecolor("#91a7ff")
        body.set_alpha(0.6)
    ax.axhline(0, c="k", lw=0.6)
    ax.set_xlabel("generation")
    ax.set_ylabel("weight")
    ax.set_title("Distribution of ALL the population's weights in each generation")
    ax.grid(alpha=0.3, axis="y")
    save(fig, "05_weight_distribution.png")


def plot_networks(G):
    n = len(G)
    chosen = sorted({0, n // 3, (2 * n) // 3, n - 1})
    fig, axs = plt.subplots(1, len(chosen), figsize=(4.6 * len(chosen), 4.2))
    axs = np.atleast_1d(axs)
    for ax, i in zip(axs, chosen):
        g = G[i]
        draw_network(ax, g["champion"], f"Champion of generation {g['generation']}\n{g['best']:.0f} points")
    fig.suptitle("Green = positive weight (excites), red = negative (inhibits). Thickness = strength.", fontsize=10)
    save(fig, "06_networks.png")


def parameters_table(params):
    lines = ["| Section | Parameter | Value | What it does |", "|---|---|---|---|"]
    for (sec, key), text in EXPLANATIONS.items():
        if sec in params and key in params[sec]:
            lines.append(f"| `{sec}` | `{key}` | {params[sec][key]} | {text} |")
    return "\n".join(lines)


def champion_table(c):
    lines = ["| From | To | Weight |", "|---|---|---|"]
    for a, b, w, enabled in sorted(c["connections"], key=lambda x: -abs(x[2])):
        if enabled:
            lines.append(f"| {node_name(a)} | {node_name(b)} | {w:+.3f} |")
    return "\n".join(lines)


def generations_table(G):
    lines = ["| Generation | Best | Average | Median | Species | Champion's connections |",
             "|---|---|---|---|---|---|"]
    for g in G:
        con = sum(1 for c in g["champion"]["connections"] if c[3])
        lines.append(f"| {g['generation']} | {g['best']:.0f} | {g['average']:.0f} | {g['median']:.0f} "
                     f"| {g['n_species']} | {con} |")
    return "\n".join(lines)


def main():
    path = os.path.join(OUTPUT, "history.json")
    if not os.path.exists(path):
        print("No history yet. First run: python lessons/03-neuroevolution/train.py")
        return
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    G = data["generations"]
    limit = data["score_limit"]

    plot_fitness(G, limit)
    plot_species(G)
    plot_complexity(G)
    plot_champion_weights(G)
    plot_distribution(G)
    plot_networks(G)

    first_goal = next((g["generation"] for g in G if g["best"] >= limit), None)
    last = G[-1]
    md = f"""# Training report — neuroevolution

> Generated automatically by `report.py` from `history.json`.
> Run the training again (with a different `SEED`) and this report changes — each seed is a different evolution.

## Summary

| | |
|---|---|
| Generations | {len(G)} |
| Dinos per generation | {data['parameters']['NEAT']['pop_size']} |
| First generation to reach {limit} points | {first_goal if first_goal is not None else 'not reached'} |
| Population average: generation 0 → last | {G[0]['average']:.0f} → {last['average']:.0f} |
| Final champion: connections / hidden neurons | {sum(1 for c in last['champion']['connections'] if c[3])} / {sum(1 for k in last['champion']['nodes'] if int(k) >= 3)} |

## 1. The score over the generations

![fitness](01_fitness.png)

The **best** climbs fast (all it takes is one lucky dino). The **average** climbs more slowly: that is the whole population inheriting what works.

## 2. Species

![species](02_species.png)

NEAT splits the networks into species, and each one competes mainly within its own group. This protects innovations (a new connection, which at first usually makes the network worse) from being wiped out before they have a chance to mature.

## 3. Network size

![complexity](03_complexity.png)

NEAT starts with the smallest network and only adds neurons and connections when that helps. It also removes what is not useful.

## 4. The weights adjusting

![champion weights](04_champion_weights.png)

Each line is a connection of the final champion, and shows how much it was worth in each generation's champion. The champions are different individuals, but related: genes (connections) are inherited and keep undergoing small mutations. Note that **there is no gradient here** — the weights change through random mutation + selection of the best.

![weight distribution](05_weight_distribution.png)

All the weights of all the dinos, per generation. In generation 0 they are random around zero; over time the distribution changes shape as selection favors certain values.

## 5. The champions' networks

![networks](06_networks.png)

## 6. The final champion, weight by weight

Sorted by the strength (absolute value) of the weight:

{champion_table(last['champion'])}

## 7. Parameters used in this training

{parameters_table(data['parameters'])}

The full file is in [`neat-config.txt`](../neat-config.txt).

## 8. Generation by generation

{generations_table(G)}
"""
    with open(os.path.join(OUTPUT, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Report generated: {os.path.join(OUTPUT, 'REPORT.md')}")


if __name__ == "__main__":
    main()
