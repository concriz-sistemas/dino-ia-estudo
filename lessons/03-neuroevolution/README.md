**English** · [Português](README.pt-BR.md) · [Español](README.es.md)

# Lesson 3 — Neuroevolution: 50 dinos evolving until they play well

> **Goal:** train neural networks to play **without a gradient**, using only natural selection,
> and record every generation so you can watch the weights change.

![champions of several generations](results/champions.gif)

*The champion of generations 0, 7, 14 and 29 playing the same match. Generation 0's champion crashes early (129 points); the others are still alive when their clip ends (the GIF stops each clip at 600 points).*

Run them one at a time, in this order.

1. Watch the evolution. The V key speeds it up, and ESC stops and saves what has been recorded so far:

```bash
python lessons/03-neuroevolution/train.py
```

2. Generate the charts and `REPORT.md`:

```bash
python lessons/03-neuroevolution/report.py
```

3. Watch the champion of generation 0:

```bash
python lessons/03-neuroevolution/watch.py --generation 0
```

4. And compare it with the champion of the last generation. The ← → arrows change the generation, and R starts a new match:

```bash
python lessons/03-neuroevolution/watch.py
```

To train without a window (about 3 seconds): `train.py --fast`. To change the number of generations (30 by default): `train.py --generations 50`.

To record the champions GIF again, after training:

```bash
python lessons/03-neuroevolution/watch.py --gif
```

📊 **[See the report of an example run](results/REPORT.md)**

---

## The idea: evolution instead of gradient

In [Lesson 1](../01-neuron-and-gradient/README.md), each weight was adjusted by **calculating** which direction made the loss go down. Here there is no calculation at all:

```
1. Create 50 networks with random weights
2. Make all of them play the same match
3. Each one's score = how many frames it survived
4. The best become "parents"; the children inherit the weights with small random mutations
5. Go back to step 2
```

Nobody programmed "jump when the cactus is close". That rule **emerges** because, by chance, some dinos jump at the right moment, survive and pass their weights on.

| | Gradient (Lesson 1) | Neuroevolution (this lesson) |
|---|---|---|
| How it decides to change a weight | Calculates the derivative of the loss | Random mutation + selection |
| Needs the "right answer"? | Yes, labeled examples | No, just a score at the end |
| Changes the network's structure? | No, it is fixed | Yes, it creates and removes neurons |
| Scales to billions of weights? | Yes (it is how LLMs are trained) | No, it gets far too slow |
| Good for | Lots of data, large networks | Small problems, no gradient available |

---

## NEAT

**NEAT** (*NeuroEvolution of Augmenting Topologies*, 2002) is the algorithm used here, through the `neat-python` library. It has three tricks:

1. **Starts minimal:** only inputs connected to outputs, no hidden layer. Neurons and connections are **added by mutation** when they help.
2. **Genes with history:** each connection has an "innovation" number, so two parents can cross networks of different shapes without scrambling everything.
3. **Species:** similar networks compete among themselves. A freshly created innovation (which at first usually makes the network worse) gets time to mature without being wiped out by the established champions.

---

## What gets recorded

`train.py` has a `Recorder` that, at the end of **every generation**, saves to `results/history.json`:

- the best score, average, median and standard deviation of the population;
- the species and their sizes;
- the **full champion**: every neuron with its bias and every connection with its weight;
- **all the weights** of the whole population.

`history.json` is plain text. Open it in an editor and search for `"champion"` to see a whole network in numbers.

`report.py` turns this into charts and into [`REPORT.md`](results/REPORT.md):

| Chart | What it shows |
|---|---|
| `01_fitness.png` | Best, average and median per generation |
| `02_species.png` | How the population splits into species |
| `03_complexity.png` | How many neurons and connections the champion has |
| `04_champion_weights.png` | The value of each connection of the final champion across the generations |
| `05_weight_distribution.png` | All the weights of all the dinos, generation by generation |
| `06_networks.png` | Drawing of the champion's network at 4 moments |

---

## What to look for in the example report

NEAT draws random mutations, but `train.py` fixes the starting point of that randomness with `SEED = 0`: it seeds Python's `random` before creating the population. So **your run gives exactly the same numbers** as the example run in [`results/REPORT.md`](results/REPORT.md) (30 generations, 50 dinos per generation). The values in parentheses come from it. If you change `SEED`, you get a different evolution with other numbers, but the patterns repeat:

- **The best climbs in jumps, the average climbs slowly.** One lucky dino reaches 5000 early (generation 6 in the example), but the whole population takes much longer to improve (the average goes from 121 in generation 0 to 1084 in the last one).
- **The best of a generation can be worse than the previous one's** (669 → 316 → 235 in generations 0, 1 and 2 of the example; and after reaching 5000 in generation 6, generation 7 dropped to 1477, then 5000 in every generation from 8 to 29). Each generation plays a **different** match, with another sequence of obstacles, and sometimes the match is harder. This is on purpose: if they all played the same level, the dinos would memorize the level instead of learning to play (*overfitting*).
- **The median stays low** even while the average rises. Most children get bad mutations and die early. Evolution is wasteful: many try, few succeed.
- **The network's size varies from run to run.** Sometimes the mutations cut connections and the champion ends up simpler than it started, with 18 connections (6 connections and 1 hidden neuron in the example); sometimes it gains hidden neurons. See `03_complexity.png` and `06_networks.png`. NEAT does not look for the "prettiest" network, just one that works.
- **The main rule always shows up: obstacle far = don't jump.** In the example it comes directly, as a connection `distance` → `jump` with weight **−5.204**, the strongest of the final champion. With other seeds it can go through a hidden neuron instead: `distance` → hidden neuron with a positive weight, and hidden neuron → `jump` with a negative weight. Positive times negative is negative, that is, the same rule by another path. Nobody taught the dinos this.

The other strong weights of the example's final champion: `speed` → `duck` (−1.921), `speed` → `nothing` (−0.993), `hidden 178` → `jump` (+0.632) and `elevation` → `hidden 178` (+0.585). The full table is in section 6 of the report.

---

## The parameters (`neat-config.txt`)

The most important ones to experiment with (the full list, with explanations, is in the report):

| Parameter | Value | What it does |
|---|---|---|
| `pop_size` | 50 | Dinos per generation. More = more chances per generation, but each generation takes longer. |
| `weight_mutate_rate` | 0.8 | Chance of each weight being mutated. |
| `weight_mutate_power` | 0.5 | Size of the mutation. It is the "equivalent" of the learning rate. |
| `node_add_prob` | 0.2 | Chance of a new neuron appearing. |
| `conn_add_prob` | 0.5 | Chance of a new connection appearing. |
| `compatibility_threshold` | 3.0 | Smaller = more species. |
| `elitism` | 2 | The 2 best pass unchanged to the next generation (a good network is never lost to mutation). |
| `survival_threshold` | 0.2 | Only the top 20% of each species reproduce. |

## Exercises

1. Run `watch.py --generation 0` and then `watch.py`. Use ← → to go through the generations and see at which point the dino "gets" the birds.
2. Change `pop_size` to 10 and then to 200. Run `train.py --fast` and `report.py`. In which generation does it reach 5000? And in how much real time?
3. Set `weight_mutate_power = 3.0`. What happens to the average chart?
4. In `train.py`, change `seed=self.n` to `seed=1` (every generation plays the same level). Train and use `watch.py`, which always draws a new level. Does the champion still play well?
5. Change the score: in `eval_genomes`, set `genome.fitness = dino.score - 10 * (number of jumps)`. Does the playing style change? (You will need to count the jumps.)
6. In `train.py`, change `SEED = 0` to another number and run `train.py --fast` and `report.py`. Compare your new `REPORT.md` with the example: in which generation does it reach 5000? How big is the final champion? Does the rule `distance` → `jump` show up directly or through a hidden neuron? (This overwrites `results/`; put `SEED = 0` back and train again to get the example back.)

**Next lesson:** [reinforcement learning with gradient (PPO)](../04-reinforcement/README.md)
