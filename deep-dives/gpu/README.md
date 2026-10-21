**English** · [Português](README.pt-BR.md) · [Español](README.es.md)

# Deep dive: why GPUs?

> Why does everyone train neural networks on graphics cards? What calculations are actually happening,
> and why do they need so much computing power? With a real measurement on the Mac used for this study.

To repeat the measurement on your own computer (about 1 minute):

```bash
python deep-dives/gpu/benchmark.py
```

Then, to redraw the charts that turn the numbers into time (a few seconds):

```bash
python deep-dives/gpu/scale.py
```

---

## 1. What a neural network actually computes

Almost everything is one operation, repeated an enormous number of times: **multiply two numbers and add the result to a running total** (a *multiply-add*).

Take the first layer of [lesson 1](../../lessons/01-neuron-and-gradient/README.md). For each of the 600 examples, each of the 6 hidden neurons computes `weight₁ × distance + weight₂ × speed + bias`. That's 600 × 6 = 3,600 small sums, each with 2 multiply-adds. Written as a table of numbers, it is a **matrix multiplication**:

```
examples (600 × 2)  ×  weights (2 × 6)  =  sums (600 × 6)
```

Every network in this study works like this. Bigger networks just have bigger tables: a layer of an LLM multiplies tables with thousands of rows and columns.

Two things matter about these multiply-adds:

- **There are a lot of them.** Multiplying two 4096 × 4096 tables takes 4096³ ≈ 69 billion multiply-adds.
- **They don't depend on each other.** The sum for example 1, neuron 1 doesn't need the one for example 2, neuron 5. They can all be done **at the same time**.

Training costs about 3 times as much as just using the network, because the backward pass (backpropagation, lesson 1) needs about two more matrix multiplications for every one in the forward pass.

---

## 2. CPU vs GPU: a few master chefs or an army of helpers

**Analogy.** A **CPU** is a kitchen with a few master chefs (in this measurement, PyTorch used 4 of them). Each one is very fast and can follow any complicated recipe, step by step, deciding what to do next. A **GPU** is a kitchen with **thousands of helpers**. Each one is simpler and slower, and they all have to do the same kind of task, but they work at the same time.

To cook one elaborate dish, the chefs win. To chop 69 billion onions, the helpers win, as long as you have enough onions to keep them all busy.

![a few chefs vs many helpers](charts/fill.gif)

*Filling the result table of a matrix multiplication. Each cell is one row times one column, and no cell needs another one to be ready first. With 4 workers, 256 cells take 64 steps; with 64 workers, 4 steps. A single helper is slower than a chef, but there are so many of them that, on a big enough table, they win.*

Matrix multiplication is chopping onions: billions of identical, independent multiply-adds. That's why GPUs, designed to compute millions of pixels at the same time for games, turned out to be perfect for neural networks.

---

## 3. Measured on this Mac

The machine: a **Mac mini with an Apple M4 chip** (10 cores, 4 of them high-performance) and 24 GB of memory.

![speed](charts/02_speed.png)

| Matrix size | CPU | GPU | Faster |
|---|---|---|---|
| 32 × 32 | 0.001 ms | 0.22 ms | CPU, about 200× |
| 256 × 256 | 0.024 ms | 0.24 ms | CPU, about 10× |
| 1024 × 1024 | 1.15 ms | 1.68 ms | CPU, a little |
| 2048 × 2048 | 9.5 ms | 5.3 ms | GPU, 1.8× |
| 4096 × 4096 | 75 ms | 41.5 ms | GPU, 1.8× |

At its best, this Mac's GPU did **3.3 trillion operations per second**, and its CPU **1.8 trillion**.

Two surprises:

- **Small multiplications are much faster on the CPU.** Every job sent to the GPU has a fixed cost, about 0.2 ms here, to prepare it, send it and wait for the answer. For a small table, that cost is much bigger than the work itself. It's calling the whole kitchen staff to chop one onion.
- **On this Mac, the GPU is "only" 1.8× faster, even on big tables.** Apple's chips have a special unit for matrix math next to the CPU, so their CPU is unusually good at this. On a typical computer, the CPU is much slower at matrix multiplication, and the gap to a graphics card is much bigger.

![tiny steps](charts/03_tiny_steps.png)

This also explains the choices in the study. Lessons 3, 4 and 5 run on the CPU: their networks are tiny (lesson 4's has 2,692 weights), every step is a small multiplication, and the game itself runs on the CPU. In the chart above, 2,000 tiny steps in a row take about the same time on both (25 ms on the CPU, 29 ms on the GPU). Lessons 6 and 7 use the GPU, because images and text batches are big enough to keep it busy.

---

## 4. How much computing did each lesson need?

Counting the multiply-adds of each training run (one multiply-add = 2 operations):

| Training | Approximate operations | How it's counted |
|---|---|---|
| Lesson 1: the 25-weight network | 0.26 billion | 36 per example × 3 (training) × 600 examples × 4,000 epochs |
| Lesson 4: PPO | about 0.3 trillion | 2,692 weights, 2 million steps, 10 passes over the data per update |
| Lesson 7: the tiny transformer | 5.5 trillion | 6 × 111,029 weights × 8.2 million tokens read |
| Lesson 6: the eyes (CNN) | about 76 trillion | 20 million per image × 3 × 1.3 million images seen in training |
| A large LLM (example) | about 6 million million million million (6 × 10²⁴) | 6 × 70 billion weights × 15 trillion tokens |

The "6 × weights × tokens" rule is a common way to estimate the cost of training a language model: 2 operations per weight for each token read, times 3 for training. The large LLM line uses sizes like those published for open models with 70 billion weights.

Numbers this big are hard to picture, so here they are as time. First, **if a person did it by hand**, one multiply-add per second, without ever stopping:

![by hand](charts/04_by_hand.png)

Even the 25 weights of lesson 1 would take a person about **4 years**. Lesson 7's tiny transformer, about **87,000 years**.

And on real machines, at the best speeds measured above:

| Training | This Mac's CPU | This Mac's GPU |
|---|---|---|
| Lesson 1 | under a millisecond | under a millisecond |
| Lesson 4 | 0.18 seconds | 0.1 seconds |
| Lesson 7 | 3 seconds | 1.7 seconds |
| Lesson 6 | 41 seconds | 23 seconds |

These are *ideal* times, at full speed all the time. The real trainings took longer (lesson 7 took about 26 seconds; lesson 6, about 22 minutes, because it also plays the game and its batches are small). The order of size is what matters.

Now the large LLM, on one shared scale:

![how long](charts/05_how_long.png)

- On this Mac's GPU: about **60,000 years**.
- On one current data-center GPU: these do on the order of a thousand trillion operations per second (10¹⁵) in the number formats used for training, but in practice they run well below their peak; assuming 40% of it, about **500 years**.
- On **10,000 of them at once**: about **18 days**.

That's why big models are trained on **thousands of GPUs at once**, for weeks or months, and why it costs so much: not one hard calculation, but an astronomical number of easy ones.

---

## 5. And when the model answers?

Using a trained LLM also costs about **2 operations per weight for each token it writes**. For a model with 70 billion weights, that's about 140 billion operations per token, which is why writing an answer takes noticeable time and why chat services need big GPU servers. Writing is also harder to spread across helpers than training: each token depends on the one before, so the model has to write one at a time (lesson 7's `generate.py` does exactly that).

---

## Summary

| | CPU | GPU |
|---|---|---|
| Analogy | a few master chefs | thousands of helpers |
| Good at | complicated, step-by-step work; small jobs | the same simple job on huge amounts of data |
| In this study | lessons 1-5: small networks, the game | lessons 6-7: images and text in big batches |
| Fixed cost per job | almost none | about 0.2 ms here: not worth it for tiny jobs |

**Back to:** [the study home](../../README.md)
