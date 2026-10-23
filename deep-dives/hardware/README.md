**English** · [Português](README.pt-BR.md) · [Español](README.es.md)

# Deep dive: chips, memory and quantization

> What kinds of chips run neural networks, from a gaming graphics card to a data center? Why does the
> *memory* matter as much as the chip? What can a regular computer run or train? And what is
> quantization, the trick that lets big models run on small machines?

The charts come from two scripts. The second one uses lesson 7's model, so train lesson 7 first if you haven't:

```bash
python deep-dives/hardware/memory.py
```

```bash
python deep-dives/hardware/quantization.py
```

This page builds on [Why GPUs?](../gpu/README.md). Specs change fast; the numbers here are published specs at the time of writing.

---

## 1. The kinds of chips

| Kind | Examples | Memory the model can use | Memory speed | Good for |
|---|---|---|---|---|
| **Regular PC, CPU only** | Intel Core, AMD Ryzen, with DDR3, DDR4 or DDR5 RAM | whatever RAM it has (8-16 GB on older PCs, 32 GB is common today) | ~26 GB/s (DDR3) · ~51 GB/s (DDR4) · ~80-100 GB/s (DDR5) | small models, and anything that isn't a big network |
| **Gaming graphics card** | NVIDIA GeForce RTX 4060 / 4090 / 5090, AMD Radeon | 8 / 24 / 32 GB on the card | 272 / 1,008 / 1,792 GB/s | fast, as long as the model fits in the card's memory |
| **Unified memory** (CPU and GPU share one pool) | Apple M-series chips, AMD Ryzen AI Max | up to ~75% of the total on a Mac by default; up to 96 GB on the Ryzen AI Max+ 395 | 120 GB/s (M4) to 546 GB/s (M4 Max); 256 GB/s (Ryzen AI Max+ 395) | big models on one machine, more slowly than a big graphics card |
| **Data-center GPU** | NVIDIA H100 / B200, AMD Instinct MI300X | 80 / 192 / 192 GB | 3,350 / 8,000 / 5,300 GB/s | training and serving big models, in groups of thousands |

There are also chips made only for neural networks, like Google's TPUs, used in Google's own data centers.

**Analogy.** Think of a restaurant kitchen. The chip is the cooks; the memory is the **pantry**. *Memory size* is how big the pantry is: if the ingredients don't fit, you can't cook the dish at all. *Memory speed* (bandwidth) is how wide the pantry door is: how many ingredients can reach the cooks per second. Thousands of cooks (a GPU) are useless if the ingredients come through a narrow door.

---

## 2. Why memory matters so much

### The model has to fit

The size of a model is simple arithmetic: **number of weights × bytes per weight**. Usually each weight is stored in 16 bits (2 bytes), so a model with 7 billion weights takes 14 GB, and one with 70 billion takes 140 GB.

![does it fit](charts/04_does_it_fit.png)

- A **7B model in 4 bits** (3.5 GB) fits almost anywhere, even on this study's Mac mini.
- A **70B model in 16 bits** (140 GB) only fits on the biggest data-center GPUs. On anything else it has to be split across several GPUs, or shrunk (section 4).
- **Training** needs much more than running: besides the weights, it keeps the gradient of every weight and the optimizer's memory (Adam keeps two running averages per weight). A common estimate is **about 16 bytes per weight**, so training even a 7B model needs about 112 GB: out of reach for one gaming card.

### The memory speed sets how fast it writes

To write **one token**, an LLM has to read **all of its weights** from memory once (lesson 7's `generate.py` runs the whole network for every character). So, however fast the chip is, there is a ceiling:

```
tokens per second  ≤  memory speed  ÷  model size
```

![tokens per second](charts/05_tokens_per_second.png)

| Device | 7B model, 4 bits (3.5 GB) | 70B model, 4 bits (35 GB) |
|---|---|---|
| Regular PC, CPU only, DDR3 | 7 tokens/s at most | doesn't fit in 16 GB |
| Regular PC, CPU only, DDR4 | 15 | doesn't fit in 32 GB |
| Regular PC, CPU only, DDR5 | 26 | doesn't fit in 32 GB |
| GeForce RTX 4090 | 288 | doesn't fit |
| Mac mini M4 (this study) | 34 | doesn't fit |
| Mac with M4 Max, 128 GB | 156 | 16 |
| AMD Ryzen AI Max+ 395, 128 GB | 73 | 7 |
| NVIDIA H100 | 957 | 96 |
| NVIDIA B200 | 2,286 | 229 |

These are ceilings; real speeds are lower. Published tests of 70B models in 4 bits get about 4 to 6 tokens per second on the Ryzen AI Max+ 395, below its ceiling of 7, as expected.

This is why the **kind of memory** matters:

- **DDR3 / DDR4 / DDR5 / LPDDR5** (regular RAM): large and cheap, but a narrow door. Each generation roughly doubles the width: an older PC with DDR3 still runs small models, just more slowly (for this study's lessons 1-5, any of them is plenty).
- **GDDR6 / GDDR7** (gaming cards): soldered right next to the graphics chip, with a much wider door, but only 8 to 32 GB.
- **HBM** (data-center GPUs): stacks of memory chips placed right next to the GPU, connected by thousands of wires. The widest door of all, and very expensive.
- **Unified memory** (Apple, AMD Ryzen AI Max): one big pool shared by CPU and GPU, so a big model fits, with a door in between the other two.

---

## 3. What a regular computer can and can't do

| | A regular computer | A data center |
|---|---|---|
| **Run** a small model (up to ~8B weights, in 4 bits) | ✅ comfortably | ✅ |
| **Run** a 70B model | ⚠️ only with lots of memory (a big unified-memory machine), in 4 bits, slowly | ✅ fast |
| **Train** something like this study's models | ✅ seconds to minutes | ✅ |
| **Train** a large LLM from scratch | ❌ memory and time are out of reach | ✅ thousands of GPUs for weeks or months |
| **Fine-tune** a small model a little | ⚠️ possible with tricks that train only a few extra weights | ✅ |

And beyond memory and speed:

- **Power and heat.** A GeForce RTX 5090 is rated at 575 W, an H100 at 700 W, and a B200 at about 1,000 W, each one, and data centers run thousands of them.
- **Connections between GPUs.** Training on thousands of GPUs only works because they're connected by very fast links (900 GB/s per H100), so they can share their results all the time. Regular computers don't have that.

---

## 4. Quantization: storing weights with fewer bits

**Analogy.** Think of prices in a store. €3.4719 is exact, but €3.47 is good enough, and €3 would start to cause problems. **Quantization** does the same with the weights of a network: instead of storing each one with 32 or 16 bits, it rounds them to a few allowed values, stored with 8, 4 or even 2 bits.

It's the same as storing a picture with fewer colors:

![colors](charts/03_colors.png)

With 16 colors the chart looks almost the same. With 4, the smooth shading becomes bands. With 2, only the general shape is left.

### What happens to the weights

![rounding](charts/01_rounding.png)

These are the 16,384 weights of one table of lesson 7's model. With 4 bits, each weight is moved to the nearest of 15 allowed values (in orange). With 2 bits there are only 3 allowed values, and almost every weight becomes 0.

### The experiment: lesson 7's model, rounded

[`quantization.py`](quantization.py) rounds every weight table of lesson 7's tiny language model and measures it again on the unseen text:

![quality](charts/02_quality.png)

| Bits per weight | Size | Loss on unseen text, one ruler per table | Loss, one ruler per row | What it writes after "the dino " (one ruler per row) |
|---|---|---|---|---|
| 32 (original) | 434 KB | 1.642 | | `only matches teacher's when example find of the 292020000 15` |
| 8 | 114 KB | 1.643 | 1.642 | `only matches teacher's when example find of the 292020000 15` |
| 6 | 87 KB | 1.647 | 1.643 | `only matches teacher's when example find of the 292020000 15` |
| 4 | 61 KB | 1.806 | 1.699 | `only model step for from -1.99. the mird of the 200020000 15` |
| 3 | 47 KB | 2.996 | 1.982 | `only mateat the age appeare enviou + into siticndary; alll 5eps` |
| 2 | 34 KB | 4.552 | 4.122 | `o lh maeeap, led olenas es uess sluteinananho` |

- **8 and 6 bits change almost nothing.** The model writes the same text, with a quarter of the size or less.
- **4 bits costs a little**, especially with one ruler per row.
- **2 bits destroys it**: worse than a model that guesses every character at random (loss 3.97).

**One ruler per table vs one per row.** Rounding needs a *scale*, the size of the steps between allowed values. With one scale for the whole table, rows with small weights waste most of the steps: their weights all round to the same few values. Giving each row its own scale (or each group of a few dozen weights, as real tools do) keeps much more detail. At 3 bits, it's the difference between a model that still writes almost-English (1.98) and one that doesn't (3.00).

This is how people run big models at home: models are often shared already quantized to around 4 bits. A 70B model goes from 140 GB to about 35 GB and can fit on a big unified-memory machine. The price is a small loss in quality, which, as the experiment shows, grows quickly below 4 bits.

---

## Summary

| Question | Short answer |
|---|---|
| Which chip? | For neural networks: a GPU, or a chip like it. Gaming cards are fast but have little memory; unified-memory machines have lots of memory but a narrower door; data-center GPUs have both, at a huge price. |
| Why does memory matter? | The model must fit (weights × bytes per weight), and writing each token reads all the weights, so memory speed caps the speed. |
| Can I train an LLM at home? | A tiny one, like lesson 7's, yes. A large one, no: memory and time are out of reach. |
| What is quantization? | Rounding the weights to fewer bits: 8 bits is almost free, 4 bits costs a little, 2 bits breaks the model. |

**Back to:** [the study home](../../README.md)
