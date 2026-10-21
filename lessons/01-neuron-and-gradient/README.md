**English** · [Português](README.pt-BR.md) · [Español](README.es.md)

# Lesson 1 — Neuron, loss and gradient (from scratch, numpy only)

> **Goal:** understand what happens *inside* when a neural network "learns".
> No AI libraries: every calculation is written by hand in
> [`train_from_scratch.py`](train_from_scratch.py).

```bash
python lessons/01-neuron-and-gradient/train_from_scratch.py --live
```

The script trains the network (about 5 seconds), records the value of every weight at every epoch and then **plays the training back in a window, like a video**: what the network thinks (with the straight line of each hidden neuron in purple), the loss going down, each of the 25 weights adjusting and the gradient of each weight at that moment.

| Key | What it does |
|---|---|
| SPACE | pause / resume |
| ← → | back / forward one epoch (to see one gradient step at a time) |
| ↑ ↓ | faster / slower (from 1 to 100 epochs per frame) |
| R | restart from zero |

The bottom of the window reminds you: "SPACE pause · ← → one epoch · ↑ ↓ speed · R restart".

![training replay](results/training.gif)

To record this animation again:

```bash
python lessons/01-neuron-and-gradient/train_from_scratch.py --gif
```

Without `--live`, the script trains in about 5 seconds, explains the first calculations in the terminal and saves the charts and the CSV to [`results/`](results/), which are the ones used in the rest of this page.

---

## 1. The problem

Before putting an AI in the game, let's solve one piece of it on its own:

> Given the **distance to the cactus** and the **game speed**, is it time to jump?

The right answer is a **band**: if the cactus is too close, it's already too late; if it's too far, you land before reaching it. And the faster the game, the earlier you need to jump.

![data](results/01_data.png)

The network **never sees this rule**. It only gets 600 examples (the dots above) and has to discover the pattern on its own. The terminal confirms it: "600 examples, 137 of them are 'jump'."

### What the dots in the charts are

They are the **answer key**. Before training, the script draws 600 random situations and uses the true rule to answer each one. From then on, in the network's charts, the black dots are the 137 examples marked as "jump" (the 463 "don't jump" ones are left out so the chart stays clean). The legend calls them "answer key: 'jump' examples".

The dots **never change**. What changes during training is the **colored background**: the network's answer for each spot on the plane. Training means making the background turn red where the dots are and blue everywhere else.

This is called **supervised learning**: the answers to the examples are known from the start, and the network learns by comparing what it answered with the answer key. In lessons 3 and 4 there is no answer key; the dino only gets a score or a reward.

---

## 2. The network

```
 inputs            hidden layer           output
 distance  ──┐   ┌─ neuron 1 ─┐
             ├───┼─ neuron 2 ─┼───  P(jump)
 speed     ──┘   └─ ...     6 ┘
```

Every neuron always does the same thing:

```
output = activation( weight₁·input₁ + weight₂·input₂ + ... + bias )
```

| Term | What it is | In the code |
|---|---|---|
| **Weight** | How much an input matters (can be negative) | `W1`, `W2` |
| **Bias** | A fixed "push" that shifts the neuron's threshold | `b1`, `b2` |
| **Activation** | A curve that bends the sum. Without it, the whole network would be just a straight line | `tanh`, `sigmoid` |
| **Forward** | Passing the inputs through the network to get the prediction | `forward()` |

> 🔍 **New to these words?** The [deep dive](concepts/README.md) explains weight, bias, activation, loss and gradient one at a time, with everyday analogies, charts, an animation and small examples worked out by hand.

The sum inside the activation, `weight₁·input₁ + weight₂·input₂ + bias`, is exactly the formula of a **linear regression**. Section 6 shows what that means in practice.

This network has 2×6 + 6 + 6×1 + 1 = **25 adjustable numbers**. Training = finding good values for these 25 numbers. (A large LLM has **hundreds of billions** of them — but the idea is the same.)

---

## 3. Loss: measuring how wrong the network is

To improve, the network needs a score. The **loss** is a number that gets big when the network is wrong and small when it is right. We use *binary cross-entropy*, which punishes hard whoever is wrong **with confidence** (saying 99% "jump" when the answer was "don't jump").

---

## 4. Gradient: which way should each weight move?

Picture the loss as hilly terrain, where each position is a combination of the 25 weights. We want to reach the valley.

The **gradient** answers, for each weight: *"if I increase this weight a tiny bit, does the loss go up or down? And by how much?"*

- **Positive** gradient → increasing the weight makes it worse → so we **decrease** it.
- **Negative** gradient → increasing the weight makes it better → so we **increase** it.

The whole learning rule is this line:

```python
weight = weight - learning_rate * gradient
```

The script shows this really happening in the first epochs:

```
Epoch 0: loss=0.9365  accuracy=46%
  example with one weight, W2[0,0]:
    current value = +1.93379
    gradient      = -0.12817   (+ sign => increasing the weight would increase the loss)
    new value     = +1.93379 - 0.5 × -0.12817 = +1.99788
```

### How is the gradient calculated? Backpropagation

The `backward()` function applies the **chain rule** from calculus, from back to front: first the error at the output, then how much each hidden neuron contributed to it, then how much each input weight contributed to the neuron. That's why it's called *backpropagation*: the error flows **back** through the network.

### How do we know the math is right? Numerical check

The script compares the formula with a brute-force estimate (it nudges the weight by +0.00001 and −0.00001 and measures the difference in the loss). If they match, backpropagation is correct. In the terminal this appears under "[Checking the gradient]  formula  vs  brute force":

```
  W1[0, 0]:  -0.084432  vs  -0.084432
  W2[3, 0]:  +0.259212  vs  +0.259212
```

---

## 5. Watching the training

### The learning curve

![loss](results/02_loss.png)

- The **loss** drops sharply in the first few hundred epochs and then goes down slowly.
- The **accuracy** goes up from ~46% (guessing) to 100%.
- The **gradient** keeps shrinking: near the valley, the terrain gets flat and the steps get small.

### The weights changing

![weights](results/03_weights.png)

Each line is one of the 25 numbers. The CSV [`results/weight_history.csv`](results/weight_history.csv) has the exact value of all of them at **every epoch** — open it in Excel and explore.

### What the network "thinks" during training

![boundaries](results/04_boundaries.png)

The title of this chart sums it up: "Background = what the network thinks (red = 'jump!', blue = 'wait'). Dots = answer key, fixed from the start."

- **`epoch 0`:** random weights, the network guesses.
- **`epoch 50`:** it started to notice that "far = don't jump".
- **`epoch 300`:** it has already found the band, still blurry.
- **`epoch 4000`:** sharp band, practically the same as the true rule.

### Two phases: fast descent, then fine-tuning

In the replay (`--live`) you can see that the band shows up in the first few hundred epochs and then barely changes. There are two phases:

1. **Fast descent (epochs 0 to ~300).** With random weights, the network is very wrong, so the gradient is big. Since the step is `rate × gradient`, the steps are big too, and the shape appears. The loss drops sharply.
2. **Fine-tuning (from ~300 on).** The network is already close to the bottom of the loss valley. The terrain gets flat, the gradient gets small (watch the "total size" in the gradient panel drop from ~1 at epoch 0 to ~0.03 at 300 and ~0.002 at 4000) and each step moves the weights very little. This is called **convergence**.

The shape doesn't stand completely still: the weights keep growing slowly (weights panel), and that makes the edges of the band **sharper**. The network is getting more *confident*. With cross-entropy, you can always lower the loss a little more by betting with more certainty, so the weights grow without stopping, more and more slowly.

Section 8 shows these two phases from above, on a map of the loss.

---

## 6. Each neuron is a linear regression

Every neuron computes `w₁·distance + w₂·speed + b`, the formula of a **linear regression**. Where that sum is zero, the neuron flips from "no" to "yes", and those points form a **straight line** on the chart. The activation (`tanh` or `sigmoid`) only bends the answer near the line; it doesn't change where the line is.

### A single neuron: one line is not enough

![single neuron](results/05_single_neuron.png)

A neuron with a sigmoid and no hidden layer is a **logistic regression**: it can only draw **one straight line**. The band needs two edges, so it gives up and learns "never jump" — which gets 77% right only because most examples are "don't jump". This is a classic trap: **high accuracy doesn't mean it learned**.

### Six neurons: six lines, combined into a band

![hidden neuron lines](results/06_hidden_lines.png)

Each purple line is where one hidden neuron flips. The thicker the line, the more that neuron matters to the output.

- **`epoch 0`:** all six lines cross in the middle of the chart. The biases start at zero, so every line passes through the center point; only the angles are random.
- **`epoch 50` and `epoch 300`:** the lines spread out and rotate toward the dots.
- **`epoch 4000`:** two neurons sit exactly on the two edges of the band, with strong output weights of opposite signs (+9.7 and −9.4): one says "the cactus is close enough", the other says "but not too close". The other four moved their lines out of the playing area (to distances above 2) and barely matter anymore.

So the network only needed 2 of its 6 neurons, which matches what `N_HIDDEN = 2` shows in the exercises. The output layer combines the two lines into the band. **More neurons = more lines to combine = more complicated shapes.**

You can watch these lines move in the replay (`--live`).

---

## 7. Hyperparameters: the training knobs

They are at the top of the script. They are not learned — **you** choose them.

| Parameter | Value | What happens if you change it |
|---|---|---|
| `LEARNING_RATE` | 0.5 | Too small (0.01): after 4000 epochs it still hasn't left "never jump" (78%). Too big (20): huge steps, the loss jumps by up to 10× before settling down. |
| `N_HIDDEN` | 6 | With 1: it can't get the band. With 2 it already can (2 lines = 2 edges). More neurons give room for more complicated patterns. |
| `EPOCHS` | 4000 | Too few: it stops halfway (see epoch 300). |
| `N_EXAMPLES` | 600 | With 20: 100% on training, but only ~91% on new data — the network "memorized" the examples (*overfitting*). |
| `SEED` | 42 | Changes the initial weights and the data. Similar final results, different paths. |

---

## 8. The loss valley: learning rate and cosine schedule

The 25 weights make a loss "terrain" with 25 dimensions, impossible to draw. To see gradient descent from above, [`loss_valley.py`](loss_valley.py) shrinks the problem to the simplest neuron there is: **no activation at all**, a pure linear regression with only two weights:

```
jump distance = w · speed + b
```

The task still comes from the game: the faster the game, the earlier you must jump. The script makes 200 noisy examples and walks downhill four times from the same starting point, changing only the learning rate.

```bash
python lessons/01-neuron-and-gradient/loss_valley.py
```

![loss valley](results/loss_valley.gif)

Each point of the map is one possible line `(w, b)`; darker means smaller loss. The gold star is the bottom of the valley, the best line. After 60 steps:

| Walk | Learning rate | What happens | Loss after 60 steps |
|---|---|---|---|
| gray | 0.05, fixed | Safe, but too slow: still far from the bottom | 0.069 |
| green | 0.5, fixed | Goes down and settles at the bottom | 0.00097 |
| red | 0.78, fixed | Overshoots the bottom, bounces from side to side, farther each time, and blows up | 4.5 |
| purple | 0.78 → 0, cosine | Starts with the same "too big" rate and bounces, but the rate shrinks and it calms down at the bottom | 0.0011 |

The bottom of the valley has a loss of 0.00095.

Three things to notice:

- **The valley is a long, narrow bowl.** Going across it is steep, along it is gentle. A big step overshoots across the steep side and bounces; that is the zigzag. Bigger networks can have valleys like this in many dimensions at once.
- **The two phases of section 5 show up here too.** The fast descent is the first few steps, when the paths drop toward the valley; the fine-tuning is the long, slow crawl along the bottom.
- **The cosine schedule** gets the best of both: big steps at the start, when the network is far from the bottom, and small steps at the end, when it is close. The learning rate follows half a cosine wave, from 0.78 down to 0 (bottom-right chart). Until about step 50 the purple path has a lower loss than the green one, and it ends practically tied with it. Big models, including LLMs, are usually trained with a schedule like this instead of a fixed rate.

A neuron with an activation (like the ones in `train_from_scratch.py`) has a curvier valley than this perfect bowl, but the walk is the same: `weight = weight - learning_rate * gradient`, over and over.

---

## Exercises

1. Set `LEARNING_RATE = 20` and look at `02_loss.png`. Then try `0.01`. What happened in each case?
2. Try `N_HIDDEN = 1` and then `2`. What is the minimum number of neurons to draw the band? Why? Look at `04_boundaries.png`.
3. Set `N_EXAMPLES = 20`. Compare the accuracy on training with the accuracy on new data.
4. Open the CSV and find the weight that changed the most. Which input is it connected to?
5. In `loss_valley.py`, change the "too big" rate from 0.78 to 0.75. Does it still blow up? Then try 0.8. Somewhere around 0.76 is the edge between "bounces but settles" and "blows up".

---

## How does this relate to LLMs?

An LLM is trained with **exactly** this cycle — forward, loss, gradient, step — except that:

- the input is a piece of text and the output is the probability of each next word;
- there are hundreds of billions of weights instead of 25;
- the data is trillions of words instead of 600 dots;
- they use smarter variations of gradient descent (like *Adam*), on thousands of GPUs.

---

## Beyond this lesson: cosine similarity

> **Not used in this lesson.** This is here as an analogy, because "cosine" comes up a lot when people talk about LLMs, and it has nothing to do with the cosine schedule of section 8.

A **vector** is just a list of numbers. Each hidden neuron in this lesson has one: its two input weights, `(weight of the distance, weight of the speed)`. That vector sets the direction of the neuron's line.

**Cosine similarity** measures how much two vectors point the same way, ignoring their size: 1 = same direction, 0 = unrelated, −1 = opposite directions.

In the trained network:

- The two neurons on the edges of the band have vectors (−8.91, 3.07) and (−8.15, 1.97). Their cosine similarity is **0.996**: they point almost the same way, which is why the two edges of the band are almost parallel.
- Another neuron, with tiny weights (−0.17, 0.08), also scores **0.995** against the first one. It points the same way but is 50 times weaker. Cosine similarity only looks at direction, not size.

LLMs use the same tool on a much bigger scale. They turn each word or text into a vector with hundreds or thousands of numbers (an *embedding*). Texts with similar meaning end up pointing in similar directions, so "dog" and "puppy" have a high cosine similarity and "dog" and "spreadsheet" a low one. That is how search engines and assistants find text with a similar meaning, even when the words are different. [Lesson 7](../07-language-model/README.md) does it for real, with a tiny language model.

---

**Next lesson:** [the environment — the game the AI will play](../02-the-environment/README.md)
