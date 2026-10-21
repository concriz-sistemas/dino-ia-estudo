**English** · [Português](README.pt-BR.md) · [Español](README.es.md)

# 🦖 Dino AI — learning neural networks step by step

A hands-on study: from the first neuron written by hand to a population of dinosaurs that teaches itself to play Chrome's offline dinosaur game.

Each lesson is a folder with a **README explaining the concepts**, a **short, commented script** and the **charts the script generates**, so you can watch the learning happen.

![champions of several generations](lessons/03-neuroevolution/results/champions.gif)

---

## Roadmap

| # | Lesson | What you learn | Time to run |
|---|---|---|---|
| 1 | [Neuron and gradient](lessons/01-neuron-and-gradient/README.md) | Weight, bias, activation, loss, **gradient**, backpropagation, hyperparameters, overfitting — all from scratch with numpy | ~5 s |
| 2 | [The environment](lessons/02-the-environment/README.md) | The game in pygame, what the AI "sees" (sensors), actions and reward | you play |
| 3 | [Neuroevolution (NEAT)](lessons/03-neuroevolution/README.md) | Training **without a gradient**, by selection and mutation. A record of every generation, weight charts and an automatic report | ~3 s without a window |
| 4 | [Reinforcement learning (PPO)](lessons/04-reinforcement/README.md) | A dino learns by gradient from rewards: actor, critic, advantage, clip. Where the learning is stored, and the link to fine-tuning LLMs | ~2 min without a window |
| 5 | [Imitation](lessons/05-imitation/README.md) | You play and a network learns to copy you, or copies lesson 4's agent. Supervised learning from demonstrations, and why LLMs imitate before they use reinforcement | ~2 s to train |
| 6 | [Learning to see](lessons/06-vision/README.md) | A convolutional network (CNN) reads the screen instead of the 6 numbers: filters, layers, and why a network fails on scenes it never saw (DAgger fixes it) | ~20 min to train; trained eyes included |
| 7 | [A tiny language model](lessons/07-language-model/README.md) | A tiny transformer, the design of an LLM, learns to write by predicting the next character: tokens, attention, temperature, embeddings and cosine similarity for real, and memorizing vs learning | ~30 s to train |
| 8 | [Hard mode: when the world changes](lessons/08-hard-mode/README.md) | A harder game with bird formations. The old agents fail, reinforcement learning alone can't learn it, a planner solves it, and a student that imitates a coach learns it (and almost forgets the old game) | ~1 min to watch; experiments 5-20 min |

Common questions, such as "does anyone teach the AI the rule?" and "where is the learning stored?", are answered in [Questions that come up along the way](#questions-that-come-up-along-the-way).

---

## Installation

You need **Python 3.9+**.

Copy and run one command at a time:

```bash
git clone https://github.com/remofurlanetto/dino-ai-study.git
```

```bash
cd dino-ai-study
```

```bash
python3 -m venv .venv
```

Activate the environment. On Mac or Linux:

```bash
source .venv/bin/activate
```

On Windows: `.venv\Scripts\activate`

```bash
pip install -r requirements.txt
```

**Every command runs from the repository root**, for example:

```bash
python lessons/01-neuron-and-gradient/train_from_scratch.py --live
```

Every time you open a new terminal, run `source .venv/bin/activate` first, or Python won't find the libraries.

## Step by step

Do the lessons in order. Each step has a command, what you will see and what you learn from it. First, in every new terminal:

```bash
source .venv/bin/activate
```

### Step 1: a neural network learning with a gradient

```bash
python lessons/01-neuron-and-gradient/train_from_scratch.py --live
```

The terminal shows the first gradient calculations, weight by weight. Then the training replay opens:

![lesson 1 replay](lessons/01-neuron-and-gradient/results/training.gif)

Keys: **SPACE** pauses, **← →** moves one epoch, **↑ ↓** changes the speed, **R** restarts.

Notice how the red band shows up quickly and then only gets sharper, while the gradient bars shrink. [Lesson 1](lessons/01-neuron-and-gradient/README.md) explains why.

### Step 2: the game the AI will play

```bash
python lessons/02-the-environment/play.py
```

You play (SPACE jumps, ↓ ducks) and see in the corner the 6 numbers the AI gets instead of eyes. [Lesson 2](lessons/02-the-environment/README.md).

### Step 3: 50 dinos evolving

```bash
python lessons/03-neuroevolution/train.py
```

In each generation, 50 dinos with different networks play together; the best ones have children with mutations. The ↑ ↓ arrows change the speed (1x to 64x) and V jumps to maximum speed:

![every 3rd generation, sped up](lessons/03-neuroevolution/results/evolution.gif)

Then generate the charts and the report:

```bash
python lessons/03-neuroevolution/report.py
```

And watch the champion of each generation play, with its network drawn beside it (← → changes generation):

```bash
python lessons/03-neuroevolution/watch.py
```

![champions of several generations](lessons/03-neuroevolution/results/champions.gif)

[Lesson 3](lessons/03-neuroevolution/README.md).

### Step 4: reinforcement learning

```bash
python lessons/04-reinforcement/train.py
```

A single dino plays thousands of matches and adjusts its weights by gradient to earn more reward (`+0.1` per frame alive, `−1` when it crashes). It combines the gradient from step 1 with the score from step 3. It takes about 2 minutes, without a window. Then generate the charts:

```bash
python lessons/04-reinforcement/report.py
```

And watch the dino play, with the chance of each action and the expected value beside it (← → changes saved model):

```bash
python lessons/04-reinforcement/watch.py
```

![lesson 4 models](lessons/04-reinforcement/results/models.gif)

The lesson also answers where the learning is stored (in a weights file, like an LLM) and shows the link to fine-tuning LLMs. [Lesson 4](lessons/04-reinforcement/README.md).

### Step 5: imitation

```bash
python lessons/05-imitation/train.py
```

A student network learns to play by copying a teacher: here, 10 matches recorded from lesson 4's agent. One recorded match and 2 seconds of training are enough for a perfect player. Then watch it next to its teacher's choices:

```bash
python lessons/05-imitation/watch.py
```

![the lesson 5 student](lessons/05-imitation/results/student.gif)

And teach it yourself: play a few matches and every frame is recorded.

```bash
python lessons/05-imitation/record.py
```

[Lesson 5](lessons/05-imitation/README.md).

### Step 6: learning to see

```bash
python lessons/06-vision/watch.py
```

The dino plays only from the screen. A convolutional network (the eyes) reads the image and estimates the 6 numbers; lesson 4's agent (the brain) decides from them. The trained eyes are included; training them yourself takes about 20-25 minutes:

```bash
python lessons/06-vision/train_eyes.py
```

![eyes + brain playing from the screen](lessons/06-vision/results/eyes.gif)

[Lesson 6](lessons/06-vision/README.md).

### Step 7: a tiny language model

```bash
python lessons/07-language-model/train.py
```

A tiny transformer, the same design as an LLM, reads the READMEs of this study and learns to predict the next character. In about 30 seconds it goes from random letters to text that sounds like English. Then make it write:

```bash
python lessons/07-language-model/generate.py
```

![the tiny language model writing](lessons/07-language-model/results/generation.gif)

[Lesson 7](lessons/07-language-model/README.md).

### Step 8: hard mode, when the world changes

```bash
python lessons/08-hard-mode/watch.py
```

A harder game, with tunnels and staircases of birds that can only be passed by ducking. Every agent trained on the original game fails. Reinforcement learning alone can't learn the new game, even with a curriculum and more exploration; a planner that searches solves it but is slow; and a student that imitates a coach learns it, at 5000 on both games.

![lesson 4's agent and lesson 8's student on the same match](lessons/08-hard-mode/results/before_after.gif)

[Lesson 8](lessons/08-hard-mode/README.md).

### Recording the animations

The GIFs on this page are generated by the scripts themselves:

```bash
python lessons/01-neuron-and-gradient/train_from_scratch.py --gif
```

```bash
python lessons/03-neuroevolution/train.py --gif
```

```bash
python lessons/03-neuroevolution/watch.py --gif
```

```bash
python lessons/04-reinforcement/watch.py --gif
```

```bash
python lessons/06-vision/watch.py --gif
```

---

## Structure

```
dino-ai-study/
├── dino/game.py                      # the game (used in lessons 2, 3 and 4)
├── dino/sprites.py                   # the pixel art figures, editable as text
└── lessons/
    ├── 01-neuron-and-gradient/
    │   ├── README.md                 # the theory, with the charts
    │   ├── train_from_scratch.py     # a neural network with numpy only
    │   └── results/                  # charts + CSV with the weights at every epoch
    ├── 02-the-environment/
    │   ├── README.md
    │   └── play.py                   # you play and see the AI's sensors
    ├── 03-neuroevolution/
    │   ├── README.md
    │   ├── neat-config.txt           # the evolution parameters, commented
    │   ├── train.py                  # trains and records every generation
    │   ├── report.py                 # generates charts + REPORT.md
    │   ├── watch.py                  # watch the champion of any generation play
    │   └── results/                  # history.json, charts, REPORT.md
    ├── 04-reinforcement/
    │   ├── README.md
    │   ├── environment.py            # the game in Gymnasium format (observation, action, reward)
    │   ├── train.py                  # trains with PPO and saves the model at several points
    │   ├── report.py                 # charts, decision map + REPORT.md
    │   ├── watch.py                  # the dino playing, with its "mind" beside it
    │   └── results/models/           # the learning: files with the network's weights
    ├── 05-imitation/
    │   ├── README.md
    │   ├── record.py                 # you play (or lesson 4's agent plays) and every frame is recorded
    │   ├── train.py                  # a student network learns to imitate the recordings
    │   ├── experiments.py            # strong, weak and slow teachers: is the student as good?
    │   ├── watch.py                  # the student playing, next to its teacher's choices
    │   └── results/demos/            # the recorded demonstrations (CSV)
    ├── 06-vision/
    │   ├── README.md
    │   ├── vision.py                 # the camera, the CNN eyes and lesson 4's agent as the brain
    │   ├── train_eyes.py             # trains the eyes, with DAgger rounds
    │   ├── pixels_to_keys.py         # the other approach: from pixels straight to keys
    │   ├── watch.py                  # eyes + brain playing, with what the eyes see
    │   └── results/eyes.pt           # the trained eyes
    ├── 07-language-model/
    │   ├── README.md
    │   ├── tiny_lm.py                # the text, the bigram and the tiny transformer
    │   ├── train.py                  # trains it and keeps the best model on unseen text
    │   ├── generate.py               # makes it write, with temperature
    │   ├── inside.py                 # embeddings, cosine similarity and attention
    │   └── data/study.txt            # the training text: this study's READMEs
    └── 08-hard-mode/
        ├── README.md
        ├── hard_game.py              # the hard game (the original one is untouched)
        ├── planner.py                # a planner that searches, without learning
        ├── test_agents.py            # every agent of the study on both games
        ├── rl_attempts.py            # four reinforcement learning attempts
        ├── train_student.py          # a student that imitates a coach, with DAgger
        └── watch.py                  # lesson 4's agent and the student on the same match
```

---

## Deep dives

Slower, more visual pages on topics that come up along the way:

- [Weight, bias, activation and loss](lessons/01-neuron-and-gradient/concepts/README.md): each idea with an everyday analogy, a chart, an animation and a small example worked out by hand.
- [Why GPUs?](deep-dives/gpu/README.md): what a neural network actually computes, CPU vs GPU measured on a Mac, and how much computing each lesson (and a large LLM) needs.

---

## Quick glossary

| Term | In one sentence |
|---|---|
| **Weight** | A number that says how much an input matters to a neuron. It is what training adjusts. |
| **Bias** | A number added to the neuron, which shifts the point where it "turns on". |
| **Activation** | A curve applied to the neuron (tanh, sigmoid...). Without it, the network could only draw straight lines. |
| **Loss** | A number that measures how wrong the network is. Training tries to lower it. |
| **Gradient** | For each weight: which way and how much to move it to lower the loss. |
| **Backpropagation** | The efficient way to compute the gradient, from the output back to the input. |
| **Learning rate** | The size of the step taken along the gradient. |
| **Epoch** | One pass through all the training examples. |
| **Hyperparameter** | A setting you choose instead of one that is learned (learning rate, network size...). |
| **Overfitting** | Memorizing the training examples instead of learning the general pattern. |
| **Fitness / reward** | The score for how well the AI did in the environment. |
| **Generation** | One round of evolution: everyone plays, the best ones reproduce. |
| **Mutation** | A random change in a weight or in the structure of the network. |
| **Species** | A group of similar networks that compete with each other in NEAT. |

The first five (weight, bias, activation, loss and gradient) are explained slowly, with analogies, charts and an animation, in the [lesson 1 deep dive](lessons/01-neuron-and-gradient/concepts/README.md).

---

## Questions that come up along the way

**Does anyone teach the AI the rule?**
No. In no lesson does the AI receive the rule "jump when the cactus is close". In lesson 1, the rule exists in the code only to generate 600 already-answered examples, and the network learns from the examples without ever seeing the rule. In lessons 3 and 4 not even that: the AI only gets a score or a reward, which says whether it worked, never how to do it.

**Does the dino learn by itself to see where the obstacles are?**
No, and that is on purpose. In no lesson does the dino see the screen. The game code measures everything and hands over 6 ready-made numbers: the distance, width, height and elevation of the next obstacle, the dino's height and the speed. It is like giving it a distance sensor instead of eyes. What it learns by itself is **what to do** with those numbers: when to jump, when to duck, when to wait. Learning to see from the screen's pixels, as DeepMind did with Atari games, requires a convolutional network and much more training. That is [lesson 6](lessons/06-vision/README.md): there, a CNN learns to estimate those 6 numbers from the screen, and the dino plays from pixels alone.

**What about lesson 1, then?**
Lesson 1 has no dino playing at all. It is an exercise with ready-made examples (the *answer key*, the dots in the charts) to watch the gradient work up close. The ones that really learn to play, with nobody knowing the answer, are the dinos of lessons 3 and 4.

**So is the learning random?**
Only partly. The weights start out random in every lesson. In lessons 1 and 4, from then on the gradient computes which way to move each weight, with no dice involved. In lesson 3, the mutations are random, but the selection is not: the dinos that get further have children. Chance proposes and the score decides. That is why each seed gives different numbers, but the main rule ("far = don't jump") always shows up.

**Why does the shape in the replay appear quickly and then barely change?**
At first the network is very wrong, the gradient is large and each step moves the weights a lot. Near the bottom of the loss "valley", the gradient becomes tiny (from ~1 to ~0.002 in lesson 1) and the weights almost stop. This is called convergence. [Lesson 1](lessons/01-neuron-and-gradient/README.md) has the full explanation, including the link to linear regression.

**Why does lesson 4's training stall and then jump?**
For about 600 thousand steps, the dino is stuck on "jump nonstop", which earns about 150 points and is hard to give up (a *local optimum*). When it discovers that waiting for the cactus to come close pays much more, the average shoots up to almost 5000. In the first tests, without tweaks, it did not leave the plateau within 500 thousand steps, and with 2 million it only left near the end. What brought the jump forward was making each action last 2 frames and giving a bigger bonus for exploring.

**Does the AI do what a human would?**
Not always. Facing a bird at head height, lesson 4's dino stays ducked while the bird is far away and jumps over it when it gets close. Nobody taught it that, and it works. The reward pays for the result, not for the method.

**Why does the dino play better in testing than in training?**
During training, lesson 4's agent still picks random actions to explore, so sometimes it makes mistakes on purpose. In testing it always picks the most likely action. That is why the training average stays around 4,300, while the final model scored 5000 in all 10 new matches.

**Once trained, is the learning stored somewhere?**
Yes, in a weights file. Lesson 4's dino is the file `lessons/04-reinforcement/results/models/ppo_02000k.zip`: about 62 KB holding 2,692 numbers. Loading that file means having the dino that knows how to play, without training again. In lesson 1, the last row of `weight_history.csv` has the final weights; in lesson 3, `champions.pkl` stores the champion of each generation.

**Is this similar to an LLM?**
It is the same idea at a gigantic scale. An LLM is also a weights file, only with billions of numbers and anywhere from a few to hundreds of gigabytes. And, as here, using it does not change the file: running `watch.py` a thousand times teaches the dino nothing, and an LLM does not learn while it talks with you. It only changes when a new training run produces a new file, that is, a new version of the model.

**What is overfitting?**
It's when a model **memorizes its training examples instead of learning the pattern behind them**. It does great on what it has already seen and badly on anything new, like a student who memorizes the answers to the practice sheet and then fails the exam because the questions changed. It shows up twice in this study: in lesson 1, a network trained on only 20 examples got 100% of them right but only about 91% of new situations; in lesson 7, the language model kept getting better on its training text while getting worse on text it had never seen. That's why every lesson keeps part of the data aside: it's the only way to tell learning from memorizing.

**So does an LLM really "think", or does it just repeat what it memorized?**
Partly memorizes, but mostly not. An LLM is trained on far more text than its weights could store word for word, so to do well it has to keep **patterns**: grammar, common facts, how an explanation or a piece of code usually goes. The tiny model of lesson 7 did the same at its scale: nobody told it that "3" and "4" are numbers, but it grouped them because that helped it guess. And you can measure the copying: [lesson 7](lessons/07-language-model/README.md#measuring-it-is-the-text-new-or-copied) checks every character it writes against the training text and colors the copied parts. At its best moment the tiny model writes mostly new combinations; trained too long without dropout, half of what it writes is pasted. Big LLMs show both behaviors too: they can repeat some texts word for word, especially ones they saw many times, and they get worse far from what they saw, like the eyes of lesson 6. Whether that counts as "thinking" is an open question that depends on what we mean by the word. What can be measured is that they generalize well beyond memorizing, with impressive successes and strange mistakes.

---

## How does this relate to LLMs?

They use the same building blocks as these lessons, at a huge scale:

1. **Pre-training** — the cycle of **Lesson 1** (forward, loss, gradient, step), with billions of weights learning to predict the next word in trillions of texts. In a way it is already imitation: the model copies how people write. [Lesson 7](lessons/07-language-model/README.md) builds a tiny one.
2. **Supervised fine-tuning** — **Lesson 5**: people write examples of good answers and the model learns to imitate them.
3. **Reinforcement learning** — the cycle of **Lesson 4** (act, get a reward, adjust by gradient): the model generates answers, gets a score (from people or from automatic tests) and is adjusted to score better.

The neuroevolution of **Lesson 3** is not used to train LLMs (it would be far too slow with billions of weights), but it is the most visual way to understand that **"learning" means adjusting numbers until a score improves**.

---

## Libraries used

- [numpy](https://numpy.org/) — matrix math
- [matplotlib](https://matplotlib.org/) — charts
- [pygame](https://www.pygame.org/) — the game (window, drawing, keyboard)
- [neat-python](https://github.com/CodeReclaimers/neat-python) — the NEAT algorithm
- [Gymnasium](https://gymnasium.farama.org/) — the standard format for reinforcement learning environments
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — the PPO algorithm (it uses [PyTorch](https://pytorch.org/), which is a large download)
