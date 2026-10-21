**English** · [Português](README.pt-BR.md) · [Español](README.es.md)

# Lesson 6 — Learning to see: the dino plays from the screen

> **Goal:** until now the dino got 6 ready-made numbers from the game. Here it gets only **the screen**,
> and a convolutional network (CNN) learns to read it.
> It's the same kind of training as "cat or dog" on photos, with labels the game gives for free.

![eyes + brain playing](results/eyes.gif)

*Playing only from the screen. The orange line is where the eyes think the next obstacle starts. Below: the small image the eyes receive, and the 6 numbers, true (from the game) and estimated (by the eyes).*

Run them one at a time, in this order.

1. Watch eyes + brain play. The trained eyes are already in the repository. R starts a new match:

```bash
python lessons/06-vision/watch.py
```

2. Train the eyes yourself. It takes about 20-25 minutes on a recent Mac and longer on a computer without a graphics card that PyTorch can use:

```bash
python lessons/06-vision/train_eyes.py
```

3. Try the other approach, from pixels straight to keys (a few minutes):

```bash
python lessons/06-vision/pixels_to_keys.py
```

This lesson uses PyTorch, which was already installed for lesson 4.

---

## 1. From 6 numbers to pixels

The screen is 800 × 300 pixels, but most of it doesn't matter. [`vision.py`](vision.py) crops the area from the top of a jump to just below the ground, and up to 600 pixels in front of the dino, shrinks every 3 × 5 block of pixels into one and turns it to grayscale. The network gets a **200 × 44 image**, with the dark things (dino, cacti, birds) turned bright.

One image is not enough. In a photo you can't tell how fast something moves, and the speed decides when to jump. So the network gets **two images: the current one and the one from 3 frames ago**. The difference between them is the movement.

![what each layer sees](results/04_what_layers_see.png)

The top row shows what the network receives: the scene now, and "now minus 3 frames ago", where the cactus appears twice (red where it is now, blue where it was).

---

## 2. A convolutional network

A **convolution** slides a small filter (here 5 × 5 pixels) over the whole image and, at each position, measures how much that piece of the image looks like the filter. The same filter is used everywhere, so a cactus on the left and a cactus on the right are detected the same way. That is what makes CNNs good at images.

The eyes have 3 convolutional layers and then a dense layer that turns what they found into the 6 sensor numbers: 630,406 weights in total.

- **Layer 1** (the second row of the image above) reacts to small details: the outline of the dino and of the cactus.
- **Layer 2** combines them into small shapes.
- **Layer 3** sees much less detail but bigger things: one blob for the dino, one for the cactus. It says "something is here".

These are the 16 filters of the first layer, as learned:

![filters](results/03_filters.png)

In textbooks, first-layer filters look like neat edge detectors. These look noisier: the images are tiny and the training is short. Some still show horizontal or vertical stripes of bright and dark.

One extra trick: convolutions are good at saying **what** is in the image but not **where**. The eyes get an extra input channel that just says "how far to the right is this pixel" (a technique called *CoordConv*), which helps them measure the distance.

---

## 3. Two ways to play from pixels

**End to end** ([`pixels_to_keys.py`](pixels_to_keys.py)): one network, from the two images straight to "nothing / jump / duck", copying lesson 4's agent, like the student of lesson 5.

**Eyes + brain** ([`train_eyes.py`](train_eyes.py)): the CNN only estimates the 6 sensor numbers (the **eyes**). Lesson 4's agent, unchanged, decides from those numbers (the **brain**). The brain doesn't know whether the numbers came from the game or from the eyes.

| Approach | Agreement with the teacher | Mean score in 10 new matches | Matches with 5000 |
|---|---|---|---|
| End to end, pixels → keys | 94.5% of the frames | 3013 | 4 |
| Eyes + brain, round 0 (only the teacher's scenes) | — | 316 | 0 |
| Eyes + brain, after 6 DAgger rounds | — | **5000** | **10** |

Without any extra trick, the end-to-end network did better than eyes + brain. But when the end-to-end network crashes, there's no way to know why: it is one block from pixels to keys. With eyes + brain you can compare, scene by scene, what the eyes estimated with the truth. That is what showed the problem, and what led to the fix.

---

## 4. Why the first eyes failed

In round 0 the eyes learn only from scenes recorded while the brain plays with the **true** sensors. Their average error near obstacles is 55 pixels, and eyes + brain crash early in every match (mean 316).

Following one crash frame by frame, in an earlier version of these eyes, showed the cause. While the cactus is far, lesson 4's agent likes to **duck**. When the cactus gets close, it is normally already jumping, so the recorded scenes almost never show **a ducking dino right next to a cactus**. With the eyes in charge, the jump comes a tiny bit late, the dino is still ducking in that position, and the eyes, facing a scene they have never seen, go wrong: the estimated distance *grows* while the cactus comes closer. The brain thinks the cactus is far away and doesn't jump.

It's the trap from lesson 5 again, now in perception: a network trained only on the teacher's scenes doesn't know what to do in the scenes its own mistakes lead to.

---

## 5. The fix: DAgger

**DAgger** (*Dataset Aggregation*) closes that gap:

1. Eyes + brain play **together**, mistakes and all.
2. Every scene they get into is recorded with the **true** sensor numbers as the answer key (the game always knows them).
3. The eyes keep training on everything recorded so far.
4. Repeat.

Each round adds exactly the scenes the system was getting wrong.

![rounds](results/01_rounds.png)

| Round | Scenes recorded | Mean score | Matches with 5000 | Distance error near obstacles |
|---|---|---|---|---|
| 0 | 20,000 | 316 | 0 | 55 pixels |
| 1 | 20,962 | 1063 | 0 | 36 pixels |
| 2 | 26,969 | 2359 | 4 | 16 pixels |
| 3 | 37,442 | 3322 | 6 | 12 pixels |
| 4 | 50,349 | 4576 | 9 | 9 pixels |
| 5 | 65,349 | 4576 | 9 | 7 pixels |
| 6 | 80,349 | **5000** | **10** | 6 pixels |

The first rounds add few scenes, because eyes + brain still crash early. As the eyes get better, the matches get longer, each round brings more scenes, and the remaining mistakes go away.

![true vs estimated distance](results/02_estimates.png)

After round 6, the distance the eyes estimate (orange) follows the true one (black) for every obstacle in a match.

---

## 6. The labels came for free

To train "cat or dog" on photos, someone has to label thousands of pictures by hand. Here, every image comes with its labels for free: the game knows the exact distance, size and height of every obstacle. That is also why DAgger was easy: any new scene could be labeled at once.

Self-driving cars and robots use the same idea. A lot of their training happens in simulators, where every pixel comes with its answer. The hard part is the same as here: the real world shows them scenes the simulator never did.

---

## Exercises

1. In `vision.py`, set `FRAME_GAP = 1` and train again. Can the eyes still estimate the speed?
2. Remove the CoordConv channel (the `columns` buffer) from the eyes. How does the distance error change?
3. Run `train_eyes.py --rounds 0` (no DAgger) and then `--rounds 2`. Compare with the table above.
4. Apply the DAgger idea to `pixels_to_keys.py`: let it play, label the scenes it visits with the teacher's keys, retrain. Does it reach 5000 every time?
5. Use the lesson 5 student as the brain instead of lesson 4's agent. It reads the same 6 numbers.

---

## How does this relate to LLMs?

- **Models that see images** work in a similar way: a vision network turns the image into lists of numbers, and the language model reads those numbers as if they were words. That's eyes + brain again, with a much bigger brain.
- **Scenes never seen in training** are a problem for LLMs too. When a question is very different from anything in its training data, a model can answer with confidence and still be wrong, just like the eyes that "saw" a cactus far away when it was right in front of the dino.
- **Labels are the expensive part.** Here the game gave them for free. For LLMs, good examples written by people are costly, which is one reason they are pre-trained on raw text first: the next word is a label that comes for free.

**Previous lesson:** [imitation](../05-imitation/README.md) · **Study home:** [all lessons](../../README.md)
