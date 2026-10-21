**English** · [Português](README.pt-BR.md) · [Español](README.es.md)

# Lesson 2 — The environment: the game the AI will play

> **Goal:** understand the AI's "world" — what it sees, what it can do
> and how we know whether it did well.

```bash
python lessons/02-the-environment/play.py
```

Play a few rounds. In the corner of the screen, under "What the AI sees:", you'll find the **6 numbers the AI will receive**. Notice how they change when a cactus gets close.

---

## What is pygame?

It's a **Python** library for making 2D games. It opens a window, draws rectangles and images, reads the keyboard and controls time (60 frames per second). Install it with `pip install pygame`.

Our whole game is in [`dino/game.py`](../../dino/game.py) — a little over 200 lines, all commented.

---

## The game loop

Every game (and every AI environment) is a loop that repeats:

```
┌──────────────────────────────────────────────┐
│ 1. OBSERVE    the current state → sensors()  │
│ 2. DECIDE     an action       → human or AI  │
│ 3. ACT        advance one frame → step()     │
│ 4. EVALUATE   crashed? +1 point?             │
└──────────────── repeat ──────────────────────┘
```

This same loop — **observation → action → reward** — is the basis of reinforcement learning, the technique also used in the post-training of LLMs.

---

## The three ingredients the AI needs

### 1. Observation: what it "sees"

The AI doesn't see the screen. It receives 6 numbers, calculated in `Game.sensors()`:

| # | Sensor | Meaning | Range |
|---|---|---|---|
| 1 | distance | How far away the next obstacle is | 0 (right next to it) to 1 (edge of the screen) |
| 2 | width | Width of the obstacle | ~0.3 to ~0.9 |
| 3 | height | Height of the obstacle | ~0.4 to ~0.9 |
| 4 | elevation | Distance from the obstacle to the ground | 0 = cactus; > 0 = bird |
| 5 | dino height | Height of the dino (is it in the air?) | 0 = on the ground |
| 6 | speed | Current game speed | 0.4 at the start, 1 at most |

All the numbers are **normalized** (close to 0 to 1). Neural networks learn poorly when one number is 800 and another is 0.02 — that was the same reason we used `normalize()` in Lesson 1.

> 💡 **Choosing good inputs is half the work.** If we forgot the speed, the AI would have no way of knowing it needs to jump earlier when the game speeds up.

### 2. Action: what it can do

Three options: `NOTHING`, `JUMP` or `DUCK`. The network will have **3 outputs**, and the biggest one wins.

### 3. Reward (or *fitness*): how we know it did well

**+1 point per frame survived.** Simple, and it doesn't say *how* to play — only what a good result is. The AI will have to figure out the rest.

> ⚠️ The reward defines what the AI learns. If we gave points per jump, it would learn to jump nonstop. This problem — the AI optimizing the score instead of the real goal — has a name: *reward hacking*. It also happens with LLMs.

---

## Why many dinos at the same time?

`Game` accepts `n_dinos`. They all face **the same obstacles** in the same match, which makes comparing them fair. This will be essential in the next lesson, where 50 dinos compete to pass on their "genes".

## Exercises

1. How many points can you get? (Spoiler: the AI will reach 5000.)
2. When the first bird shows up, what is the value of the "elevation" sensor? And for each of the 3 kinds of bird (`bird_low`, `bird_mid`, `bird_high`)?
3. In `dino/game.py`, change `JUMP_FORCE` to `-12`. Is the game still possible?

**Next lesson:** [neuroevolution — 50 dinos evolving](../03-neuroevolution/README.md)

---

## Changing the pictures

The dino, the cacti and the bird are drawn in [`dino/sprites.py`](../../dino/sprites.py) as **text**: `#` is a painted pixel, `w` is white and `.` is transparent. Change the drawings and create any character you want.

This is **visual only**. For the game, each thing is still a rectangle (`rect` in `game.py`), and the collision and the sensors use those rectangles. The AI never sees the pictures, only the 6 numbers. That's why changing the drawing changes nothing in the learning.
