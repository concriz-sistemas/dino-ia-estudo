**English** · [Português](README.pt-BR.md) · [Español](README.es.md)

# Lesson 7 — A tiny language model

> **Goal:** build the smallest possible version of an LLM and look inside it.
> It reads text and learns to predict the next character, with the same design as the big ones
> (a *transformer*), just tiny. It trains on the READMEs of this study, in about 30 seconds.

![the model writing](results/generation.gif)

*The model writes one character at a time. Below, the chance it gives to the 8 most likely next characters; the orange one is the character drawn.*

Run them one at a time, in this order.

1. Train the model (about 30 seconds on a recent Mac, longer on a computer without a graphics card that PyTorch can use):

```bash
python lessons/07-language-model/train.py
```

2. Make it write:

```bash
python lessons/07-language-model/generate.py
```

3. Watch how it chooses each character:

```bash
python lessons/07-language-model/generate.py --explain
```

4. Look inside it: embeddings, cosine similarity and attention:

```bash
python lessons/07-language-model/inside.py
```

The trained model is already in the repository, so steps 2 to 4 work without training.

---

## 1. The whole job: guess the next character

A language model does one thing: given the text so far, it gives a **chance to every possible next piece of text**. To write, it draws one piece according to those chances, adds it to the text and asks again.

Here each piece (each **token**) is a single character: 53 different ones, letters, digits, punctuation, space and line break. Real LLMs use pieces of words instead ("dino", "saur", "jump", "ing"), with vocabularies of tens of thousands to a couple of hundred thousand tokens.

The training text is the English README of the study home and of lessons 1 to 6, cleaned up (no tables, no code, no links), in lowercase: **46,605 characters**. It's frozen in [`data/study.txt`](data/study.txt). The first 90% is used for training; the last 10% is **unseen text**, to check how the model does on text it never trained on.

The **loss** is the same idea as in lesson 1: how surprised the model is by the real next character. A model that guesses at random has a loss of 3.97. Lower is better.

---

## 2. Model 1: counting pairs

The simplest possible model, the **bigram**, just counts which character follows which in the training text: after "q", almost always "u"; after a space, often "t". No network, no training.

Its loss on unseen text is **2.44**, much better than random. But it only ever looks at the one previous character, so it writes like this:

```
the dino ond, mpenche me fond ps, --oen. one min torith, 3v/7;qoomat5'sschinletchtsth
```

---

## 3. Model 2: a tiny transformer

[`tiny_lm.py`](tiny_lm.py) builds the same kind of network as an LLM, just tiny:

1. **Embeddings.** Each character becomes a vector of 64 numbers (its *embedding*), plus a vector for its position (1st, 2nd, 3rd... character). These numbers are learned.
2. **Attention.** Each position looks back at the previous characters (up to 64) and decides which ones matter for what comes next. Positions can never look at the future: that would be peeking at the answer.
3. **A small network** (like lesson 1's) processes what attention gathered.
4. Steps 2 and 3 form one **block**; the model stacks 2 of them. At the end, a score for each of the 53 characters, turned into chances.

In total, **111,029 weights**, trained by gradient descent like every network in this study.

![loss](results/01_loss.png)

| Step | Loss on training text | Loss on unseen text | What it writes after "the dino " |
|---|---|---|---|
| 0 | 4.15 | 4.16 | `o'9,sqj'/9+r%-62)'m=3%nw(-)zet(o60i` |
| 250 | 2.25 | 2.28 | `on the en mote the ands weiget. the min tory` |
| 500 | 1.70 | 1.87 | `only step. ritep (the screace work + in to it` |
| 1000 | 1.38 | 1.68 | `only much mutes for 300 muches. the minutions` |
| 2000 | 1.14 | 1.64 | `only matches teaches path. in the explore,` |
| 4000 | 0.95 | 1.72 | `only model steps to find -1 expected value` |

First it learns which characters are common, then short words ("the", "on"), then longer ones ("network", "matches"), then something that sounds like English but means nothing. [`results/samples.md`](results/samples.md) has the full samples.

### Learning or memorizing?

Look at the two curves. The loss on the **training text** keeps going down until the end. The loss on **unseen text** stops improving around step 1400 (best: **1.64**) and then slowly gets worse. From there on, the model is **memorizing** this particular text instead of learning English: it's *overfitting*, the same trap as in lesson 1 with 20 examples.

`train.py` keeps the model from its best moment on unseen text. A *dropout* of 10% (randomly switching off signals during training) also makes memorizing harder.

The cure for real is **much more text**. 47 thousand characters fit in a few pages; LLMs train on trillions of tokens, so they can't memorize all of it and have to learn the patterns.


### Measuring it: is the text new, or copied?

The loss curves say *when* memorizing starts. To see it in the text itself, [`memorization.py`](memorization.py) works like a plagiarism checker: for every character the model writes, it finds the **longest piece around it that appears, exactly, in the training text**. Short pieces ("the ", "ing ") show up in any English text and mean nothing. A copied piece of **20 or more characters** is a sign of memorization.

As a fair reference, the same check runs on the unseen 10% of the text: real English, written by a person, that the model never saw. Only **2.5%** of it sits inside copied pieces of 20+ characters, just by chance (the READMEs repeat some phrases).

The script trains two models for 12,000 steps, three times longer than `train.py`: one with dropout, one without.

```bash
python lessons/07-language-model/memorization.py
```

![memorization](results/05_memorization.png)

| Step | With dropout: loss on unseen text | With dropout: copied | Without dropout: loss on unseen text | Without dropout: copied |
|---|---|---|---|---|
| 1,000 | 1.70 | 0% | 1.75 | 0% |
| 4,000 | 1.71 | 2.5% | 3.11 | 22% |
| 9,000 | 1.82 | 7% | 4.35 | **55%** |
| 12,000 | 1.83 | 10% | 4.68 | 50% |

Without dropout, the model gets worse on unseen text while it copies more and more: by step 9,000, more than half of what it writes is pasted from the READMEs. With dropout it copies much less, but it slowly drifts the same way. That's why `train.py` stops at the best moment.

Here is the text itself, colored character by character:

![copied text](results/06_copied_text.png)

- **Top**, with dropout at its best: almost all green and yellow. Familiar words and short phrases, combined in new ways.
- **Middle**, without dropout at step 9,000: long red stretches, such as "lesson 4's 04decisions.png: it's practically the same", copied word for word from lesson 5's README.
- **Bottom**, the reference: real English the model never saw. Mostly green, with yellow for common phrases.

So yes, you can check whether a model is repeating what it read, at least for text you have: compare what it writes with what it trained on. Researchers do the same with big LLMs and find both. They can repeat some texts word for word, especially ones that appear many times in their training data, but most of what they write is new combinations.

---

## 4. Writing: chance and temperature

`generate.py --explain` shows each choice. After "the dino sco", the model gives 97% to "r": it has learned the word "score". After a space, the chances spread out, because many words can come next:

```
                       text so far   the 5 most likely next characters         drawn
                       'the dino '   a  15%  s  14%  t   8%  p   7%  i   7%   -> 's'
                      'the dino s'   t  42%  a  22%  c  17%  e   6%  h   4%   -> 'c'
                     'the dino sc'   o  63%  r  30%  a   3%  e   2%  h   1%   -> 'o'
                    'the dino sco'   r  97%  n   1%  v   1%  m   0%  s   0%   -> 'r'
```

The **temperature** changes how the chances are used. Low temperature makes the likely characters even more likely; high temperature gives the unlikely ones a better shot. Try it:

```bash
python lessons/07-language-model/generate.py --compare
```

| Temperature | What it writes after "the dino " |
|---|---|
| 0.3 | `steps the same a good step of the example step over action 50 and 2000 and the step and the dino and the and the same the same and the` |
| 0.8 | `only matches teacher's when example find of the 292020000 15 is in lotic still it learning:` |
| 1.5 | `only much. rule for 300,5-4zen. the minutoring, 24bunce, at5k scompletes fipt)? helthile togree varixelys` |

At 0.3 it gets stuck repeating the safest words. At 1.5 it invents words. Chat assistants usually write somewhere in the middle.

---

## 5. Embeddings and cosine similarity, for real

In lesson 1, cosine similarity was only an analogy. Here it's real: every character has a vector of 64 numbers, and the cosine similarity says how much two of them point the same way.

![embeddings map](results/03_embeddings.png)

*Each character's 64 numbers squeezed to 2, so they fit on a map.*

The **digits** form their own group, away from the letters. Nobody told the model that "3" and "4" are both numbers. It noticed that they show up in the same kinds of places (next to other digits, before "k" in "500k", after "lesson "), so it gave them vectors that point the same way.

![cosine similarity](results/02_cosine.png)

| Group | Average cosine similarity with each other | With the rest |
|---|---|---|
| digits | +0.18 | −0.02 |
| vowels | +0.09 | −0.04 |
| consonants | +0.03 | −0.02 |
| punctuation and space | +0.02 | −0.02 |

The closest characters to "3" are "4" (+0.45), "7" (+0.38) and "0" (+0.24). The closest to "." include ";" and "?", other punctuation marks.

These signals are weak for most characters: a single letter carries little meaning on its own. In LLMs, whose tokens are words and pieces of words, the same mechanism puts "dog" and "puppy" close together, because they are used in the same places. It's the same idea, with more meaning to capture.

---

## 6. Attention: what each position looks at

![attention](results/04_attention.png)

Each panel is one of the 4 attention heads of the last layer, reading "the dino jumps over the cactus.". Each row is one position, and the dark cells are the earlier characters it looks at when guessing what comes next. Nothing appears above the diagonal: the future is hidden.

In this sentence, some heads mostly look back at the beginning of the current word (in head 1, the "u", "m" and "p" of "jumps" look at the "j"); others look one or two characters back. Nobody programmed these rules; they came out of training, because they help guess the next character.

---

## 7. This model vs a real LLM

| | This lesson | A large LLM |
|---|---|---|
| Token | 1 character | a piece of a word |
| Vocabulary | 53 tokens | tens of thousands to ~200,000 tokens |
| Context (how far back it can look) | 64 characters | thousands to millions of tokens |
| Weights | 111,029 | billions to hundreds of billions |
| Training text | 46,605 characters | trillions of tokens |
| Training | about 30 seconds on a laptop | weeks or months on thousands of GPUs |

The design is the same: embeddings, attention blocks, a score for every possible next token, gradient descent on the loss. The difference is scale, and with scale comes the ability to write things that make sense.

And this lesson is only the first stage of an LLM, **pre-training**. After it, real models are fine-tuned by imitating good answers ([lesson 5](../05-imitation/README.md)) and by reinforcement learning ([lesson 4](../04-reinforcement/README.md)).

---

## Exercises

1. Train it on any other text file, like a book in plain text: `train.py --text book.txt`. Does it learn faster with more text? Does it still memorize?
2. In `tiny_lm.py`, set `DROPOUT = 0` and train again. Where does the loss on unseen text stop improving now?
3. Set `CONTEXT = 8`. What happens to the loss and to the words it writes?
4. Set `LAYERS = 1`. Is one block enough?
5. Run `generate.py --prompt "the gradient "` at temperatures 0.3 and 1.5 (use `--temperature`). Which one sounds more like the study?

**Previous lesson:** [learning to see](../06-vision/README.md) · **Study home:** [all lessons](../../README.md)
