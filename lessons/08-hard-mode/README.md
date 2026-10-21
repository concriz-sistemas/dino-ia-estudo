**English** · [Português](README.pt-BR.md) · [Español](README.es.md)

# Lesson 8 — Hard mode: when the world changes

> **Goal:** change the game and see what happens to everything that was trained on the old one.
> The new game has bird formations where jumping is fatal. Who can still play, who can learn it,
> and what does it take?

![before and after](results/before_after.gif)

*The same match twice. Top: lesson 4's agent jumps into a staircase of birds and crashes. Bottom: lesson 8's student ducks under it and keeps going.*

The original game (`dino/game.py`) is not changed: lessons 1-7 still use it. The hard game is a separate version, in [`hard_game.py`](hard_game.py).

Play it yourself:

```bash
python lessons/08-hard-mode/hard_game.py
```

Then run the experiments one at a time, in this order. The trained student is already in the repository, so you can skip step 3 and still run the others.

1. Who can still play? Every agent of the study, on both games (about 4 minutes):

```bash
python lessons/08-hard-mode/test_agents.py --student
```

2. Can reinforcement learning learn the new game by itself? Four attempts (about 10 minutes):

```bash
python lessons/08-hard-mode/rl_attempts.py
```

3. A student that imitates a coach, with DAgger (about 20 minutes):

```bash
python lessons/08-hard-mode/train_student.py
```

4. Watch lesson 4's agent and the student on the same match. R starts a new match:

```bash
python lessons/08-hard-mode/watch.py
```

---

## 1. The new formations

Once birds start to appear (around 700 points), sometimes a **formation** shows up instead of a single obstacle:

| Formation | What it is | Jump? | Duck? |
|---|---|---|---|
| **Tunnel** | a long row of birds at head height, too long to jump over at any speed | ❌ | ✅ the only way |
| **Staircase** | birds going down: high (75), middle (50), then three low ones (26) | ❌ a jump hits one | ✅ the only way: stand under the first two, duck under the rest |
| **Pair** | two birds at head height, close together | ✅ | ✅ both work |

The cacti and single birds of the original game still appear too. A cactus can only be jumped, and the bird at head height can be jumped or ducked.

None of this was guessed: [`hard_game.py`](hard_game.py) tries every sequence of keys against every formation, at every speed from 10.5 to 20, and checks that there is a way through, and which keys it needs:

```bash
python lessons/08-hard-mode/hard_game.py --verify
```

---

## 2. The world changed: the old agents fail

![who can play](results/01_who_can_play.png)

| Agent | Original game | Hard game | What it crashed into (10 hard matches) |
|---|---|---|---|
| Lesson 3: neuroevolution champion | 5000 | 930 | 5 tunnels, 4 staircases, 1 pair |
| Lesson 4: reinforcement learning | 5000 | 902 | 5 tunnels, 3 staircases, 2 pairs |
| Lesson 5: imitation student | 5000 | 946 | 6 tunnels, 4 staircases |

All three were perfect on the game they knew, and all three crash at the first formations, right after birds start to appear. They learned "the bird at head height: duck while it's far, jump when it's close", which was a fine rule in the old game. In a tunnel or a staircase, that jump is fatal.

Nothing is wrong with the agents. **The world changed** and they had never seen it.

---

## 3. Reinforcement learning alone doesn't learn the new game

The obvious fix is to train again on the new game. [`rl_attempts.py`](rl_attempts.py) tries four times, each attempt fixing what might have gone wrong in the previous one:

![reinforcement learning attempts](results/02_rl_attempts.png)

| Attempt | Training | Original game | Hard game |
|---|---|---|---|
| 1. From scratch | 3 million steps, only the hard game | 5000 | 951 |
| 2. Fine-tuning lesson 4 | 1 million more steps on the hard game | 5000 | 950 |
| 3. + curriculum | 2 million steps; in training, formations appear from the start, in half the obstacles | 5000 | 953 |
| 4. + more exploration | like 3, with an exploration bonus 3× bigger, 3 million steps | 4294 | 955 |

None of them gets through. Why?

![why reinforcement learning fails](results/03_why_rl_fails.png)

This is attempt 4 approaching a staircase. While the birds are far, it gives about 60% to "duck". When they get close, it switches to "jump" (almost 90%), its old habit for birds, jumps into the staircase and crashes. To get through, it would have to keep ducking through many decisions in a row, every one of them right, at exactly the moment its habit says "jump". Doing that by chance, often enough to notice that it pays off, practically never happens.

This is the **exploration problem** of reinforcement learning: it only learns from what it tries, and some solutions are too long or too unlikely to be found by trying at random. More exploration (attempt 4) didn't fix it, and it even made the agent worse on the original game.

---

## 4. A planner solves it, but doesn't learn

[`planner.py`](planner.py) doesn't learn anything. At each moment it looks at the obstacles on screen, imagines the next 80 frames with the game's own physics, and searches for a sequence of keys that doesn't crash.

| | Original game | Hard game |
|---|---|---|
| The planner | **5000** | **5000** |

It's perfect, but:

- it's **slow**: about 10 to 20 seconds per match on this Mac, while a trained network plays a match in under a second;
- it needs to **know the rules of the game** (gravity, jump force, speed) to imagine what will happen;
- it never gets better: every decision is calculated again from scratch.

---

## 5. A student that imitates a coach

So: a learner can't find the solution by itself, and the planner finds it but is slow. The fix is to let the planner **teach**: lesson 5's imitation, plus lesson 6's DAgger.

**The first idea didn't work.** With the planner alone as the teacher, the student learned almost nothing: 463 points on the hard game after round 0, and only 88 to 312 on the *original* game. Two more DAgger rounds didn't get it past 1161. You can repeat it:

```bash
python lessons/08-hard-mode/train_student.py --teacher planner --rounds 2
```

The planner pressed "jump" in only **2%** of its decisions: it prefers to do nothing, so it jumps once, at the **last possible moment**, and then waits. That's perfect for something that calculates, but a terrible example for a student. The jumps are so rare that the student barely learns them (lesson 5's accuracy trap), and a last-moment jump leaves no margin, so a student that is one decision late crashes (like the late teachers of lesson 5). **A good planner is not necessarily a good teacher.**

**What worked: a coach.** [`agents.py`](agents.py) builds one from two pieces:

- **lesson 4's agent** plays as usual, with its comfortable, early jumps;
- when one of the new formations is close, **the planner takes over**, preferring to duck early and stay ducked.

[`train_student.py`](train_student.py) then trains a student (a small network that sees the same 6 sensors) to press what the coach would press:

- **round 0:** the coach plays 6 hard matches, and every decision is recorded;
- **rounds 1-6 (DAgger):** the *student* plays; in every situation it gets into, the coach says what it would press; those examples are added, and a new student is trained on everything so far.

![student rounds](results/04_student_rounds.png)

| Round | Examples | Hard game (6 matches) | Original game (3 matches) |
|---|---|---|---|
| 0 | 15,000 | 3171 | 3231, 3331, 2624 |
| 1 | 20,676 | 1053 | 5000, 5000, 5000 |
| 2 | 25,764 | 1053 | 5000, 5000, 5000 |
| 3 | 32,773 | 3021 | 5000, 5000, 5000 |
| 4 | 42,773 | 3677 | 5000, 5000, 5000 |
| 5 | 52,773 | **5000** | 5000, 5000, 5000 |
| 6 | 62,773 | **5000** | 5000, 5000, 5000 |

And in the final test, with 10 new matches of each game: **5000 in all of them, on both games**, as good as the planner and fast like a network.

Two things to notice:

- **Learning the new thing cost the old one, for a while.** After round 0, the student was also worse on the original game (around 3,000 instead of 5000). The DAgger rounds brought it back as examples piled up. The examples always included the old obstacles too (the hard game still has cacti and single birds), so the student kept practicing them. Getting worse at old skills while learning new ones is called *catastrophic forgetting*, and mixing old and new examples is one of the usual cures.
- **Progress wasn't smooth.** Rounds 1 and 2 got stuck at 1053, crashing at the same staircase every time, until enough examples of that situation piled up.

---

## 6. Two ways through: what does each one choose?

The **pair** can be passed either way. [`pair_choices.py`](pair_choices.py) watches every pair each agent gets through and checks whether the dino was in the air (jumped) or ducking (ducked):

```bash
python lessons/08-hard-mode/pair_choices.py
```

In 12 hard matches:

| Agent | Pairs it got through | Jumped | Ducked |
|---|---|---|---|
| Lesson 4's agent | 9 | **9** | 0 |
| The coach | 85 | 0 | **85** |
| Lesson 8's student | 80 | 0 | **80** |

- **Lesson 4's agent always jumps.** That's its old habit for birds, and on a pair it happens to work. (It passes so few because it crashes at the first tunnel or staircase.)
- **The coach always ducks**, because its planner was told to prefer ducking near the new formations.
- **The student always ducks too.** It never had to choose: it copied the coach's *style*, not only its results.

Both answers are right. Which one an agent picks depends on where it learned it: from its own experience (lesson 4) or from a teacher with a preference (the coach). A student picks up its teacher's choices even when they were only a matter of taste.

---

## 7. How does this relate to LLMs?

The same problems, at a much bigger scale:

| In the dino | In an LLM |
|---|---|
| The old agents don't know the formations | the model doesn't know what happened after its training data was collected (its *knowledge cutoff*) |
| Lesson 6's eyes fail on a scene they never saw | the model does worse on topics, formats or languages that were rare in its training text |
| The student gets worse at the original game while learning the hard one | a model fine-tuned on new data can get worse at things it used to do well (*catastrophic forgetting*) |
| The agents keep an old habit ("jump at birds") that is now fatal | the model learns shortcuts from its data, like "every number in the question is used in the answer" — and adding one irrelevant sentence with a number can throw it off |
| Reinforcement learning can't find a long solution by trying at random | reinforcement learning on LLMs also needs a starting point that already does the task reasonably well |

And the fixes are also similar:

- **Look things up instead of remembering them.** The planner looks at the screen *now*. LLMs are given search tools and documents at the moment of the question, so they don't depend only on what they learned.
- **Show examples before rewarding.** The student needed a teacher to show it how to get through; reinforcement alone couldn't find it. LLMs are fine-tuned on good examples (lesson 5) before reinforcement learning (lesson 4).
- **Mix the old with the new.** The student recovered the original game because the examples always mixed both. When LLMs are fine-tuned, old data is mixed in for the same reason, and changes are kept small so the model doesn't drift too far from what it knew.
- **Train on the hard cases.** DAgger added exactly the situations the student got wrong. LLM builders collect examples of the cases where models fail, including questions with irrelevant details, and train on them.
- **Measure everything, before and after.** Every agent in this lesson is tested on both games. Improving one thing can make another worse, so LLMs are checked on large sets of tests before each new version is released.

**Previous lesson:** [a tiny language model](../07-language-model/README.md) · **Study home:** [all lessons](../../README.md)
