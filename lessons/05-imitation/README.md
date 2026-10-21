**English** · [Português](README.pt-BR.md) · [Español](README.es.md)

# Lesson 5 — Imitation: you teach the dino by playing

> **Goal:** train a network by **copying a teacher** instead of chasing a reward.
> The teacher can be you (you play and every frame is recorded) or the agent from lesson 4.
> It's supervised learning, like lesson 1, with the teacher's keys as the answer key.

![the student playing](results/student.gif)

*The student playing on its own. On the right, the chance it gives to each action, what the teacher would press in the same situation, and how often they agree.*

Run them one at a time, in this order.

1. Train the student on the example demonstrations (10 matches played by lesson 4's agent, already in the repository):

```bash
python lessons/05-imitation/train.py
```

2. Watch the student play. R starts a new match:

```bash
python lessons/05-imitation/watch.py
```

3. Run the four experiments: is the student as good as its teacher?

```bash
python lessons/05-imitation/experiments.py
```

4. Now teach it yourself. Play a few matches: SPACE or ↑ jumps, ↓ ducks, R starts the next match and ESC saves and quits:

```bash
python lessons/05-imitation/record.py
```

The terminal then prints the command to train a student on your own demonstrations.

---

## 1. A teacher instead of a reward

| | Lesson 1 — gradient | Lesson 4 — reinforcement | Lesson 5 — imitation |
|---|---|---|---|
| What the network gets | the right answer for each example | a reward at each step | what the teacher pressed at each frame |
| Who knows the answer? | the code (the true rule) | nobody | the teacher |
| How the weights change | gradient | gradient | gradient |
| Cost to learn to play | — | 2 million steps, about 80 s | 1 recorded match, about 2 s |

Imitation is supervised learning again: the same loop as lesson 1 (forward, loss, gradient, step). The difference is where the answer key comes from. In lesson 1 a rule wrote it; here a player writes it, just by playing.

---

## 2. The demonstrations

[`record.py`](record.py) writes one row per game frame: the 6 sensors the AI sees, and the key that was pressed in that frame. The example file, [`results/demos/teacher_02000k.csv`](results/demos/teacher_02000k.csv), has 10 matches played by lesson 4's final agent: 50,000 rows.

To record the teacher yourself (a different teacher, a delay, more matches), see the options at the top of `record.py`. For example:

```bash
python lessons/05-imitation/record.py --teacher
```

![actions in the demonstrations](results/teacher_02000k/01_actions.png)

Here the teacher pressed "jump" in 40% of the frames, "nothing" in 33% and "duck" in 28%. That is the **accuracy trap** from lesson 1 again: a student that always answered "jump" would already be 40% accurate, and would crash at the first cactus. With your own demonstrations the trap is usually much bigger, because people spend most of the time pressing nothing.

---

## 3. Training the student

The student is a small network written with numpy only, like lesson 1: 6 sensors → 16 hidden neurons → 3 actions. With 3 possible answers instead of 2, the output uses a *softmax*, which turns three scores into three chances that add up to 100%.

The last match of the demonstrations is kept aside: the student never trains on it, so it shows how the student does on situations it has never seen.

![training](results/teacher_02000k/02_training.png)

With the example demonstrations:

- the student presses the same key as the teacher in **93.7%** of the training frames, and **93.5%** of the frames of the match kept aside;
- playing on its own, it scores **5000 in all 10 new matches**.

The whole student is a file with 163 numbers, `results/models/student_teacher_02000k.npz`. Like lesson 4's agent, that file *is* the learning.

---

## 4. 93% agreement, perfect play

If the student disagrees with the teacher in 6% of the frames, how does it never crash? Because most disagreements happen when the key barely matters:

- **56%** of the disagreements happen while the dino is in the air, where pressing "jump" again does nothing;
- another **24%** are "nothing" versus "duck" while the obstacle is still far away.

Accuracy measures how well the student **copies**; the score measures how well it **plays**. They are not the same thing, and the goal is the second one.

---

## 5. What the student learned

![decisions](results/teacher_02000k/03_decisions.png)

Compare it with lesson 4's [`04_decisions.png`](../04-reinforcement/results/04_decisions.png): it's practically the same picture. The student learned the teacher's slanted line for the cactus and even its odd strategy for the bird at head height (stay ducked while it's far, jump over it when it's close). The student never saw a reward; it only copied, and it copied everything, including the quirks.

---

## 6. Is the student as good as its teacher?

[`experiments.py`](experiments.py) records four different teachers, trains one student on each, and lets teacher and student play the same 10 new matches. The "late" teachers are lesson 4's final agent pressing each key a few frames after seeing the situation, like a person's reaction time.

![experiments](results/experiments.png)

| Teacher | Examples | Teacher's mean score | Student's mean score |
|---|---|---|---|
| Strong, only 1 match recorded | 5,000 | 5000 | **5000** |
| Weak (lesson 4 after 500k steps) | 1,803 | 151 | 198 |
| Strong, 8 frames late | 37,313 | 2870 | **5000** |
| Strong, 12 frames late | 3,757 | 752 | 172 |

- **Imitation is cheap.** One recorded match and 2 seconds of training gave a perfect player. Lesson 4 needed 2 million steps to discover the same thing alone.
- **A weak teacher makes a weak student.** The student copies whatever it sees. It cannot know that the teacher is bad, because it never sees a score.
- **The student can beat a sloppy teacher.** The teacher 8 frames late sometimes jumps too late and crashes; the student learns its *average* behavior, which is good, and loses the occasional slip. It scored 5000 where the teacher averaged 2870.
- **But not a bad one.** At 12 frames late, the lateness is the behavior, not a slip. The student copies it and does even worse.

---

## 7. Your turn: teach it yourself

Record a few matches with `record.py` and train a student on them with the command the script prints. Then compare your student with the example one. Some things that usually happen with human demonstrations:

- **The trap gets bigger.** Most of your frames are "nothing", so the "always nothing" student looks very accurate and plays terribly. Look at `01_actions.png` and at the dashed line in `02_training.png`.
- **Your reaction time is in the data.** Like the "late" teachers above, the student learns when *you* pressed, not when you should have pressed.
- **More matches help**, especially matches where you reached the fast part of the game with birds. The student can only copy situations that are in the demonstrations.

---

## Exercises

1. Train only on the first match of the example demonstrations (`train.py --matches 1`). Does the student still score 5000?
2. In `experiments.py`, try delays of 4, 10 and 16 frames. Where is the line between "the student cleans up the teacher's mistakes" and "the student copies them"?
3. Record your own demonstrations and train a student. Is it better or worse than you? Watch it with `watch.py --student` and the path the training prints.
4. Mix your demonstrations with the teacher's: `train.py --demos` accepts several files. What happens to the student?

---

## How does this relate to LLMs?

LLMs are trained in the same order as this study:

1. **Pre-training** is imitation on a huge scale: the model learns to predict the next word of trillions of words written by people. It copies human text, good and bad.
2. **Supervised fine-tuning** is this lesson: people write examples of good answers, and the model learns to imitate them. It's cheap and fast, like the 1-match student, but the model can only be as good as the examples, like the weak-teacher student.
3. **Reinforcement learning** (lesson 4) comes last: the model tries, gets a score, and is adjusted to score higher. That is how it can go beyond its examples.

Imitating first and then using reinforcement combines the strengths of both: imitation gives a good starting point cheaply, and reinforcement improves it where the examples fall short.

**Previous lesson:** [reinforcement learning](../04-reinforcement/README.md) · **Study home:** [all lessons](../../README.md)
