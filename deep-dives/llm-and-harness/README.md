**English** · [Português](README.pt-BR.md) · [Español](README.es.md)

# Deep dive: today's LLMs and the harness around them

> Lesson 7's model writes one character at a time. How is that related to the LLMs people use today,
> which read files, run code and fix their own mistakes? Mostly, it is the same loop, a lot bigger,
> with another program wrapped around it: the **harness**. Here we build a toy one, and teach
> lesson 7's model to use it.

The charts come from two scripts. Both use lesson 7's model, so train lesson 7 first if you haven't:

```bash
python deep-dives/llm-and-harness/toy_agent.py      # a few seconds
```

```bash
python deep-dives/llm-and-harness/teach_tools.py    # about 5 minutes on a Mac mini M4
```

This page builds on [lesson 7](../../lessons/07-language-model/README.md), and reuses ideas from lessons [4](../../lessons/04-reinforcement/README.md) (reinforcement) and [5](../../lessons/05-imitation/README.md) (imitation).

---

## 1. The loop you already wrote

This is the heart of lesson 7 (`tiny_lm.py`, `generate`):

```python
for _ in range(n):
    ids.append(draw(self.next_probabilities(ids, temperature)))
```

The model can do only one thing: given the text so far, give the chance of each possible next token. Today's LLMs do exactly the same, in the same loop. They have no separate mode that "thinks up the whole answer". The whole answer is this loop, run thousands of times.

What changes is the size:

| | Lesson 7 | A large LLM today |
|---|---|---|
| Token | 1 character (53 of them) | pieces of words (50,000 to 200,000 of them) |
| Context | 64 characters | hundreds of thousands to about a million tokens |
| Size | 2 layers, vectors of 64 numbers, ~110,000 weights | dozens of layers, vectors of thousands of numbers, hundreds of billions of weights (often a "mixture of experts": only part of the network works on each token) |
| Training text | the study's READMEs (~47,000 characters) | trillions of tokens |
| Training | predict the next character | three stages, below |

---

## 2. How a large LLM is trained: three stages you have already seen

1. **Pre-training.** What lesson 7 did: predict the next token, on a huge amount of text. The result is a very well-read autocomplete. It continues text; it doesn't yet follow instructions.
2. **Supervised fine-tuning (SFT).** This is [lesson 5](../../lessons/05-imitation/README.md), imitation. The model is trained on examples of good conversations, and copies them. This is also where it learns formats, such as how to ask for a tool. Section 5 does exactly this to lesson 7's model.
3. **Reinforcement learning.** This is [lesson 4](../../lessons/04-reinforcement/README.md), PPO. The model writes, something gives a score (people, another model acting as judge, or a test that passes or fails), and the model is pushed toward what scored well. Models that "think before answering" came largely from here: with reinforcement, they learned that writing out reasoning before the answer earns more reward.

Remember lesson 4's plateau of "always jump"? LLMs also find shortcuts that earn reward without doing the job (*reward hacking*). Designing rewards that can't be gamed is a large part of the work.

---

## 3. The harness: the loop around the model

Even after all that training, the model is still a function from text to text. It can't read a file, run a command, or remember anything between calls. The program that does those things is the **harness**. Claude Code is a harness; so are the chat apps.

![Three loops](charts/00_three_loops.png)

1. **The token loop**, inside the model: lesson 7's `generate`.
2. **The agent loop**, in the harness. The model writes some text. If it asks for a tool, the harness runs the tool, pastes the result into the conversation, and asks the model again:

   ```python
   transcript = system_prompt + tool_descriptions + question
   while True:
       reply = model(transcript)              # the token loop, inside
       transcript += reply
       if reply asks for a tool:
           transcript += run_tool(reply)      # the HARNESS runs it, not the model
       else:
           break                              # the final answer
   ```

3. **The session loop**, also in the harness. The context has a limit (64 characters for lesson 7's model), so when the conversation gets too long, the harness summarizes the old part. Memory between sessions is just a text file that the harness pastes in at the start.

This deep dive was itself written by an AI model working inside a harness. To read `tiny_lm.py`, the model wrote a request like "run `grep` on this file"; the harness ran it and pasted the output back. The model never touched the disk. It only wrote text in an agreed format.

---

## 4. A toy agent

`harness.py` has the whole thing in a few dozen lines. The conversation is one piece of text, the **transcript**, and each line starts with who wrote it:

```
q: what is 37+48?          <- the user
call calc: 37+48           <- the model asks for a tool
result: 85                 <- the harness ran the tool and pasted the result
answer: 85                 <- the model answers, and the loop stops
```

- **The tools** are ordinary Python functions: `calc` adds or subtracts two numbers, `letters` counts the letters of a word.
- **The model** is anything that takes the transcript and returns one line. The fake model is a script of a few `if`s. Like a real LLM, it looks at nothing but the transcript.
- **The harness** is the loop: ask the model, look for `call <tool>: <input>`, run the tool, paste `result: ...`, ask again, until a line starts with `answer:`.

![The agent loop, step by step](charts/01_agent_loop.gif)

Now the interesting part: **the same harness**, with lesson 7's model in place of the fake one:

```
q: what is 37+48?
the step and the score the student the s
result: error: write call or answer
the student the script the same and the
result: error: write call or answer
the student the script the same and the
...                                          (gives up after 4 turns)
```

Three things to notice:

- **The harness is the same, and the result is completely different.** For the loop to work, the model has to know the `call` / `answer` format. Lesson 7's model only ever read READMEs, so it writes "like a README".
- **It can't even see the question for long.** With a context of 64 characters, it sees only the last 64. A real harness puts the tool descriptions at the start of the transcript, and a real LLM can read them there. Ours would never see them.
- **Turns 2, 3 and 4 are identical.** The model has no memory: the same text in gives the same text out (here the harness picks the most likely character every time). All it "knows" about the conversation is the text the harness builds.

---

## 5. Teaching the model the format

So let's teach it, the way [lesson 5](../../lessons/05-imitation/README.md) taught the dino: **imitation**. The teacher is the fake model; its transcripts are the examples. `teach_tools.py` starts three students from lesson 7's weights, and trains each one with lesson 7's own loop:

| Student | Examples it learns from | Steps |
|---|---|---|
| 1. answer directly | `q: what is 37+48?` → `answer: 85`, no tools | 3,000 |
| 2. call tools | the fake model's transcripts: two-digit numbers, words from the study | 3,000 |
| 3. call tools, varied data | the same format, but numbers of 1 to 3 digits and made-up words like `qzwkt` | 25,000 |

Then all four models (lesson 7 as is, and the three students) run in the same harness, on questions **none of them saw in training**: two-digit arithmetic, real words kept out of training, and four-digit numbers, longer than anything in training.

![Right answers by model](charts/02_teach_tools.png)

What happened, with real transcripts (all in `charts/samples.md`):

**Student 1 can't do arithmetic: 0%.** It answers `46-30` with `-10`. A network this small doesn't learn to add from a few thousand examples. Large LLMs are much better at mental arithmetic, but they still make mistakes on long numbers, and that is one reason they are given a calculator or a code tool. (It got 41% on counting letters. Always guessing the most common length would get 19%, so it learned something about word length, but not to count.)

**Student 2 calls the calculator perfectly: 100%.** The model doesn't add; it **copies** `46-30` into `call calc:`, and Python adds. But look at the words and the big numbers:

```
q: how many letters in blows?        q: what is 6950+1180?
call letters: block                  call calc: 69+18
result: 5                            result: 87
answer: 5                            answer: 87
```

It learned the **shape** of its examples, not the rule "copy the input". Every number it saw had two digits, so it writes two digits. It had seen the study's words thousands of times, so it writes one it knows (`block`) instead of the new one (`blows`). Its 11% on letters is luck: `block` and `blows` both have 5 letters.

**Student 3 learned to copy: 90% on real words it never saw.** Made-up words can't be memorized, so the only way to get them right is to really copy, letter by letter. But it took a long time:

![Student 3 learning to copy](charts/03_copy_emerges.png)

Copying two-digit numbers came within 2,500 steps. Copying words stayed at 0% for 20,000 steps, then jumped to over 80% almost at once. Researchers have seen the same kind of sudden jump in real transformers when they learn to copy from earlier in the text (the "induction heads" studied by Anthropic in 2022).

**Nobody handles four-digit numbers: 0%.** Student 3 writes `695+118` for `6950+1180`. It saw numbers of up to 3 digits, so it writes up to 3 digits. The tool computes `813` correctly, for the wrong question, and the model reports it with confidence. A tool is only as good as the request the model writes.

What this says about real LLMs:

- **Calling tools is a learned skill**, taught by fine-tuning and reinforcement, with huge and very varied sets of examples, so the model learns the rule rather than the shape of a few examples.
- **Tools beat memorizing**: 0% vs 100% on the same arithmetic, with the same tiny network.
- **The failures look like real ones**: a wrong argument sent to a correct tool, then reported confidently. Good harnesses help by checking (running the tests, showing errors to the model, asking a person before risky actions).
- Counting letters is famously hard for large LLMs, for a different reason: they read pieces of words, not letters, so they don't "see" the letters. Ours reads one character at a time, which is why student 1 could at least get close.

---

## 6. Why the harness matters as much as the model

The study has seen this before, with the dino:

| In the study | In an LLM agent |
|---|---|
| The policy network of [lesson 4](../../lessons/04-reinforcement/README.md): 6 numbers in, an action out | the model: text in, text out |
| `game.py` and the main loop: build the observation, apply the action, return the result | the harness |
| [Lesson 6](../../lessons/06-vision/README.md): CNN "eyes" turn the screen into the 6 numbers the brain needs | the harness turning files, web pages and command output into text for the model |
| [Lesson 8](../../lessons/08-hard-mode/README.md): the network alone scores about 900; with a planner beside it, 5000 | a model alone vs a model that can run code, read errors and try again |

With the **same model**, a good harness (useful tools, a well-built transcript, ways to check its own work) makes a big difference in what gets done. A bad one wastes a good model. The model is the engine; the harness is the rest of the car.

---

## What we simplified

- Real tool calls are structured (usually JSON with a tool name and named arguments), and the harness checks them before running anything. Ours is one line of text.
- Real fine-tuning usually learns only from the model's own lines. Ours also learned to predict the questions and the results, which costs nothing here.
- A real harness sends the list of tools and their descriptions in every request, so the model can use tools it has never seen in training. Our model has a context of 64 characters, so the format has to be baked in by training.

---

## Summary

| Question | Short answer |
|---|---|
| Is a large LLM different from lesson 7's model? | The same loop and the same kind of network, millions of times bigger, trained in three stages: pre-training (lesson 7), imitation (lesson 5) and reinforcement (lesson 4). |
| What is a harness? | The program around the model: it builds the transcript, runs the tools the model asks for, pastes the results back, and manages the context and memory. |
| Does the model run the tools? | No. It only writes text asking for them. The harness runs them. |
| Can lesson 7's model use tools? | Not as is. After fine-tuning on examples, yes, but only as generally as the examples were varied. |

**Back to:** [the study home](../../README.md)
