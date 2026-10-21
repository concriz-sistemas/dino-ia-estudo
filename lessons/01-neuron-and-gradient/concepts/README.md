**English** · [Português](README.pt-BR.md) · [Español](README.es.md)

# Deep dive: weight, bias, activation and loss

> A slower, more visual companion to [lesson 1](../README.md). Each idea comes with an everyday analogy,
> a chart, a small example worked out by hand, and where it lives in the code.

![turning the knobs](charts/knobs.gif)

*A neuron is a little machine with knobs. Here the knobs are turned one at a time: first the weight, then the bias. Watch what each one does to the curve.*

All the charts on this page use the simplest possible neuron: **one input**, the distance to the cactus (`x`, from 0 = touching it to 1 = far away), and one output, how much it "wants to jump". To draw the charts yourself:

```bash
python lessons/01-neuron-and-gradient/concepts/concepts.py
```

---

## 1. Weight: a volume knob

**Analogy.** Picture a mixing desk, with one volume knob per instrument. A **weight** is one of those knobs, one per input: it says how loud that input is in the final mix. Turn it up and that input matters more. Turn it to zero and the input is muted. A *negative* weight is a knob that works backwards: the louder the input, the more it pushes the other way.

**Formula.** A neuron multiplies each input by its weight and adds everything up:

```
sum = weight × input
```

![weight](charts/c1_weight.png)

With one input, the weight is just the **slope** of the line. Positive: the farther the cactus, the bigger the sum. Negative: the farther the cactus, the *smaller* the sum. Zero: the neuron ignores the distance entirely.

**Worked example, two inputs.** A neuron with weight −3 for the distance and +1 for the speed:

| Distance | Speed | Sum |
|---|---|---|
| 0.2 (close) | 0.5 | −3 × 0.2 + 1 × 0.5 = **−0.1** |
| 0.8 (far) | 0.5 | −3 × 0.8 + 1 × 0.5 = **−1.9** |

The weight −3 says "distance matters a lot, and the farther, the less I want to jump". The weight +1 says "speed matters a bit, and the faster, the more I want to jump".

**In the code:** `W1` (from the 2 inputs to the 6 hidden neurons) and `W2` (from the hidden neurons to the output). Those are 18 of the network's 25 numbers.

---

## 2. Bias: a thermostat

**Analogy.** A thermostat doesn't change *how* the heating works; it changes *when* it turns on. The **bias** is the same: a number added to the sum, which moves the point where the neuron switches from "no" to "yes".

**Formula.**

```
sum = weight × input + bias
```

![bias](charts/c2_bias.png)

Here the weight is fixed at −10 and only the bias changes. The neuron says "jump" (chance above 50%) where the sum is positive, so it switches at `x = bias / 10`: with bias 3 it starts wanting to jump when the cactus is closer than 0.3; with bias 7, already at 0.7.

Without a bias, every neuron would be forced to switch exactly at `x = 0`. The bias gives each neuron its own switch point.

**In the code:** `b1` (one per hidden neuron) and `b2` (one for the output): the other 7 of the 25 numbers. They start at zero, which is why, at epoch 0 of lesson 1, all the lines in [`06_hidden_lines.png`](../results/06_hidden_lines.png) cross at the same point.

---

## 3. Activation: a dimmer switch

**Analogy.** A plain light switch is either off or on. A **dimmer** goes smoothly from off to fully on. An activation function is a dimmer applied to the neuron's sum: it bends the straight line into a curve.

![activations](charts/c3_activation.png)

- **sigmoid** squeezes any sum into 0..1, which works as a **chance**. Lesson 1 uses it at the output: "chance of jumping".
- **tanh** squeezes into −1..1. Lesson 1 uses it in the hidden neurons.
- **ReLU** keeps positive sums and cuts negatives to zero. It and its smoother cousins (like GELU, used by the language model of [lesson 7](../../07-language-model/README.md)) are the most common in big networks today.

**Why it matters.** Without an activation, stacking neurons is useless: a line of a line is still a line. Two linear neurons in a row, the first `2x + 1` and the second `−3 × (first) + 0.5`, give:

```
−3 × (2x + 1) + 0.5 = −6x − 2.5        (just another straight line)
```

However many you stack, you get one straight line. The bend is what lets a network draw shapes.

---

## 4. Two dimmers make a band

**Analogy.** Think of a waiting room with a sign: "come in after 2pm, but not after 5pm". You need two rules, "at least 2pm" and "not yet 5pm", and you pass only when both agree. The "jump" band of lesson 1 is exactly that: "the cactus is at least this close, but not too close".

![band](charts/c4_band.png)

- **Neuron A** switches on when the distance passes 0.2. **Neuron B** switches on when it passes 0.5.
- The output neuron adds them with opposite weights, `4 × A − 4 × B − 3`: only between 0.2 and 0.5 is A on while B is still off.
- The sigmoid turns that bump into a **chance of jumping**.

| Distance | A | B | Sum 4A − 4B − 3 | Chance of jumping |
|---|---|---|---|---|
| 0.1 | −0.99 | −1.00 | −2.95 | 5% |
| 0.35 | +1.00 | −1.00 | +4.99 | **99%** |
| 0.8 | +1.00 | +1.00 | −3.00 | 5% |

That's what the trained network in lesson 1 does, with 2 inputs instead of 1: two of its hidden neurons end up on the two edges of the band ([`06_hidden_lines.png`](../results/06_hidden_lines.png)). Nobody set those numbers by hand. Training found them.

---

## 5. Loss: a bet

**Analogy.** Think of every prediction as a **bet**. When the network says "90% jump", it's betting 90% on jump. The **loss** is how much the bet costs once the right answer is revealed: a confident right bet costs almost nothing, a timid bet costs a bit, and a confident *wrong* bet costs a lot.

**Formula** (cross-entropy, the loss of lesson 1), with `p` = the chance of jumping the network gave:

```
if the right answer is JUMP:        loss = −log(p)
if the right answer is DON'T jump:  loss = −log(1 − p)
```

![loss](charts/c5_loss.png)

**Worked example, three bets:**

| Right answer | The network said | Loss |
|---|---|---|
| jump | 90% jump | 0.11 (confident and right: almost free) |
| don't jump | 20% jump | 0.22 (fairly sure and right) |
| jump | 5% jump | **3.00** (confident and wrong: expensive) |
| | **average** | **1.11** |

The loss of the network is the **average** over all examples. Training means making that average as small as possible.

**Why not just count mistakes?** "Right or wrong" can't tell a 51% bet from a 99% one, so it can't say *how* to improve. The loss changes smoothly with every weight, and that's exactly what the gradient needs.

**Why not squared error?** The right chart shows that squared error barely punishes a confident wrong answer (it never goes above 1). Cross-entropy shoots up, so those are the mistakes the network fixes first.

---

## 6. Gradient: walking downhill in fog

**Analogy.** You're on a hill in thick fog and want to reach the bottom. You can't see the valley, but your feet feel which way the ground slopes. So you take a step downhill, feel again, take another step. The **gradient** is that slope, felt at the spot where you stand. The **learning rate** is the size of your step.

![gradient](charts/c6_gradient.png)

Here the loss is drawn as a function of a single weight (the weight of the distance), with the bias fixed. The orange lines are the slope at each step:

| Step | Weight | Loss | Slope | New weight = weight − 12 × slope |
|---|---|---|---|---|
| 0 | +2.00 | 2.690 | +0.4025 | −2.83 |
| 1 | −2.83 | 0.930 | +0.2721 | −6.10 |
| 2 | −6.10 | 0.427 | +0.0527 | −6.73 |
| 9 | −7.54 | 0.390 | +0.0021 | −7.57 |

- A **positive slope** means "increasing the weight increases the loss", so the weight goes down. A negative slope would push it up.
- Far from the bottom the slope is steep and the steps are big. Near the bottom it flattens and the steps become tiny: the two phases of lesson 1's replay.

Lesson 1 does this for all 25 weights at once, and backpropagation computes the 25 slopes in one go. Section 8 of lesson 1 shows the same walk on a map with 2 weights, and what happens when the steps are too small or too big.

---

## Summary

| Idea | Analogy | What it does | In lesson 1's code |
|---|---|---|---|
| Weight | a volume knob | how much an input matters, and in which direction | `W1`, `W2` |
| Bias | a thermostat | where the neuron switches on | `b1`, `b2` |
| Activation | a dimmer | bends the sum into a curve, so the network can draw shapes | `np.tanh`, `sigmoid` |
| Loss | a bet | how much the network's answers cost, on average | `loss()` |
| Gradient | the slope under your feet in fog | which way, and how much, to move each weight | `backward()` |

**Back to:** [lesson 1](../README.md)
