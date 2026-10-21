# Training report — reinforcement learning (PPO)

> Generated automatically by `report.py` from `episodes.csv`, `updates.csv` and the models in `models/`.
> Run the training again and this report changes.

## Summary

| | |
|---|---|
| Training steps | 2,002,960 |
| Matches played during training | 8,206 |
| Mean score: first 200 → last 200 matches | 115 → 4329 |
| First saved model with mean ≥ 4000 on new matches | 750k steps |
| Final model on 10 new matches | mean 5000, worst 5000, best 5000 |

## 1. Score per match

![scores](01_scores.png)

Notice the **plateau**: for a good while the agent does not improve, and then suddenly the average shoots up. It discovered that jumping at the right moment pays off more than jumping nonstop.

## 2. Inside PPO

![ppo](02_ppo.png)

- **Entropy** plunges together with the jump in scores: at first the agent picks almost at random; once it discovers what works, it becomes decisive. It does not go to zero because, with the obstacle far away, the action does not matter, and the entropy bonus rewards it for keeping on exploring.
- **Critic's error** rises while the agent improves (matches get long and unpredictable) and falls once it survives almost every time: then the reward is easy to predict.
- **KL** and **clip fraction** stay small the whole time: that is PPO's *clip* holding back each update so the policy does not change too much at once.

## 3. Each saved model on new matches

![evaluation](03_evaluation.png)

| Model | Mean | Worst | Best |
|---|---|---|---|
| 0k steps | 86 | 86 | 86 |
| 250k steps | 116 | 88 | 179 |
| 500k steps | 145 | 96 | 354 |
| 750k steps | 5000 | 5000 | 5000 |
| 1000k steps | 5000 | 5000 | 5000 |
| 1500k steps | 5000 | 5000 | 5000 |
| 2000k steps | 5000 | 5000 | 5000 |

## 4. What the network "thinks"

![decisions](04_decisions.png)

Each point is a question asked to the network: "with the obstacle at this distance and the game at this speed, what do you do?".

- **Top, the cactus:** at first the network answers the same for everything. At the end a **slanted boundary** appears: jump when the cactus gets close, and earlier the faster the game. It is the same idea as the band from lesson 1, but there the network was given ready-made examples; here, only the reward.
- **Bottom, the bird at head height:** the color shows the most likely action. Notice that the network does not have to choose what we would choose: jumping over the bird also works. The reward only says whether it worked, not how.
