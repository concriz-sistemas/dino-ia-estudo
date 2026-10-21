"""
watch.py — The dino trained with PPO playing, with its "mind" shown alongside.

Run from the repository root, after training:
    # the final model
    python lessons/04-reinforcement/watch.py
    # the model before training (random weights)
    python lessons/04-reinforcement/watch.py --model 0
    # records results/models.gif: several models on the same match
    python lessons/04-reinforcement/watch.py --gif

Keys: ← / → = previous / next model, R = new match, ESC = quit

In the panel on the right:
    - the chance the network gives each action right now (the highest one is chosen);
    - the VALUE: how much reward the critic expects to earn from here on.
      Notice how it drops when a hard obstacle is approaching.
"""
import argparse
import os
import random
import sys

FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FOLDER)

import pygame
import torch
from stable_baselines3 import PPO

from environment import DinoEnv, ACTION_NAMES, WIDTH, HEIGHT

MODELS_FOLDER = os.path.join(FOLDER, "results", "models")
PANEL = 300
ACTION_COLORS = [(150, 150, 150), (224, 49, 49), (25, 113, 194)]


def list_models():
    names = sorted(a for a in os.listdir(MODELS_FOLDER) if a.endswith(".zip"))
    return [(int(a[4:9]), os.path.join(MODELS_FOLDER, a)) for a in names]


def think(model, obs):
    """Asks the network: the probability of each action and the expected value of this moment."""
    with torch.no_grad():
        t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        probs = model.policy.get_distribution(t).distribution.probs[0].numpy()
        value = float(model.policy.predict_values(t)[0, 0])
    return probs, value


def draw_panel(screen, font, probs, value, value_history):
    x0 = WIDTH
    pygame.draw.rect(screen, (235, 235, 240), (x0, 0, PANEL, HEIGHT))
    screen.blit(font.render("Chance of each action", True, (40, 40, 40)), (x0 + 15, 12))
    choice = int(probs.argmax())
    for i, (name, p) in enumerate(zip(ACTION_NAMES, probs)):
        y = 40 + i * 30
        screen.blit(font.render(name, True, (40, 40, 40)), (x0 + 15, y))
        pygame.draw.rect(screen, (210, 210, 215), (x0 + 95, y, 150, 18))
        pygame.draw.rect(screen, ACTION_COLORS[i], (x0 + 95, y, int(150 * p), 18))
        if i == choice:
            pygame.draw.rect(screen, (255, 190, 0), (x0 + 93, y - 2, 154, 22), 2)
        screen.blit(font.render(f"{p:4.0%}", True, (40, 40, 40)), (x0 + 252, y))

    screen.blit(font.render(f"Expected value: {value:5.1f}", True, (40, 40, 40)), (x0 + 15, 145))
    # Chart of the value over the last frames
    gx, gy, gw, gh = x0 + 15, 170, PANEL - 30, 110
    pygame.draw.rect(screen, (250, 250, 252), (gx, gy, gw, gh))
    if len(value_history) > 1:
        # Scale fitted to the last frames, so the drops show up
        vmax = max(value_history) + 0.5
        vmin = min(value_history) - 0.5
        pts = [(gx + i * gw / (len(value_history) - 1),
                gy + gh - (v - vmin) / (vmax - vmin + 1e-9) * gh) for i, v in enumerate(value_history)]
        pygame.draw.lines(screen, (174, 62, 201), False, pts, 2)


def prepare(env, models, idx, seed):
    steps, path = models[idx]
    model = PPO.load(path, device="cpu")
    obs, _ = env.reset(seed=seed)
    return model, obs, f"Model after {steps}k training steps"


def record_gif(models, filename):
    """A few models playing the SAME match, one after the other."""
    from PIL import Image

    pygame.init()
    screen = pygame.Surface((WIDTH + PANEL, HEIGHT))
    font = pygame.font.SysFont("monospace", 14)
    chosen = sorted({0, len(models) // 3, 2 * len(models) // 3, len(models) - 1})
    env = DinoEnv()
    frames = []
    for idx in chosen:
        model, obs, title = prepare(env, models, idx, seed=7)
        history, after_crash, step = [], 0, 0
        while step < 350 and after_crash < 15:
            probs, value = think(model, obs)
            history = (history + [value])[-120:]
            dino = env.game.dinos[0]
            if dino.alive:
                obs, _, _, _, _ = env.step(int(probs.argmax()))
                step += 1
            else:
                after_crash += 1
            if step % 2 == 0 or not dino.alive:
                info = [title, f"Score: {dino.score}"] + ([] if dino.alive else ["Crashed!"])
                env.game.draw(screen, font, info, highlight=dino)
                draw_panel(screen, font, probs, value, history)
                img = Image.frombytes("RGB", screen.get_size(), pygame.image.tobytes(screen, "RGB"))
                frames.append(img.resize((img.width * 3 // 5, img.height * 3 // 5)))
        print(f"  {title}: score {env.game.dinos[0].score}")
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=66, loop=0, optimize=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=int, default=-1,
                   help="which saved model, in thousands of steps (default: the last one)")
    p.add_argument("--gif", action="store_true", help="records results/models.gif instead of opening the window")
    args = p.parse_args()

    if not os.path.isdir(MODELS_FOLDER) or not list_models():
        print("No saved model yet. First run: python lessons/04-reinforcement/train.py")
        return
    models = list_models()
    if args.gif:
        filename = os.path.join(FOLDER, "results", "models.gif")
        record_gif(models, filename)
        print(f"Animation saved to: {filename}")
        return

    idx = len(models) - 1
    if args.model >= 0:
        idx = min(range(len(models)), key=lambda i: abs(models[i][0] - args.model))

    pygame.init()
    screen = pygame.display.set_mode((WIDTH + PANEL, HEIGHT))
    pygame.display.set_caption("Dino AI — reinforcement learning")
    font = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()
    env = DinoEnv()
    model, obs, title = prepare(env, models, idx, random.randint(0, 10**9))
    history = []

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_r, pygame.K_LEFT, pygame.K_RIGHT):
                if e.key == pygame.K_RIGHT:
                    idx = min(idx + 1, len(models) - 1)
                elif e.key == pygame.K_LEFT:
                    idx = max(idx - 1, 0)
                model, obs, title = prepare(env, models, idx, random.randint(0, 10**9))
                history = []

        probs, value = think(model, obs)
        history = (history + [value])[-120:]
        dino = env.game.dinos[0]
        if dino.alive and dino.score < 5000:
            obs, _, _, _, _ = env.step(int(probs.argmax()))

        info = [title, f"Score: {dino.score}", "← → switch model | R new match"]
        if not dino.alive:
            info.append("Crashed!")
        env.game.draw(screen, font, info, highlight=dino)
        draw_panel(screen, font, probs, value, history)
        pygame.display.flip()
        clock.tick(DinoEnv.metadata["render_fps"])


if __name__ == "__main__":
    main()
