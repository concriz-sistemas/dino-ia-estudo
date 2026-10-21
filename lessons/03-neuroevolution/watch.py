"""
watch.py — Shows a generation's champion playing, with its network beside it.

Run from the repository root, after training:
    # champion of the last generation
    python lessons/03-neuroevolution/watch.py
    # champion of generation 0
    python lessons/03-neuroevolution/watch.py --generation 0
    # records results/champions.gif: champions of several generations in the same match
    python lessons/03-neuroevolution/watch.py --gif

Keys: R = new match, ← / → = champion of the previous / next generation, ESC = quit

In the right-hand panel: inputs on the left (the bluer, the larger the value),
outputs on the right (the yellow one is the chosen action). Green lines = positive
weight, red = negative; thickness = strength of the weight.
"""
import argparse
import os
import pickle
import random
import sys

FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(FOLDER, "..", "..")))

import neat
import pygame

from dino.game import Game, WIDTH, HEIGHT, NOTHING, JUMP, DUCK
from net_utils import INPUT_NAMES, OUTPUT_NAMES

ACTIONS = [NOTHING, JUMP, DUCK]
PANEL = 340


def draw_network(screen, font, genome, config, inputs, outputs):
    gc = config.genome_config
    x0 = WIDTH
    pygame.draw.rect(screen, (235, 235, 240), (x0, 0, PANEL, HEIGHT))
    hidden = [k for k in genome.nodes if k not in gc.output_keys]
    pos = {}
    for i, k in enumerate(gc.input_keys):
        pos[k] = (x0 + 85, (i + 1) * HEIGHT // (len(gc.input_keys) + 1))
    for i, k in enumerate(hidden):
        pos[k] = (x0 + 185, (i + 1) * HEIGHT // (len(hidden) + 1))
    for i, k in enumerate(gc.output_keys):
        pos[k] = (x0 + 265, (i + 1) * HEIGHT // (len(gc.output_keys) + 1))

    for (a, b), c in genome.connections.items():
        if c.enabled and a in pos and b in pos:
            color = (60, 160, 80) if c.weight > 0 else (200, 70, 70)
            pygame.draw.line(screen, color, pos[a], pos[b], max(1, min(5, int(abs(c.weight)))))

    choice = outputs.index(max(outputs))
    for i, k in enumerate(gc.input_keys):
        v = max(0.0, min(1.0, inputs[i]))
        pygame.draw.circle(screen, (int(255 * (1 - v)), int(255 * (1 - v)), 255), pos[k], 9)
        pygame.draw.circle(screen, (50, 50, 50), pos[k], 9, 1)
        screen.blit(font.render(INPUT_NAMES[i], True, (40, 40, 40)), (x0 + 4, pos[k][1] - 7))
    for k in hidden:
        pygame.draw.circle(screen, (190, 180, 230), pos[k], 7)
    for i, k in enumerate(gc.output_keys):
        color = (255, 200, 0) if i == choice else (220, 220, 220)
        pygame.draw.circle(screen, color, pos[k], 10)
        pygame.draw.circle(screen, (50, 50, 50), pos[k], 10, 1)
        screen.blit(font.render(OUTPUT_NAMES[i], True, (40, 40, 40)), (pos[k][0] + 14, pos[k][1] - 7))


def record_gif(champions, config, path):
    """Champions of a few generations playing the SAME match, one after the other."""
    from PIL import Image

    last = len(champions) - 1
    generations = sorted({0, last // 4, last // 2, last})
    pygame.init()
    screen = pygame.Surface((WIDTH + PANEL, HEIGHT))
    font = pygame.font.SysFont("monospace", 14)
    frames = []
    for g in generations:
        net = neat.nn.FeedForwardNetwork.create(champions[g], config)
        game = Game(1, seed=7, colors=[(40, 120, 220)])
        after_crash = 0
        while game.frame < 600 and after_crash < 45:
            dino = game.dinos[0]
            inputs = game.sensors(dino)
            outputs = net.activate(inputs)
            if dino.alive:
                game.step([ACTIONS[outputs.index(max(outputs))]])
            else:
                after_crash += 1
            # 1 in every 4 game frames; after the crash, 1 in every 3 (just to pause the scene)
            if (dino.alive and game.frame % 4 == 0) or (not dino.alive and after_crash % 3 == 0):
                info = [f"Champion of generation {g}", f"Score: {dino.score}"]
                if not dino.alive:
                    info.append("Crashed!")
                game.draw(screen, font, info, highlight=dino)
                draw_network(screen, font, champions[g], config, inputs, outputs)
                img = Image.frombytes("RGB", screen.get_size(), pygame.image.tobytes(screen, "RGB"))
                frames.append(img.resize((img.width * 3 // 5, img.height * 3 // 5)))
        print(f"  generation {g}: {game.dinos[0].score} points")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=66, loop=0, optimize=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generation", type=int, default=-1, help="which generation to watch (default: the last)")
    p.add_argument("--gif", action="store_true", help="record results/champions.gif instead of opening the window")
    args = p.parse_args()

    path = os.path.join(FOLDER, "results", "champions.pkl")
    if not os.path.exists(path):
        print("No saved training yet. First run: python lessons/03-neuroevolution/train.py")
        return
    with open(path, "rb") as f:
        champions = pickle.load(f)
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        os.path.join(FOLDER, "neat-config.txt"),
    )
    if args.gif:
        gif_path = os.path.join(FOLDER, "results", "champions.gif")
        record_gif(champions, config, gif_path)
        print(f"Animation saved to: {gif_path}")
        return
    idx = args.generation % len(champions)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH + PANEL, HEIGHT))
    font = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()

    def setup(i):
        pygame.display.set_caption(f"Dino AI — champion of generation {i}")
        net = neat.nn.FeedForwardNetwork.create(champions[i], config)
        game = Game(1, seed=random.randint(0, 10**9), colors=[(40, 120, 220)])
        return net, game

    net, game = setup(idx)
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_r:
                    net, game = setup(idx)
                elif e.key == pygame.K_RIGHT:
                    idx = min(idx + 1, len(champions) - 1)
                    net, game = setup(idx)
                elif e.key == pygame.K_LEFT:
                    idx = max(idx - 1, 0)
                    net, game = setup(idx)

        dino = game.dinos[0]
        inputs = game.sensors(dino)
        outputs = net.activate(inputs)
        if dino.alive:
            game.step([ACTIONS[outputs.index(max(outputs))]])

        info = [f"Champion of generation {idx} (of {len(champions) - 1})",
                f"Score: {dino.score}", f"Speed: {game.speed:.1f}",
                "← → change generation | R new match"]
        if not dino.alive:
            info.append("Crashed!")
        game.draw(screen, font, info, highlight=dino)
        draw_network(screen, font, champions[idx], config, inputs, outputs)
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
