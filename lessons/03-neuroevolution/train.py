"""
Lesson 3 — Trains dinosaurs with neuroevolution (NEAT) and RECORDS everything.

Run from the repository root:
    # with a window, so you can watch
    python lessons/03-neuroevolution/train.py
    # without a window, much faster
    python lessons/03-neuroevolution/train.py --fast
    python lessons/03-neuroevolution/train.py --generations 50

Keys during training (window mode):
    V   = toggle drawing on/off (speeds things up a lot)
    ESC = stop and save what has been recorded so far

What gets saved in results/:
    history.json   — for EACH generation: scores, species, the full champion
                     (all of its weights) and the weights of the whole population
    champions.pkl  — the champion of each generation, to watch later
Then run the report:
    python lessons/03-neuroevolution/report.py
"""
import argparse
import configparser
import json
import os
import pickle
import random
import statistics
import sys

FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(FOLDER, "..", "..")))

import neat
import pygame

from dino.game import Game, WIDTH, HEIGHT, NOTHING, JUMP, DUCK
from net_utils import genome_to_dict

CONFIG_FILE = os.path.join(FOLDER, "neat-config.txt")
OUTPUT = os.path.join(FOLDER, "results")
HISTORY_FILE = os.path.join(OUTPUT, "history.json")
CHAMPIONS_FILE = os.path.join(OUTPUT, "champions.pkl")

# NEAT draws all its random numbers (initial weights, mutations, who reproduces)
# from Python's `random` module. Seeding it makes every run evolve exactly the same
# way, so two runs give the same results. Change it to see a different evolution.
SEED = 0

# Score cap per match — without it, a perfect dino would play forever
SCORE_LIMIT = 5000
ACTIONS = [NOTHING, JUMP, DUCK]


class TrainingInterrupted(Exception):
    pass


# ---------------------------------------------------------------------------
# The training "notebook".
# NEAT calls post_evaluate() at the end of each generation, with everyone's scores.
# ---------------------------------------------------------------------------
class Recorder(neat.reporting.BaseReporter):
    def __init__(self):
        self.generation = -1
        self.generations = []
        self.champions = []

    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        scores = [g.fitness for g in population.values()]
        # every enabled weight of the whole population (to see the distribution)
        weights = [round(c.weight, 3)
                   for g in population.values()
                   for c in g.connections.values() if c.enabled]
        species_list = [
            {"id": sid, "size": len(s.members),
             "best": max(population[k].fitness for k in s.members if k in population)}
            for sid, s in species.species.items()
        ]
        champion = max(population.values(), key=lambda g: g.fitness)
        self.generations.append({
            "generation": self.generation,
            "best": max(scores),
            "average": round(statistics.mean(scores), 2),
            "median": statistics.median(scores),
            "stdev": round(statistics.pstdev(scores), 2),
            "n_species": len(species_list),
            "species": species_list,
            "champion": genome_to_dict(champion),
            "population_weights": weights,
        })
        self.champions.append(pickle.loads(pickle.dumps(champion)))  # a copy

    def save(self, config_path):
        os.makedirs(OUTPUT, exist_ok=True)
        cp = configparser.ConfigParser()
        cp.read(config_path)
        data = {
            "parameters": {sec: dict(cp[sec]) for sec in cp.sections()},
            "score_limit": SCORE_LIMIT,
            "generations": self.generations,
        }
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        with open(CHAMPIONS_FILE, "wb") as f:
            pickle.dump(self.champions, f)


# ---------------------------------------------------------------------------
# Evaluation: each generation plays ONE match, all together
# ---------------------------------------------------------------------------
class Trainer:
    def __init__(self, show):
        self.show = show
        self.draw = show
        self.stop = False
        self.n = 0
        if show:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Dino AI — training with NEAT")
            self.font = pygame.font.SysFont("monospace", 16)
            self.clock = pygame.time.Clock()

    def _events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                self.stop = True
            if e.type == pygame.KEYDOWN and e.key == pygame.K_v:
                self.draw = not self.draw

    def eval_genomes(self, genomes, config):
        if self.stop:
            raise TrainingInterrupted()
        self.n += 1
        # Same seed for the whole generation => everyone faces the same obstacles.
        # A different seed each generation => nobody can "memorize" the level.
        game = Game(len(genomes), seed=self.n)
        nets = [neat.nn.FeedForwardNetwork.create(g, config) for _, g in genomes]

        while game.any_alive() and game.frame < SCORE_LIMIT and not self.stop:
            if self.show:
                self._events()

            # 1) each network looks at the sensors and picks an action (the largest output wins)
            actions = []
            for dino, net in zip(game.dinos, nets):
                if not dino.alive:
                    actions.append(NOTHING)
                    continue
                outputs = net.activate(game.sensors(dino))
                actions.append(ACTIONS[outputs.index(max(outputs))])

            # 2) the game advances one frame
            game.step(actions)

            # 3) draw (optional)
            if self.show and self.draw:
                alive = [d for d in game.dinos if d.alive]
                game.draw(
                    self.screen, self.font,
                    [f"Generation: {self.n - 1}",
                     f"Alive: {len(alive)}/{len(game.dinos)}",
                     f"Score: {game.frame}",
                     f"Speed: {game.speed:.1f}",
                     "V = speed up | ESC = stop"],
                    highlight=alive[0] if alive else None,
                )
                pygame.display.flip()
                self.clock.tick(60)
            elif self.show and game.frame % 200 == 0:
                pygame.display.set_caption(f"Generation {self.n - 1} — {game.frame} pts (V to watch)")

        # 4) each genome's score (fitness) = how many frames its dino survived
        for (_, genome), dino in zip(genomes, game.dinos):
            genome.fitness = dino.score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true", help="train without a window")
    p.add_argument("--generations", type=int, default=30)
    args = p.parse_args()

    # Must happen before the population is created: the first genomes'
    # random weights already come from `random`.
    random.seed(SEED)

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        CONFIG_FILE,
    )
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(False))
    recorder = Recorder()
    population.add_reporter(recorder)

    trainer = Trainer(show=not args.fast)
    try:
        population.run(trainer.eval_genomes, args.generations)
    except TrainingInterrupted:
        print("\nTraining interrupted — saving what has been recorded so far.")

    if not recorder.generations:
        print("No complete generation to save.")
        return
    recorder.save(CONFIG_FILE)

    last = recorder.generations[-1]
    print("\n=== Training finished ===")
    print(f"Generations:       {len(recorder.generations)}")
    print(f"Best in the last:  {last['best']}   average: {last['average']}")
    print(f"History saved to:  {HISTORY_FILE}")
    print("Next steps:")
    print("  Charts + REPORT.md:")
    print("    python lessons/03-neuroevolution/report.py")
    print("  Final champion playing:")
    print("    python lessons/03-neuroevolution/watch.py")
    print("  Compare with the champion of generation 0:")
    print("    python lessons/03-neuroevolution/watch.py --generation 0")

    if trainer.show:
        pygame.quit()


if __name__ == "__main__":
    main()
