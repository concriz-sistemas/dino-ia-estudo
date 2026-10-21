"""
game.py — A simplified version of Chrome's offline dinosaur game.

The game supports MANY dinosaurs at once, all facing the same obstacles.
That is essential for neuroevolution: the whole population plays the same
match, so comparing them is fair.

Run this file directly to play yourself:
    python dino/game.py
Controls: SPACE / ↑ = jump,  ↓ = duck,  R = restart
"""
import random
import pygame

try:
    from dino import sprites
except ImportError:  # running "python dino/game.py" directly
    import sprites

# ---------------------------------------------------------------------------
# World constants
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 800, 300
GROUND_Y = 250            # y coordinate of the ground (the dino's feet rest here)
GRAVITY = 1.0
JUMP_FORCE = -17          # initial vertical speed of a jump (negative = up)
START_SPEED = 8           # pixels per frame
MAX_SPEED = 20
ACCELERATION = 0.003      # how much the speed grows every frame

# Possible actions (this is what the neural network chooses)
NOTHING, JUMP, DUCK = 0, 1, 2

# The names of the 6 numbers returned by Game.sensors()
SENSOR_NAMES = ["distance", "width", "height", "elevation", "dino height", "speed"]

BG_COLOR = (247, 247, 247)
GROUND_COLOR = (83, 83, 83)
OBSTACLE_COLOR = (60, 60, 60)
CLOUD_COLOR = (205, 205, 205)


# ---------------------------------------------------------------------------
# Dinosaur
# ---------------------------------------------------------------------------
class Dino:
    STANDING_W, STANDING_H = 40, 50
    DUCKING_W, DUCKING_H = 55, 26

    def __init__(self, color=(83, 83, 83)):
        self.x = 60
        self.feet = GROUND_Y       # y of the bottom of the dino
        self.vy = 0.0
        self.ducking = False
        self.alive = True
        self.score = 0             # frames survived (becomes the "fitness")
        self.color = color

    @property
    def on_ground(self):
        return self.feet >= GROUND_Y

    @property
    def rect(self):
        if self.ducking and self.on_ground:
            w, h = self.DUCKING_W, self.DUCKING_H
        else:
            w, h = self.STANDING_W, self.STANDING_H
        return pygame.Rect(self.x, int(self.feet - h), w, h)

    def update(self, action):
        if action == JUMP and self.on_ground:
            self.vy = JUMP_FORCE
        self.ducking = (action == DUCK)
        # Ducking in the air makes the dino fall faster (like the original game)
        if self.ducking and not self.on_ground:
            self.vy += 2
        self.vy += GRAVITY
        self.feet += self.vy
        if self.feet >= GROUND_Y:
            self.feet = GROUND_Y
            self.vy = 0


# ---------------------------------------------------------------------------
# Obstacles
# ---------------------------------------------------------------------------
# kind: (width, height, distance from its bottom to the ground)
OBSTACLE_TYPES = {
    "small_cactus": (20, 40, 0),
    "big_cactus": (30, 55, 0),
    "cactus_group": (55, 40, 0),
    "bird_low": (40, 25, 10),     # must jump
    "bird_mid": (40, 25, 35),     # must duck (or jump over it)
    "bird_high": (40, 25, 60),    # can be ignored
}


class Obstacle:
    def __init__(self, kind, x):
        self.kind = kind
        w, h, elevation = OBSTACLE_TYPES[kind]
        self.rect = pygame.Rect(x, GROUND_Y - elevation - h, w, h)
        self.x = float(x)

    @property
    def is_bird(self):
        return self.kind.startswith("bird")

    def move(self, speed):
        self.x -= speed
        self.rect.x = int(self.x)


# ---------------------------------------------------------------------------
# The game itself
# ---------------------------------------------------------------------------
class Game:
    def __init__(self, n_dinos=1, seed=None, colors=None):
        self.rng = random.Random(seed)
        if colors is None:
            colors = [self._random_color() for _ in range(n_dinos)]
        self.dinos = [Dino(color) for color in colors]
        self.obstacles = []
        self.speed = START_SPEED
        self.frame = 0
        self.distance = 0.0        # only used to scroll the scenery (ground and clouds)
        self._next_gap = 0

    def _random_color(self):
        return tuple(self.rng.randint(40, 200) for _ in range(3))

    # ---- logic -------------------------------------------------------------
    def any_alive(self):
        return any(d.alive for d in self.dinos)

    def _spawn_obstacle(self):
        # Birds only show up after the game speeds up a bit
        kinds = ["small_cactus", "big_cactus", "cactus_group"]
        if self.speed > 10:
            kinds += ["bird_low", "bird_mid", "bird_high"]
        kind = self.rng.choice(kinds)
        self.obstacles.append(Obstacle(kind, WIDTH + 10))
        # the gap to the next obstacle grows with the speed
        self._next_gap = self.rng.randint(int(self.speed * 30), int(self.speed * 55))

    def step(self, actions):
        """Advances the game one frame. `actions` has one action per dino."""
        self.frame += 1

        for dino, action in zip(self.dinos, actions):
            if dino.alive:
                dino.update(action)

        for obs in self.obstacles:
            obs.move(self.speed)
        self.obstacles = [o for o in self.obstacles if o.rect.right > 0]

        if not self.obstacles or (WIDTH - self.obstacles[-1].rect.right) > self._next_gap:
            self._spawn_obstacle()

        # collisions (slightly smaller hitbox, to be fair)
        for dino in self.dinos:
            if not dino.alive:
                continue
            hit = dino.rect.inflate(-8, -8)
            if any(hit.colliderect(o.rect) for o in self.obstacles):
                dino.alive = False
            else:
                dino.score += 1

        self.distance += self.speed
        self.speed = min(MAX_SPEED, self.speed + ACCELERATION)

    # ---- what the AI "sees" ------------------------------------------------
    def next_obstacle(self, dino):
        for o in self.obstacles:
            if o.rect.right > dino.x:
                return o
        return None

    def sensors(self, dino):
        """
        Turns the game state into 6 numbers (normalized to roughly 0..1).
        These are the INPUTS of the neural network. Choosing good inputs is
        half the work — try changing them later!
        """
        o = self.next_obstacle(dino)
        if o is None:
            dist, width, height, elev = 1.0, 0.0, 0.0, 0.0
        else:
            dist = (o.rect.left - (dino.x + Dino.STANDING_W)) / WIDTH
            width = o.rect.width / 60
            height = o.rect.height / 60
            elev = (GROUND_Y - o.rect.bottom) / 60     # 0 = on the ground; >0 = bird in the air
        dino_height = (GROUND_Y - dino.feet) / 150     # 0 = on the ground
        speed = self.speed / MAX_SPEED
        return [dist, width, height, elev, dino_height, speed]

    # ---- drawing -----------------------------------------------------------
    def _draw_scenery(self, screen):
        screen.fill(BG_COLOR)
        # Clouds move slower than the ground (they look far away)
        for x0, y in [(100, 50), (420, 80), (700, 40), (950, 70)]:
            x = (x0 - self.distance * 0.15) % (WIDTH + 150) - 75
            screen.blit(sprites.sprite("cloud", sprites.CLOUD, CLOUD_COLOR, (60, 15)), (x, y))
        pygame.draw.line(screen, GROUND_COLOR, (0, GROUND_Y), (WIDTH, GROUND_Y), 2)
        # Pebbles on the ground, scrolling at the game's speed
        for i in range(40):
            x = (i * 53 + i * i * 7 - self.distance) % (WIDTH + 20) - 10
            y = GROUND_Y + 5 + (i * 37) % 12
            pygame.draw.line(screen, GROUND_COLOR, (x, y), (x + 2 + i % 4, y), 1)

    def _draw_obstacle(self, screen, o):
        size = o.rect.size
        if o.kind == "cactus_group":
            img = sprites.cactus_group(OBSTACLE_COLOR, size)
        elif o.is_bird:
            wing = (self.frame // 10) % 2
            img = sprites.sprite(f"bird{wing}", [sprites.BIRD_1, sprites.BIRD_2][wing],
                                 OBSTACLE_COLOR, size)
        else:
            img = sprites.sprite("cactus", sprites.CACTUS, OBSTACLE_COLOR, size)
        screen.blit(img, o.rect)

    def _draw_dino(self, screen, d, alpha):
        leg = (self.frame // 6) % 2
        if not d.alive:
            name, pixel_map = "dead", sprites.DINO_DEAD
        elif not d.on_ground:
            name, pixel_map = "jumping", sprites.DINO_JUMP
        elif d.ducking:
            name, pixel_map = f"ducking{leg}", [sprites.DINO_DUCK_1, sprites.DINO_DUCK_2][leg]
        else:
            name, pixel_map = f"running{leg}", [sprites.DINO_RUN_1, sprites.DINO_RUN_2][leg]
        screen.blit(sprites.sprite(name, pixel_map, d.color, d.rect.size, alpha), d.rect)

    def draw(self, screen, font, info_lines=(), highlight=None):
        self._draw_scenery(screen)

        for o in self.obstacles:
            self._draw_obstacle(screen, o)

        # With many dinos, they are half transparent so you can see the crowd.
        # With a single one, it is drawn solid and stays on screen after crashing.
        alone = len(self.dinos) == 1
        for d in self.dinos:
            if d is not highlight and (d.alive or alone):
                self._draw_dino(screen, d, 255 if alone else 110)
        if highlight is not None and (highlight.alive or alone):
            self._draw_dino(screen, highlight, 255)

        # "Sensor" line: from the highlighted dino to the obstacle it is looking at
        if highlight is not None and highlight.alive:
            o = self.next_obstacle(highlight)
            if o:
                pygame.draw.line(screen, (230, 80, 80), highlight.rect.center, o.rect.center, 1)

        y = 8
        for line in info_lines:
            screen.blit(font.render(line, True, (40, 40, 40)), (10, y))
            y += 20


# ---------------------------------------------------------------------------
# Human mode: play yourself to feel the difficulty
# ---------------------------------------------------------------------------
def play_manual():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Dino — human mode")
    font = pygame.font.SysFont("monospace", 16)
    clock = pygame.time.Clock()
    game = Game(1, colors=[(83, 83, 83)])

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                game = Game(1, colors=[(83, 83, 83)])

        keys = pygame.key.get_pressed()
        action = NOTHING
        if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
            action = JUMP
        elif keys[pygame.K_DOWN]:
            action = DUCK

        if game.any_alive():
            game.step([action])
        info = [f"Score: {game.dinos[0].score}", f"Speed: {game.speed:.1f}"]
        if not game.any_alive():
            info.append("GAME OVER — press R")
        # Shows the 6 numbers the AI "sees" — the same things you see with your eyes
        values = game.sensors(game.dinos[0])
        info.append("")
        info.append("What the AI sees:")
        info += [f"  {n:<11} {v:5.2f}" for n, v in zip(SENSOR_NAMES, values)]
        game.draw(screen, font, info, highlight=game.dinos[0])
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    play_manual()
