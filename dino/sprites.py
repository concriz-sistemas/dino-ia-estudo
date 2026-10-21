"""
sprites.py — The game's figures, drawn as "pixel art" in plain text.

Each figure is a list of rows: '#' = painted pixel, 'w' = white pixel
(the eye), '.' = transparent. Edit them freely to create your own
characters: the on-screen size is adjusted automatically.

Important: this is ONLY visual. Collisions still use the rectangles in
game.py, so changing the figures does not change what the AI learns.
"""
import pygame

DINO_RUN_1 = [
    "...........#########..",
    "..........##w########.",
    "..........###########.",
    "..........###########.",
    "..........######......",
    "..........#########...",
    "#.........#####.......",
    "#........######.......",
    "##.....##########.....",
    "###...#########.#.....",
    "##############........",
    "##############........",
    ".############.........",
    "..##########..........",
    "...########...........",
    "....######............",
    "....###..##...........",
    "....##....#...........",
    "....#.....##..........",
    "....##................",
]

DINO_RUN_2 = DINO_RUN_1[:16] + [
    "....###..##...........",
    "....#....##...........",
    "....##.....#..........",
    "...........##.........",
]

DINO_JUMP = DINO_RUN_1[:16] + [
    "....###..##...........",
    "....##....#...........",
    "....#.....#...........",
    "....##....##..........",
]

DINO_DEAD = [row.replace("w", ".") for row in DINO_JUMP]
DINO_DEAD[1] = "..........#w.w#######."
DINO_DEAD[2] = "..........##w########."
DINO_DEAD[3] = "..........#w.w#######."

DINO_DUCK_1 = [
    "...................##########.",
    "#..........#######.##w#######.",
    "##....#######################.",
    "###.#########################.",
    "#############################.",
    ".################.#########...",
    "..##############..............",
    "...###########................",
    "....###...##..................",
    "....##.....#..................",
    "....#......##.................",
    "....##........................",
]

DINO_DUCK_2 = DINO_DUCK_1[:8] + [
    "....###...##..................",
    "....#.....##..................",
    "....##......#.................",
    "............##................",
]

CACTUS = [
    "....##....",
    "...####...",
    "...####.#.",
    "...####.##",
    "#..####.##",
    "##.####.##",
    "##.#######",
    "##.######.",
    "#######...",
    ".######...",
    "...####...",
    "...####...",
    "...####...",
    "...####...",
    "...####...",
    "...####...",
    "...####...",
    "...####...",
    "...####...",
    "...####...",
]

BIRD_1 = [
    "......#.................",
    "......##................",
    "......###...............",
    "...#..####..............",
    "..##..#####.............",
    ".####.######............",
    "########################",
    "....##################..",
    "......##############....",
    "........#########.......",
    "..........######........",
]

BIRD_2 = [
    "........................",
    "........................",
    "........................",
    "...#....................",
    "..##....................",
    ".####...................",
    "########################",
    "....##################..",
    "......########..........",
    "......######............",
    "......#####.............",
    "......####..............",
    "......###...............",
    "......##................",
]

CLOUD = [
    "..........######........",
    "........##......##......",
    "...####.#.........#.....",
    "..#....#...........###..",
    ".#....................#.",
    "#######################.",
]

WHITE = (255, 255, 255)
_cache = {}


def _render_map(pixel_map, color):
    """Turns the text map into a pygame Surface, 1 pixel per character."""
    width = max(len(row) for row in pixel_map)
    surf = pygame.Surface((width, len(pixel_map)), pygame.SRCALPHA)
    for y, row in enumerate(pixel_map):
        for x, c in enumerate(row):
            if c == "#":
                surf.set_at((x, y), (*color, 255))
            elif c == "w":
                surf.set_at((x, y), (*WHITE, 255))
    return surf


def sprite(name, pixel_map, color, size, alpha=255):
    """Returns the figure at the requested size (cached: rendering is expensive)."""
    key = (name, color, size, alpha)
    if key not in _cache:
        surf = pygame.transform.scale(_render_map(pixel_map, color), size)
        if alpha < 255:
            surf.set_alpha(alpha)
        _cache[key] = surf
    return _cache[key]


def cactus_group(color, size):
    """Three cacti of different heights side by side."""
    key = ("group", color, size)
    if key not in _cache:
        w, h = size
        surf = pygame.Surface(size, pygame.SRCALPHA)
        width = w // 3 + 2
        for i, frac in enumerate([0.8, 1.0, 0.7]):
            height = int(h * frac)
            surf.blit(sprite("cactus", CACTUS, color, (width, height)), (i * (w - width) // 2, h - height))
        _cache[key] = surf
    return _cache[key]
