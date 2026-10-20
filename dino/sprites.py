"""
sprites.py — As figuras do jogo, desenhadas em "pixel art" como texto.

Cada figura é uma lista de linhas: '#' = pixel pintado, 'w' = pixel branco
(o olho), '.' = transparente. Edite à vontade para criar seus próprios
personagens: o tamanho da figura na tela é ajustado sozinho.

Importante: isto é SÓ visual. A colisão continua usando os retângulos de
jogo.py, então mudar as figuras não muda o que a IA aprende.
"""
import pygame

DINO_CORRENDO_1 = [
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

DINO_CORRENDO_2 = DINO_CORRENDO_1[:16] + [
    "....###..##...........",
    "....#....##...........",
    "....##.....#..........",
    "...........##.........",
]

DINO_PULANDO = DINO_CORRENDO_1[:16] + [
    "....###..##...........",
    "....##....#...........",
    "....#.....#...........",
    "....##....##..........",
]

DINO_BATEU = [linha.replace("w", ".") for linha in DINO_PULANDO]
DINO_BATEU[1] = "..........#w.w#######."
DINO_BATEU[2] = "..........##w########."
DINO_BATEU[3] = "..........#w.w#######."

DINO_ABAIXADO_1 = [
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

DINO_ABAIXADO_2 = DINO_ABAIXADO_1[:8] + [
    "....###...##..................",
    "....#.....##..................",
    "....##......#.................",
    "............##................",
]

CACTO = [
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

PASSARO_1 = [
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

PASSARO_2 = [
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

NUVEM = [
    "..........######........",
    "........##......##......",
    "...####.#.........#.....",
    "..#....#...........###..",
    ".#....................#.",
    "#######################.",
]

BRANCO = (255, 255, 255)
_cache = {}


def _desenhar_mapa(mapa, cor):
    """Transforma o mapa de texto numa Surface do pygame, 1 pixel por caractere."""
    largura = max(len(linha) for linha in mapa)
    surf = pygame.Surface((largura, len(mapa)), pygame.SRCALPHA)
    for y, linha in enumerate(mapa):
        for x, c in enumerate(linha):
            if c == "#":
                surf.set_at((x, y), (*cor, 255))
            elif c == "w":
                surf.set_at((x, y), (*BRANCO, 255))
    return surf


def figura(nome, mapa, cor, tamanho, alfa=255):
    """Devolve a figura já no tamanho pedido (guardada em cache: desenhar é caro)."""
    chave = (nome, cor, tamanho, alfa)
    if chave not in _cache:
        surf = pygame.transform.scale(_desenhar_mapa(mapa, cor), tamanho)
        if alfa < 255:
            surf.set_alpha(alfa)
        _cache[chave] = surf
    return _cache[chave]


def grupo_de_cactos(cor, tamanho):
    """Três cactos de alturas diferentes lado a lado."""
    chave = ("grupo", cor, tamanho)
    if chave not in _cache:
        w, h = tamanho
        surf = pygame.Surface(tamanho, pygame.SRCALPHA)
        larg = w // 3 + 2
        for i, frac in enumerate([0.8, 1.0, 0.7]):
            alt = int(h * frac)
            surf.blit(figura("cacto", CACTO, cor, (larg, alt)), (i * (w - larg) // 2, h - alt))
        _cache[chave] = surf
    return _cache[chave]
