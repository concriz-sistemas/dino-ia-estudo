"""
assistir.py — Mostra o campeão de uma geração jogando, com a rede dele ao lado.

Rode a partir da raiz do repositório, depois de treinar:
    python licoes/03-neuroevolucao/assistir.py               # campeão da última geração
    python licoes/03-neuroevolucao/assistir.py --geracao 0   # campeão da geração 0

Teclas: R = nova partida, ← / → = campeão da geração anterior / seguinte, ESC = sair

No painel da direita: entradas à esquerda (quanto mais azul, maior o valor),
saídas à direita (a amarela é a ação escolhida). Linhas verdes = peso positivo,
vermelhas = negativo; espessura = força do peso.
"""
import argparse
import os
import pickle
import random
import sys

PASTA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(PASTA, "..", "..")))

import neat
import pygame

from dino.jogo import Jogo, LARGURA, ALTURA, NADA, PULAR, ABAIXAR
from rede_util import NOMES_ENTRADAS, NOMES_SAIDAS

ACOES = [NADA, PULAR, ABAIXAR]
PAINEL = 340


def desenhar_rede(tela, fonte, genoma, config, entradas, saidas):
    gc = config.genome_config
    x0 = LARGURA
    pygame.draw.rect(tela, (235, 235, 240), (x0, 0, PAINEL, ALTURA))
    ocultos = [k for k in genoma.nodes if k not in gc.output_keys]
    pos = {}
    for i, k in enumerate(gc.input_keys):
        pos[k] = (x0 + 85, (i + 1) * ALTURA // (len(gc.input_keys) + 1))
    for i, k in enumerate(ocultos):
        pos[k] = (x0 + 185, (i + 1) * ALTURA // (len(ocultos) + 1))
    for i, k in enumerate(gc.output_keys):
        pos[k] = (x0 + 265, (i + 1) * ALTURA // (len(gc.output_keys) + 1))

    for (a, b), c in genoma.connections.items():
        if c.enabled and a in pos and b in pos:
            cor = (60, 160, 80) if c.weight > 0 else (200, 70, 70)
            pygame.draw.line(tela, cor, pos[a], pos[b], max(1, min(5, int(abs(c.weight)))))

    escolha = saidas.index(max(saidas))
    for i, k in enumerate(gc.input_keys):
        v = max(0.0, min(1.0, entradas[i]))
        pygame.draw.circle(tela, (int(255 * (1 - v)), int(255 * (1 - v)), 255), pos[k], 9)
        pygame.draw.circle(tela, (50, 50, 50), pos[k], 9, 1)
        tela.blit(fonte.render(NOMES_ENTRADAS[i], True, (40, 40, 40)), (x0 + 4, pos[k][1] - 7))
    for k in ocultos:
        pygame.draw.circle(tela, (190, 180, 230), pos[k], 7)
    for i, k in enumerate(gc.output_keys):
        cor = (255, 200, 0) if i == escolha else (220, 220, 220)
        pygame.draw.circle(tela, cor, pos[k], 10)
        pygame.draw.circle(tela, (50, 50, 50), pos[k], 10, 1)
        tela.blit(fonte.render(NOMES_SAIDAS[i], True, (40, 40, 40)), (pos[k][0] + 14, pos[k][1] - 7))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--geracao", type=int, default=-1, help="qual geração assistir (padrão: a última)")
    args = p.parse_args()

    arq = os.path.join(PASTA, "resultados", "campeoes.pkl")
    if not os.path.exists(arq):
        print("Nenhum treino salvo ainda. Rode primeiro: python licoes/03-neuroevolucao/treinar.py")
        return
    with open(arq, "rb") as f:
        campeoes = pickle.load(f)
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        os.path.join(PASTA, "config-neat.txt"),
    )
    idx = args.geracao % len(campeoes)

    pygame.init()
    tela = pygame.display.set_mode((LARGURA + PAINEL, ALTURA))
    fonte = pygame.font.SysFont("monospace", 14)
    relogio = pygame.time.Clock()

    def preparar(i):
        pygame.display.set_caption(f"Dino IA — campeão da geração {i}")
        rede = neat.nn.FeedForwardNetwork.create(campeoes[i], config)
        jogo = Jogo(1, semente=random.randint(0, 10**9), cores=[(40, 120, 220)])
        return rede, jogo

    rede, jogo = preparar(idx)
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_r:
                    rede, jogo = preparar(idx)
                elif e.key == pygame.K_RIGHT:
                    idx = min(idx + 1, len(campeoes) - 1)
                    rede, jogo = preparar(idx)
                elif e.key == pygame.K_LEFT:
                    idx = max(idx - 1, 0)
                    rede, jogo = preparar(idx)

        dino = jogo.dinos[0]
        entradas = jogo.sensores(dino)
        saidas = rede.activate(entradas)
        if dino.vivo:
            jogo.passo([ACOES[saidas.index(max(saidas))]])

        info = [f"Campeão da geração {idx} (de {len(campeoes) - 1})",
                f"Pontos: {dino.pontos}", f"Velocidade: {jogo.vel:.1f}",
                "← → troca de geração | R nova partida"]
        if not dino.vivo:
            info.append("Bateu!")
        jogo.desenhar(tela, fonte, info, destacar=dino)
        desenhar_rede(tela, fonte, campeoes[idx], config, entradas, saidas)
        pygame.display.flip()
        relogio.tick(60)


if __name__ == "__main__":
    main()
