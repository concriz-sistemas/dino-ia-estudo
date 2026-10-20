"""
Lição 3 — Treina dinossauros com neuroevolução (NEAT) e REGISTRA tudo.

Rode a partir da raiz do repositório:
    python licoes/03-neuroevolucao/treinar.py              # com janela, para assistir
    python licoes/03-neuroevolucao/treinar.py --rapido     # sem janela, bem mais rápido
    python licoes/03-neuroevolucao/treinar.py --geracoes 50

Teclas durante o treino (modo com janela):
    V   = liga/desliga o desenho (acelera muito)
    ESC = encerra e salva o que foi registrado até agora

O que é salvo em resultados/:
    historico.json  — para CADA geração: notas, espécies, o campeão completo
                      (todos os pesos) e os pesos de toda a população
    campeoes.pkl    — o campeão de cada geração, para assistir depois
Depois rode o relatório:
    python licoes/03-neuroevolucao/relatorio.py
"""
import argparse
import configparser
import json
import os
import pickle
import statistics
import sys

PASTA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(PASTA, "..", "..")))

import neat
import pygame

from dino.jogo import Jogo, LARGURA, ALTURA, NADA, PULAR, ABAIXAR
from rede_util import genoma_para_dict

ARQ_CONFIG = os.path.join(PASTA, "config-neat.txt")
SAIDA = os.path.join(PASTA, "resultados")
ARQ_HISTORICO = os.path.join(SAIDA, "historico.json")
ARQ_CAMPEOES = os.path.join(SAIDA, "campeoes.pkl")

# Teto de pontos por partida — sem isso, um dino perfeito jogaria para sempre
LIMITE_PONTOS = 5000
ACOES = [NADA, PULAR, ABAIXAR]


class TreinoInterrompido(Exception):
    pass


# ---------------------------------------------------------------------------
# O "caderno de anotações" do treino.
# O NEAT chama post_evaluate() ao fim de cada geração, já com as notas de todos.
# ---------------------------------------------------------------------------
class Registrador(neat.reporting.BaseReporter):
    def __init__(self):
        self.geracao = -1
        self.geracoes = []
        self.campeoes = []

    def start_generation(self, generation):
        self.geracao = generation

    def post_evaluate(self, config, population, species, best_genome):
        notas = [g.fitness for g in population.values()]
        # todos os pesos ativos de toda a população (para ver a distribuição)
        pesos = [round(c.weight, 3)
                 for g in population.values()
                 for c in g.connections.values() if c.enabled]
        especies = [
            {"id": sid, "tamanho": len(s.members),
             "melhor": max(population[k].fitness for k in s.members if k in population)}
            for sid, s in species.species.items()
        ]
        campeao = max(population.values(), key=lambda g: g.fitness)
        self.geracoes.append({
            "geracao": self.geracao,
            "melhor": max(notas),
            "media": round(statistics.mean(notas), 2),
            "mediana": statistics.median(notas),
            "desvio": round(statistics.pstdev(notas), 2),
            "n_especies": len(especies),
            "especies": especies,
            "campeao": genoma_para_dict(campeao),
            "pesos_populacao": pesos,
        })
        self.campeoes.append(pickle.loads(pickle.dumps(campeao)))  # cópia

    def salvar(self, config_path):
        os.makedirs(SAIDA, exist_ok=True)
        cp = configparser.ConfigParser()
        cp.read(config_path)
        dados = {
            "parametros": {sec: dict(cp[sec]) for sec in cp.sections()},
            "limite_pontos": LIMITE_PONTOS,
            "geracoes": self.geracoes,
        }
        with open(ARQ_HISTORICO, "w", encoding="utf-8") as f:
            json.dump(dados, f, ensure_ascii=False)
        with open(ARQ_CAMPEOES, "wb") as f:
            pickle.dump(self.campeoes, f)


# ---------------------------------------------------------------------------
# Avaliação: cada geração joga UMA partida, todos juntos
# ---------------------------------------------------------------------------
class Treinador:
    def __init__(self, mostrar):
        self.mostrar = mostrar
        self.desenhar = mostrar
        self.parar = False
        self.n = 0
        if mostrar:
            pygame.init()
            self.tela = pygame.display.set_mode((LARGURA, ALTURA))
            pygame.display.set_caption("Dino IA — treinando com NEAT")
            self.fonte = pygame.font.SysFont("monospace", 16)
            self.relogio = pygame.time.Clock()

    def _eventos(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                self.parar = True
            if e.type == pygame.KEYDOWN and e.key == pygame.K_v:
                self.desenhar = not self.desenhar

    def avaliar_genomas(self, genomas, config):
        if self.parar:
            raise TreinoInterrompido()
        self.n += 1
        # Mesma semente para todos da geração => todos enfrentam os mesmos obstáculos.
        # Semente diferente a cada geração => não dá para "decorar" a fase.
        jogo = Jogo(len(genomas), semente=self.n)
        redes = [neat.nn.FeedForwardNetwork.create(g, config) for _, g in genomas]

        while jogo.algum_vivo() and jogo.quadro < LIMITE_PONTOS and not self.parar:
            if self.mostrar:
                self._eventos()

            # 1) cada rede olha os sensores e escolhe uma ação (a maior saída vence)
            acoes = []
            for dino, rede in zip(jogo.dinos, redes):
                if not dino.vivo:
                    acoes.append(NADA)
                    continue
                saidas = rede.activate(jogo.sensores(dino))
                acoes.append(ACOES[saidas.index(max(saidas))])

            # 2) o jogo avança um quadro
            jogo.passo(acoes)

            # 3) desenha (opcional)
            if self.mostrar and self.desenhar:
                vivos = [d for d in jogo.dinos if d.vivo]
                jogo.desenhar(
                    self.tela, self.fonte,
                    [f"Geração: {self.n - 1}",
                     f"Vivos: {len(vivos)}/{len(jogo.dinos)}",
                     f"Pontos: {jogo.quadro}",
                     f"Velocidade: {jogo.vel:.1f}",
                     "V = acelerar | ESC = parar"],
                    destacar=vivos[0] if vivos else None,
                )
                pygame.display.flip()
                self.relogio.tick(60)
            elif self.mostrar and jogo.quadro % 200 == 0:
                pygame.display.set_caption(f"Geração {self.n - 1} — {jogo.quadro} pts (V para ver)")

        # 4) a nota (fitness) de cada genoma = quantos quadros o dino dele sobreviveu
        for (_, genoma), dino in zip(genomas, jogo.dinos):
            genoma.fitness = dino.pontos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rapido", action="store_true", help="treinar sem janela")
    p.add_argument("--geracoes", type=int, default=30)
    args = p.parse_args()

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        ARQ_CONFIG,
    )
    populacao = neat.Population(config)
    populacao.add_reporter(neat.StdOutReporter(False))
    registrador = Registrador()
    populacao.add_reporter(registrador)

    treinador = Treinador(mostrar=not args.rapido)
    try:
        populacao.run(treinador.avaliar_genomas, args.geracoes)
    except TreinoInterrompido:
        print("\nTreino interrompido — salvando o que já foi registrado.")

    if not registrador.geracoes:
        print("Nenhuma geração completa para salvar.")
        return
    registrador.salvar(ARQ_CONFIG)

    ultima = registrador.geracoes[-1]
    print("\n=== Treino concluído ===")
    print(f"Gerações:          {len(registrador.geracoes)}")
    print(f"Melhor na última:  {ultima['melhor']}   média: {ultima['media']}")
    print(f"Histórico salvo:   {ARQ_HISTORICO}")
    print("Próximos passos:")
    print("  python licoes/03-neuroevolucao/relatorio.py          # gráficos + RELATORIO.md")
    print("  python licoes/03-neuroevolucao/assistir.py           # campeão final jogando")
    print("  python licoes/03-neuroevolucao/assistir.py --geracao 0   # compare com o da geração 0!")

    if treinador.mostrar:
        pygame.quit()


if __name__ == "__main__":
    main()
