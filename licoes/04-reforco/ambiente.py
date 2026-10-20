"""
ambiente.py — O jogo do dino no formato padrão de aprendizado por reforço (Gymnasium).

Todo ambiente de RL responde a duas perguntas:
    reset()      -> começa uma partida nova e devolve a primeira OBSERVAÇÃO
    step(acao)   -> aplica a AÇÃO e devolve (observação, RECOMPENSA, acabou?, ...)

É o mesmo jogo da lição 2, com um dino só. Qualquer algoritmo de RL que
"fala Gymnasium" (PPO, DQN, A2C...) consegue jogar sem saber nada de dinos.
"""
import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dino.jogo import Jogo, LARGURA, ALTURA, NADA, PULAR, ABAIXAR

ACOES = [NADA, PULAR, ABAIXAR]
NOMES_ACOES = ["nada", "pular", "abaixar"]
LIMITE_PONTOS = 5000            # a partida acaba aqui, mesmo sem bater

# A RECOMPENSA é o único "professor" do RL. Mude e veja o que acontece!
RECOMPENSA_VIVO = 0.1           # por quadro sobrevivido
RECOMPENSA_BATER = -1.0         # ao bater
CUSTO_PULO = 0.0                # experimente 0.01: o dino aprende a não pular à toa

# Cada decisão do agente vale por 2 quadros do jogo (a DeepMind usava 4 no Atari).
# Menos decisões por partida = fica mais fácil ligar a ação à consequência.
REPETIR_ACAO = 2


class DinoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}  # 30 passos x 2 quadros = 60 quadros/s

    def __init__(self, render_mode=None):
        # OBSERVAÇÃO: os mesmos 6 sensores das lições anteriores
        self.observation_space = spaces.Box(low=-1.0, high=2.0, shape=(6,), dtype=np.float32)
        # AÇÃO: 0 = nada, 1 = pular, 2 = abaixar
        self.action_space = spaces.Discrete(3)
        self.render_mode = render_mode
        self.jogo = None
        self.tela = None
        self.fonte = None
        self.linhas_info = []

    def _observar(self):
        return np.array(self.jogo.sensores(self.jogo.dinos[0]), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Obstáculos diferentes a cada partida: o agente precisa GENERALIZAR, não decorar
        semente = int(self.np_random.integers(0, 2**31))
        self.jogo = Jogo(1, semente=semente, cores=[(40, 120, 220)])
        return self._observar(), {}

    def step(self, acao):
        dino = self.jogo.dinos[0]
        recompensa = 0.0
        for _ in range(REPETIR_ACAO):
            self.jogo.passo([ACOES[int(acao)]])
            if not dino.vivo or dino.pontos >= LIMITE_PONTOS:
                break
            recompensa += RECOMPENSA_VIVO
        bateu = not dino.vivo
        if bateu:
            recompensa += RECOMPENSA_BATER
        if acao == 1:
            recompensa -= CUSTO_PULO

        terminou = bateu                                # fim "de verdade": perdeu
        cortado = dino.pontos >= LIMITE_PONTOS          # fim por limite de tempo
        info = {"pontos": dino.pontos}
        if self.render_mode == "human":
            self.render()
        return self._observar(), recompensa, terminou, cortado, info

    def render(self):
        import pygame

        if self.tela is None:
            pygame.init()
            if self.render_mode == "human":
                self.tela = pygame.display.set_mode((LARGURA, ALTURA))
                pygame.display.set_caption("Dino IA — aprendizado por reforço")
            else:
                self.tela = pygame.Surface((LARGURA, ALTURA))
            self.fonte = pygame.font.SysFont("monospace", 16)
            self.relogio = pygame.time.Clock()
        dino = self.jogo.dinos[0]
        self.jogo.desenhar(self.tela, self.fonte,
                           [f"Pontos: {dino.pontos}", f"Velocidade: {self.jogo.vel:.1f}"] + self.linhas_info,
                           destacar=dino)
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.relogio.tick(self.metadata["render_fps"])
            return None
        return np.transpose(pygame.surfarray.array3d(self.tela), (1, 0, 2))

    def close(self):
        if self.tela is not None:
            import pygame
            pygame.quit()
            self.tela = None
