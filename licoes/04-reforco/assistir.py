"""
assistir.py — O dino treinado com PPO jogando, com a "cabeça" dele ao lado.

Rode a partir da raiz do repositório, depois de treinar:
    # o modelo final
    python licoes/04-reforco/assistir.py
    # o modelo antes de treinar (pesos aleatórios)
    python licoes/04-reforco/assistir.py --modelo 0
    # grava resultados/modelos.gif: vários modelos na mesma partida
    python licoes/04-reforco/assistir.py --gif

Teclas: ← / → = modelo anterior / seguinte, R = nova partida, ESC = sair

No painel da direita:
    - a chance que a rede dá para cada ação agora (a maior é a escolhida);
    - o VALOR: quanto de recompensa o crítico espera ganhar daqui para frente.
      Repare como ele cai quando um obstáculo difícil se aproxima.
"""
import argparse
import os
import random
import sys

PASTA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PASTA)

import pygame
import torch
from stable_baselines3 import PPO

from ambiente import DinoEnv, NOMES_ACOES, LARGURA, ALTURA

PASTA_MODELOS = os.path.join(PASTA, "resultados", "modelos")
PAINEL = 300
CORES_ACOES = [(150, 150, 150), (224, 49, 49), (25, 113, 194)]


def listar_modelos():
    arqs = sorted(a for a in os.listdir(PASTA_MODELOS) if a.endswith(".zip"))
    return [(int(a[4:9]), os.path.join(PASTA_MODELOS, a)) for a in arqs]


def pensar(modelo, obs):
    """Pergunta à rede: probabilidade de cada ação e valor esperado deste momento."""
    with torch.no_grad():
        t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        probs = modelo.policy.get_distribution(t).distribution.probs[0].numpy()
        valor = float(modelo.policy.predict_values(t)[0, 0])
    return probs, valor


def desenhar_painel(tela, fonte, probs, valor, historico_valor):
    x0 = LARGURA
    pygame.draw.rect(tela, (235, 235, 240), (x0, 0, PAINEL, ALTURA))
    tela.blit(fonte.render("Chance de cada ação", True, (40, 40, 40)), (x0 + 15, 12))
    escolha = int(probs.argmax())
    for i, (nome, p) in enumerate(zip(NOMES_ACOES, probs)):
        y = 40 + i * 30
        tela.blit(fonte.render(nome, True, (40, 40, 40)), (x0 + 15, y))
        pygame.draw.rect(tela, (210, 210, 215), (x0 + 95, y, 150, 18))
        pygame.draw.rect(tela, CORES_ACOES[i], (x0 + 95, y, int(150 * p), 18))
        if i == escolha:
            pygame.draw.rect(tela, (255, 190, 0), (x0 + 93, y - 2, 154, 22), 2)
        tela.blit(fonte.render(f"{p:4.0%}", True, (40, 40, 40)), (x0 + 252, y))

    tela.blit(fonte.render(f"Valor esperado: {valor:5.1f}", True, (40, 40, 40)), (x0 + 15, 145))
    # Gráfico do valor nos últimos quadros
    gx, gy, gw, gh = x0 + 15, 170, PAINEL - 30, 110
    pygame.draw.rect(tela, (250, 250, 252), (gx, gy, gw, gh))
    if len(historico_valor) > 1:
        # Escala ajustada aos últimos quadros, para as quedas aparecerem
        vmax = max(historico_valor) + 0.5
        vmin = min(historico_valor) - 0.5
        pts = [(gx + i * gw / (len(historico_valor) - 1),
                gy + gh - (v - vmin) / (vmax - vmin + 1e-9) * gh) for i, v in enumerate(historico_valor)]
        pygame.draw.lines(tela, (174, 62, 201), False, pts, 2)


def preparar(env, modelos, idx, semente):
    passos, arq = modelos[idx]
    modelo = PPO.load(arq, device="cpu")
    obs, _ = env.reset(seed=semente)
    return modelo, obs, f"Modelo com {passos}k passos de treino"


def gravar_gif(modelos, arquivo):
    """Alguns modelos jogando a MESMA partida, um depois do outro."""
    from PIL import Image

    pygame.init()
    tela = pygame.Surface((LARGURA + PAINEL, ALTURA))
    fonte = pygame.font.SysFont("monospace", 14)
    escolhidos = sorted({0, len(modelos) // 3, 2 * len(modelos) // 3, len(modelos) - 1})
    env = DinoEnv()
    quadros = []
    for idx in escolhidos:
        modelo, obs, titulo = preparar(env, modelos, idx, semente=7)
        historico, depois_de_bater, passo = [], 0, 0
        while passo < 350 and depois_de_bater < 15:
            probs, valor = pensar(modelo, obs)
            historico = (historico + [valor])[-120:]
            dino = env.jogo.dinos[0]
            if dino.vivo:
                obs, _, _, _, _ = env.step(int(probs.argmax()))
                passo += 1
            else:
                depois_de_bater += 1
            if passo % 2 == 0 or not dino.vivo:
                info = [titulo, f"Pontos: {dino.pontos}"] + ([] if dino.vivo else ["Bateu!"])
                env.jogo.desenhar(tela, fonte, info, destacar=dino)
                desenhar_painel(tela, fonte, probs, valor, historico)
                img = Image.frombytes("RGB", tela.get_size(), pygame.image.tobytes(tela, "RGB"))
                quadros.append(img.resize((img.width * 3 // 5, img.height * 3 // 5)))
        print(f"  {titulo}: {env.jogo.dinos[0].pontos} pontos")
    quadros[0].save(arquivo, save_all=True, append_images=quadros[1:], duration=66, loop=0, optimize=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", type=int, default=-1,
                   help="qual modelo salvo, em milhares de passos (padrão: o último)")
    p.add_argument("--gif", action="store_true", help="grava resultados/modelos.gif em vez de abrir a janela")
    args = p.parse_args()

    if not os.path.isdir(PASTA_MODELOS) or not listar_modelos():
        print("Nenhum modelo salvo ainda. Rode primeiro: python licoes/04-reforco/treinar.py")
        return
    modelos = listar_modelos()
    if args.gif:
        arquivo = os.path.join(PASTA, "resultados", "modelos.gif")
        gravar_gif(modelos, arquivo)
        print(f"Animação salva em: {arquivo}")
        return

    idx = len(modelos) - 1
    if args.modelo >= 0:
        idx = min(range(len(modelos)), key=lambda i: abs(modelos[i][0] - args.modelo))

    pygame.init()
    tela = pygame.display.set_mode((LARGURA + PAINEL, ALTURA))
    pygame.display.set_caption("Dino IA — aprendizado por reforço")
    fonte = pygame.font.SysFont("monospace", 14)
    relogio = pygame.time.Clock()
    env = DinoEnv()
    modelo, obs, titulo = preparar(env, modelos, idx, random.randint(0, 10**9))
    historico = []

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_r, pygame.K_LEFT, pygame.K_RIGHT):
                if e.key == pygame.K_RIGHT:
                    idx = min(idx + 1, len(modelos) - 1)
                elif e.key == pygame.K_LEFT:
                    idx = max(idx - 1, 0)
                modelo, obs, titulo = preparar(env, modelos, idx, random.randint(0, 10**9))
                historico = []

        probs, valor = pensar(modelo, obs)
        historico = (historico + [valor])[-120:]
        dino = env.jogo.dinos[0]
        if dino.vivo and dino.pontos < 5000:
            obs, _, _, _, _ = env.step(int(probs.argmax()))

        info = [titulo, f"Pontos: {dino.pontos}", "← → troca de modelo | R nova partida"]
        if not dino.vivo:
            info.append("Bateu!")
        env.jogo.desenhar(tela, fonte, info, destacar=dino)
        desenhar_painel(tela, fonte, probs, valor, historico)
        pygame.display.flip()
        relogio.tick(DinoEnv.metadata["render_fps"])


if __name__ == "__main__":
    main()
