"""
relatorio.py — Transforma o registro do treino em gráficos e num RELATORIO.md.

Rode a partir da raiz do repositório, depois de treinar:
    python licoes/04-reforco/relatorio.py

Além de ler os CSVs, ele coloca cada modelo salvo para jogar 10 partidas novas
e pergunta a cada um: "nesta situação, qual a chance de você pular?".
"""
import csv
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO

PASTA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PASTA)
from ambiente import DinoEnv, NOMES_ACOES
import ambiente
import treinar

SAIDA = os.path.join(PASTA, "resultados")
PASTA_MODELOS = os.path.join(SAIDA, "modelos")
N_PARTIDAS_TESTE = 10


def ler_csv(nome):
    with open(os.path.join(SAIDA, nome)) as f:
        linhas = list(csv.reader(f))
    return linhas[0], np.array(linhas[1:], dtype=float)


def salvar(fig, nome):
    fig.tight_layout()
    fig.savefig(os.path.join(SAIDA, nome), dpi=110)
    plt.close(fig)


def listar_modelos():
    arqs = sorted(a for a in os.listdir(PASTA_MODELOS) if a.endswith(".zip"))
    return [(int(a[4:9]) * 1000, os.path.join(PASTA_MODELOS, a)) for a in arqs]


def jogar(modelo, semente):
    env = DinoEnv()
    obs, _ = env.reset(seed=semente)
    while True:
        acao, _ = modelo.predict(obs, deterministic=True)
        obs, _, terminou, cortado, info = env.step(acao)
        if terminou or cortado:
            return info["pontos"]


def probabilidades(modelo, observacoes):
    """A política devolve uma probabilidade para cada ação. É isso que a rede 'pensa'."""
    with torch.no_grad():
        obs = torch.as_tensor(observacoes, dtype=torch.float32)
        return modelo.policy.get_distribution(obs).distribution.probs.numpy()


# Situações de teste: o mesmo obstáculo a várias distâncias e velocidades
DISTANCIAS = np.linspace(-0.05, 0.6, 66)
VELOCIDADES = np.linspace(0.4, 1.0, 31)
CACTO = (20 / 60, 40 / 60, 0.0)
PASSARO = (40 / 60, 25 / 60, 35 / 60)   # na altura da cabeça do dino
CORES_ACOES = ["#dee2e6", "#e03131", "#1971c2"]  # nada, pular, abaixar


def mapa_de_decisao(modelo, obstaculo):
    """Probabilidade de cada ação, para o obstáculo em cada distância e velocidade."""
    larg, alt, elev = obstaculo
    D, V = np.meshgrid(DISTANCIAS, VELOCIDADES)
    obs = np.column_stack([D.ravel(), np.full(D.size, larg), np.full(D.size, alt),
                           np.full(D.size, elev), np.zeros(D.size), V.ravel()])
    return probabilidades(modelo, obs).reshape(*D.shape, 3)


def numero(n):
    """1234567 -> '1.234.567'"""
    return f"{int(n):,}".replace(",", ".")


def media_movel(x, n=50):
    return np.convolve(x, np.ones(n) / n, mode="valid")


def main():
    _, ep = ler_csv("episodios.csv")
    cab, at = ler_csv("atualizacoes.csv")
    modelos = [(passos, PPO.load(arq, device="cpu")) for passos, arq in listar_modelos()]

    # 1) pontos por partida ao longo do treino
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.scatter(ep[:, 0], ep[:, 1], s=3, alpha=0.15, c="#868e96", label="cada partida")
    ax.plot(ep[49:, 0], media_movel(ep[:, 1]), c="#1c7ed6", lw=2, label="média das últimas 50")
    ax.set_xlabel("passos de treino")
    ax.set_ylabel("pontos")
    ax.set_title("Pontos por partida durante o treino")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    salvar(fig, "01_pontos.png")

    # 2) o que acontece dentro do PPO
    col = {nome: i for i, nome in enumerate(cab)}
    paineis = [
        ("entropia", "Entropia: o quão indeciso o agente está", "#ae3ec9"),
        ("erro_critico", "Erro do crítico ao prever a recompensa", "#e8590c"),
        ("kl_aprox", "Quanto a política muda a cada atualização (KL)", "#2f9e44"),
        ("fracao_clip", "Fração dos passos freados pelo clip do PPO", "#1c7ed6"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (nome, titulo, cor) in zip(axs.ravel(), paineis):
        ax.plot(at[:, 0], at[:, col[nome]], c=cor, lw=1.3)
        ax.set_title(titulo, fontsize=10)
        ax.set_xlabel("passos de treino")
        ax.grid(alpha=0.3)
    salvar(fig, "02_ppo.png")

    # 3) cada modelo salvo jogando partidas que nunca viu
    print(f"Testando {len(modelos)} modelos em {N_PARTIDAS_TESTE} partidas novas cada...")
    avaliacao = []
    for passos, modelo in modelos:
        pts = [jogar(modelo, 10_000 + s) for s in range(N_PARTIDAS_TESTE)]
        avaliacao.append((passos, pts))
        print(f"  {passos:>9,} passos: média {np.mean(pts):6.0f}".replace(",", "."))
    fig, ax = plt.subplots(figsize=(9, 4))
    rotulos = [f"{p // 1000}k" for p, _ in avaliacao]
    ax.bar(rotulos, [np.mean(p) for _, p in avaliacao], color="#1c7ed6", alpha=0.8)
    for i, (_, pts) in enumerate(avaliacao):
        ax.scatter([i] * len(pts), pts, c="k", s=8, zorder=3)
    ax.set_xlabel("modelo salvo depois de N passos de treino")
    ax.set_ylabel("pontos")
    ax.set_title(f"Cada modelo jogando {N_PARTIDAS_TESTE} partidas novas (pontos = cada partida)")
    ax.grid(alpha=0.3, axis="y")
    salvar(fig, "03_avaliacao.png")

    # 4) mapa de decisão: o que cada modelo faz em cada situação
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    escolhidos = [modelos[i] for i in sorted({0, len(modelos) // 3, 2 * len(modelos) // 3, len(modelos) - 1})]
    extensao = (DISTANCIAS[0], DISTANCIAS[-1], VELOCIDADES[0], VELOCIDADES[-1])
    fig, axs = plt.subplots(2, len(escolhidos), figsize=(4 * len(escolhidos), 8),
                            squeeze=False, layout="constrained")
    for c, (passos, modelo) in enumerate(escolhidos):
        cacto = mapa_de_decisao(modelo, CACTO)
        im = axs[0, c].imshow(cacto[..., 1], origin="lower", aspect="auto", cmap="RdBu_r",
                              vmin=0, vmax=1, extent=extensao)
        axs[0, c].set_title(f"{passos // 1000}k passos\nchance de pular (cacto)", fontsize=9)
        axs[0, c].set_xlabel("distância até o cacto", fontsize=8)

        passaro = mapa_de_decisao(modelo, PASSARO).argmax(axis=-1)
        axs[1, c].imshow(passaro, origin="lower", aspect="auto", cmap=ListedColormap(CORES_ACOES),
                         vmin=0, vmax=2, extent=extensao, interpolation="nearest")
        axs[1, c].set_title(f"{passos // 1000}k passos\nação escolhida (pássaro na cabeça)", fontsize=9)
        axs[1, c].set_xlabel("distância até o pássaro", fontsize=8)
    for linha in range(2):
        axs[linha, 0].set_ylabel("velocidade do jogo")
    fig.colorbar(im, ax=axs[0, :], shrink=0.8, label="probabilidade de pular")
    axs[1, -1].legend(handles=[Patch(color=cor, label=nome) for cor, nome in zip(CORES_ACOES, NOMES_ACOES)],
                      loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.savefig(os.path.join(SAIDA, "04_decisoes.png"), dpi=110)
    plt.close(fig)

    # 5) RELATORIO.md
    final = avaliacao[-1][1]
    primeira_boa = next((p for p, pts in avaliacao if np.mean(pts) >= 4000), None)
    ini, fim = ep[:200, 1].mean(), ep[-200:, 1].mean()
    with open(os.path.join(SAIDA, "RELATORIO.md"), "w", encoding="utf-8") as f:
        f.write(f"""# Relatório do treino — aprendizado por reforço (PPO)

> Gerado automaticamente por `relatorio.py` a partir de `episodios.csv`, `atualizacoes.csv` e dos modelos em `modelos/`.
> Rode o treino de novo e este relatório muda.

## Resumo

| | |
|---|---|
| Passos de treino | {numero(ep[-1, 0])} |
| Partidas jogadas no treino | {numero(len(ep))} |
| Média de pontos: 200 primeiras → 200 últimas partidas | {ini:.0f} → {fim:.0f} |
| Primeiro modelo salvo com média ≥ 4000 em partidas novas | {f"{primeira_boa // 1000}k passos" if primeira_boa is not None else "nenhum"} |
| Modelo final em {N_PARTIDAS_TESTE} partidas novas | média {np.mean(final):.0f}, pior {min(final)}, melhor {max(final)} |

## 1. Pontos por partida

![pontos](01_pontos.png)

Repare no **platô**: por um bom tempo o agente não melhora, e de repente a média dispara. Ele descobriu que pular na hora certa rende mais do que pular sem parar.

## 2. Por dentro do PPO

![ppo](02_ppo.png)

- **Entropia** despenca junto com o salto dos pontos: no começo o agente escolhe quase ao acaso; quando descobre o que funciona, fica decidido. Ela não vai a zero porque, com o obstáculo longe, tanto faz a ação, e o bônus de entropia premia continuar explorando.
- **Erro do crítico** sobe enquanto o agente melhora (as partidas ficam longas e imprevisíveis) e cai quando ele passa a sobreviver quase sempre: aí fica fácil prever a recompensa.
- **KL** e **fração do clip** ficam pequenos o tempo todo: é o *clip* do PPO segurando cada atualização para a política não mudar demais de uma vez.

## 3. Cada modelo salvo em partidas novas

![avaliação](03_avaliacao.png)

| Modelo | Média | Pior | Melhor |
|---|---|---|---|
""")
        for passos, pts in avaliacao:
            f.write(f"| {passos // 1000}k passos | {np.mean(pts):.0f} | {min(pts)} | {max(pts)} |\n")
        f.write("""
## 4. O que a rede "pensa"

![decisões](04_decisoes.png)

Cada ponto é uma pergunta feita à rede: "com o obstáculo a esta distância e o jogo nesta velocidade, o que você faz?".

- **Em cima, o cacto:** no começo a rede responde igual para tudo. No fim aparece uma **fronteira inclinada**: pular quando o cacto chega perto, e mais cedo quanto mais rápido o jogo. É a mesma ideia da faixa da lição 1, mas lá a rede recebeu exemplos prontos; aqui, só a recompensa.
- **Embaixo, o pássaro na altura da cabeça:** a cor mostra a ação mais provável. Repare que a rede não precisa escolher o que nós escolheríamos: pular por cima do pássaro também funciona. A recompensa só diz se deu certo, não como.
""")
    print(f"Relatório gerado: {os.path.join(SAIDA, 'RELATORIO.md')}")


if __name__ == "__main__":
    main()
