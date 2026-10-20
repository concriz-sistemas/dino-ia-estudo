"""
relatorio.py — Lê resultados/historico.json e gera gráficos + RELATORIO.md.

Rode a partir da raiz do repositório, depois de treinar:
    python licoes/03-neuroevolucao/relatorio.py

O RELATORIO.md fica em resultados/ e abre direto no GitHub, com as imagens.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PASTA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PASTA)
from rede_util import desenhar_rede, nome_no

SAIDA = os.path.join(PASTA, "resultados")

# Explicação de cada parâmetro que vale a pena entender
EXPLICACOES = {
    ("NEAT", "pop_size"): "Quantos dinos (redes) existem em cada geração.",
    ("NEAT", "fitness_threshold"): "Nota considerada 'meta atingida'.",
    ("NEAT", "no_fitness_termination"): "Se True, roda todas as gerações pedidas mesmo após bater a meta.",
    ("DefaultGenome", "num_inputs"): "Quantos sensores a rede recebe.",
    ("DefaultGenome", "num_outputs"): "Quantas ações a rede pode escolher.",
    ("DefaultGenome", "num_hidden"): "Neurônios ocultos no início (o NEAT cria mais se precisar).",
    ("DefaultGenome", "initial_connection"): "'full' = toda entrada começa ligada a toda saída.",
    ("DefaultGenome", "activation_default"): "Função de ativação dos neurônios.",
    ("DefaultGenome", "weight_init_stdev"): "Quão espalhados são os pesos aleatórios iniciais.",
    ("DefaultGenome", "weight_mutate_rate"): "Chance de cada peso sofrer mutação a cada geração.",
    ("DefaultGenome", "weight_mutate_power"): "Tamanho típico da mutação (um 'empurrão' aleatório no peso).",
    ("DefaultGenome", "weight_replace_rate"): "Chance de um peso ser trocado por um valor totalmente novo.",
    ("DefaultGenome", "bias_mutate_rate"): "Chance de cada viés sofrer mutação.",
    ("DefaultGenome", "conn_add_prob"): "Chance de nascer uma conexão nova.",
    ("DefaultGenome", "conn_delete_prob"): "Chance de uma conexão ser removida.",
    ("DefaultGenome", "node_add_prob"): "Chance de nascer um neurônio novo (no meio de uma conexão).",
    ("DefaultGenome", "node_delete_prob"): "Chance de um neurônio ser removido.",
    ("DefaultSpeciesSet", "compatibility_threshold"): "Quão diferentes duas redes precisam ser para virarem espécies distintas.",
    ("DefaultStagnation", "max_stagnation"): "Gerações sem melhorar até uma espécie ser extinta.",
    ("DefaultReproduction", "elitism"): "Quantos melhores de cada espécie passam intactos para a próxima geração.",
    ("DefaultReproduction", "survival_threshold"): "Fração de cada espécie que pode ter 'filhos'.",
}


def salvar(fig, nome):
    fig.tight_layout()
    fig.savefig(os.path.join(SAIDA, nome), dpi=110)
    plt.close(fig)


def plot_fitness(G, limite):
    x = [g["geracao"] for g in G]
    melhor = np.array([g["melhor"] for g in G])
    media = np.array([g["media"] for g in G])
    desvio = np.array([g["desvio"] for g in G])
    mediana = np.array([g["mediana"] for g in G])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(x, np.maximum(0, media - desvio), media + desvio, color="#74c0fc", alpha=0.25,
                    label="média ± desvio")
    ax.plot(x, melhor, "o-", c="#e8590c", label="melhor dino")
    ax.plot(x, media, "o-", c="#1c7ed6", label="média da população")
    ax.plot(x, mediana, "--", c="#1c7ed6", alpha=0.6, label="mediana")
    ax.axhline(limite, c="gray", ls=":", label=f"limite da partida ({limite})")
    ax.set_xlabel("geração")
    ax.set_ylabel("pontos (quadros sobrevividos)")
    ax.set_title("Evolução da nota (fitness)")
    ax.grid(alpha=0.3)
    ax.legend()
    salvar(fig, "01_fitness.png")


def plot_especies(G):
    ids = sorted({e["id"] for g in G for e in g["especies"]})
    x = [g["geracao"] for g in G]
    mat = np.zeros((len(ids), len(G)))
    for j, g in enumerate(G):
        for e in g["especies"]:
            mat[ids.index(e["id"]), j] = e["tamanho"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stackplot(x, mat, labels=[f"espécie {i}" for i in ids], alpha=0.85)
    ax.set_xlabel("geração")
    ax.set_ylabel("dinos")
    ax.set_title("Espécies: grupos de redes parecidas que competem entre si")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.set_xlim(x[0], x[-1])
    salvar(fig, "02_especies.png")


def plot_complexidade(G):
    x = [g["geracao"] for g in G]
    ocultos = [sum(1 for k in g["campeao"]["nos"] if int(k) >= 3) for g in G]
    conexoes = [sum(1 for c in g["campeao"]["conexoes"] if c[3]) for g in G]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(x, conexoes, "o-", c="#2f9e44", label="conexões ativas")
    ax.plot(x, ocultos, "s-", c="#7048e8", label="neurônios ocultos")
    ax.set_xlabel("geração")
    ax.set_title("Tamanho da rede do campeão de cada geração")
    ax.grid(alpha=0.3)
    ax.legend()
    salvar(fig, "03_complexidade.png")


def plot_pesos_campeao(G):
    """Para cada conexão do campeão final, o peso dela no campeão de cada geração."""
    final = [(a, b) for a, b, w, ativa in G[-1]["campeao"]["conexoes"] if ativa]
    x = [g["geracao"] for g in G]
    fig, ax = plt.subplots(figsize=(11, 5))
    for a, b in final:
        ys = []
        for g in G:
            pesos = {(c[0], c[1]): c[2] for c in g["campeao"]["conexoes"] if c[3]}
            ys.append(pesos.get((a, b), np.nan))
        ax.plot(x, ys, "o-", ms=3, lw=1.2, label=f"{nome_no(a)} → {nome_no(b)}")
    ax.axhline(0, c="k", lw=0.6)
    ax.set_xlabel("geração")
    ax.set_ylabel("peso")
    ax.set_title("Pesos das conexões do campeão final, ao longo das gerações\n"
                 "(buraco na linha = aquela conexão não existia no campeão daquela geração)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    salvar(fig, "04_pesos_campeao.png")


def plot_distribuicao(G):
    x = [g["geracao"] for g in G]
    dados = [g["pesos_populacao"] for g in G]
    fig, ax = plt.subplots(figsize=(11, 4))
    partes = ax.violinplot(dados, positions=x, widths=0.8, showmedians=True)
    for corpo in partes["bodies"]:
        corpo.set_facecolor("#91a7ff")
        corpo.set_alpha(0.6)
    ax.axhline(0, c="k", lw=0.6)
    ax.set_xlabel("geração")
    ax.set_ylabel("peso")
    ax.set_title("Distribuição de TODOS os pesos da população em cada geração")
    ax.grid(alpha=0.3, axis="y")
    salvar(fig, "05_distribuicao_pesos.png")


def plot_redes(G):
    n = len(G)
    escolhidas = sorted({0, n // 3, (2 * n) // 3, n - 1})
    fig, axs = plt.subplots(1, len(escolhidas), figsize=(4.6 * len(escolhidas), 4.2))
    axs = np.atleast_1d(axs)
    for ax, i in zip(axs, escolhidas):
        g = G[i]
        desenhar_rede(ax, g["campeao"], f"Campeão da geração {g['geracao']}\n{g['melhor']:.0f} pontos")
    fig.suptitle("Verde = peso positivo (incentiva), vermelho = negativo (inibe). Espessura = força.", fontsize=10)
    salvar(fig, "06_redes.png")


def tabela_parametros(param):
    linhas = ["| Seção | Parâmetro | Valor | O que faz |", "|---|---|---|---|"]
    for (sec, chave), texto in EXPLICACOES.items():
        if sec in param and chave in param[sec]:
            linhas.append(f"| `{sec}` | `{chave}` | {param[sec][chave]} | {texto} |")
    return "\n".join(linhas)


def tabela_campeao(c):
    linhas = ["| De | Para | Peso |", "|---|---|---|"]
    for a, b, w, ativa in sorted(c["conexoes"], key=lambda x: -abs(x[2])):
        if ativa:
            linhas.append(f"| {nome_no(a)} | {nome_no(b)} | {w:+.3f} |")
    return "\n".join(linhas)


def tabela_geracoes(G):
    linhas = ["| Geração | Melhor | Média | Mediana | Espécies | Conexões do campeão |",
              "|---|---|---|---|---|---|"]
    for g in G:
        con = sum(1 for c in g["campeao"]["conexoes"] if c[3])
        linhas.append(f"| {g['geracao']} | {g['melhor']:.0f} | {g['media']:.0f} | {g['mediana']:.0f} "
                      f"| {g['n_especies']} | {con} |")
    return "\n".join(linhas)


def main():
    arq = os.path.join(SAIDA, "historico.json")
    if not os.path.exists(arq):
        print("Ainda não há histórico. Rode primeiro: python licoes/03-neuroevolucao/treinar.py")
        return
    with open(arq, encoding="utf-8") as f:
        dados = json.load(f)
    G = dados["geracoes"]
    limite = dados["limite_pontos"]

    plot_fitness(G, limite)
    plot_especies(G)
    plot_complexidade(G)
    plot_pesos_campeao(G)
    plot_distribuicao(G)
    plot_redes(G)

    primeira_meta = next((g["geracao"] for g in G if g["melhor"] >= limite), None)
    ultima = G[-1]
    md = f"""# Relatório do treino — neuroevolução

> Gerado automaticamente por `relatorio.py` a partir de `historico.json`.
> Rode o treino de novo e este relatório muda — cada execução é uma evolução diferente.

## Resumo

| | |
|---|---|
| Gerações | {len(G)} |
| Dinos por geração | {dados['parametros']['NEAT']['pop_size']} |
| Primeira geração a atingir {limite} pontos | {primeira_meta if primeira_meta is not None else 'não atingiu'} |
| Média da população: geração 0 → última | {G[0]['media']:.0f} → {ultima['media']:.0f} |
| Campeão final: conexões / neurônios ocultos | {sum(1 for c in ultima['campeao']['conexoes'] if c[3])} / {sum(1 for k in ultima['campeao']['nos'] if int(k) >= 3)} |

## 1. A nota ao longo das gerações

![fitness](01_fitness.png)

O **melhor** sobe rápido (basta um dino com sorte). A **média** sobe mais devagar: é a população inteira herdando o que funciona.

## 2. Espécies

![espécies](02_especies.png)

O NEAT separa as redes em espécies, e cada uma compete principalmente dentro do próprio grupo. Isso protege inovações (uma conexão nova, que no início costuma piorar a rede) de serem eliminadas antes de terem chance de amadurecer.

## 3. Tamanho da rede

![complexidade](03_complexidade.png)

O NEAT começa com a rede mínima e só adiciona neurônios e conexões quando isso ajuda. Também remove o que não serve.

## 4. Os pesos se ajustando

![pesos do campeão](04_pesos_campeao.png)

Cada linha é uma conexão do campeão final, e mostra quanto ela valia no campeão de cada geração. Os campeões são indivíduos diferentes, mas parentes: genes (conexões) são herdados e vão sofrendo pequenas mutações. Note que **não existe gradiente aqui** — os pesos mudam por mutação aleatória + seleção dos melhores.

![distribuição dos pesos](05_distribuicao_pesos.png)

Todos os pesos de todos os dinos, por geração. Na geração 0 são aleatórios em torno de zero; com o tempo a distribuição muda de forma conforme a seleção favorece certos valores.

## 5. As redes dos campeões

![redes](06_redes.png)

## 6. O campeão final, peso por peso

Ordenado pela força (valor absoluto) do peso:

{tabela_campeao(ultima['campeao'])}

## 7. Parâmetros usados neste treino

{tabela_parametros(dados['parametros'])}

O arquivo completo está em [`config-neat.txt`](../config-neat.txt).

## 8. Geração por geração

{tabela_geracoes(G)}
"""
    with open(os.path.join(SAIDA, "RELATORIO.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Relatório gerado: {os.path.join(SAIDA, 'RELATORIO.md')}")


if __name__ == "__main__":
    main()
