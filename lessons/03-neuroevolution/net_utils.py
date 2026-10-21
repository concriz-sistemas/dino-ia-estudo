"""
rede_util.py — Funções de apoio usadas pelo treino, pelo relatório e pelo visualizador.

- genoma_para_dict(): transforma um genoma do NEAT num dicionário simples
  (nós com seus vieses, conexões com seus pesos). É isso que vai para o
  arquivo de histórico — dá para abrir e ler num editor de texto.
- desenhar_rede(): desenha uma rede com matplotlib.
"""
NOMES_ENTRADAS = ["distância", "largura", "altura", "elevação", "alt. dino", "velocidade"]
NOMES_SAIDAS = ["nada", "pular", "abaixar"]


def nome_no(chave):
    """No NEAT, entradas têm chaves negativas (-1..-6), saídas 0..2, ocultos 3+."""
    if chave < 0:
        return NOMES_ENTRADAS[-chave - 1]
    if chave < len(NOMES_SAIDAS):
        return NOMES_SAIDAS[chave]
    return f"oculto {chave}"


def genoma_para_dict(genoma):
    return {
        "id": genoma.key,
        "fitness": genoma.fitness,
        "nos": {str(k): round(n.bias, 4) for k, n in genoma.nodes.items()},
        "conexoes": [
            [a, b, round(c.weight, 4), bool(c.enabled)]
            for (a, b), c in sorted(genoma.connections.items())
        ],
    }


def desenhar_rede(ax, g, titulo=""):
    """Desenha um genoma (no formato de genoma_para_dict) num eixo do matplotlib."""
    entradas = list(range(-1, -len(NOMES_ENTRADAS) - 1, -1))
    saidas = list(range(len(NOMES_SAIDAS)))
    ocultos = sorted(int(k) for k in g["nos"] if int(k) not in saidas)

    pos = {}
    for i, k in enumerate(entradas):
        pos[k] = (0.0, 1 - (i + 0.5) / len(entradas))
    for i, k in enumerate(ocultos):
        pos[k] = (0.5, 1 - (i + 0.5) / max(1, len(ocultos)))
    for i, k in enumerate(saidas):
        pos[k] = (1.0, 1 - (i + 0.5) / len(saidas))

    for a, b, w, ativa in g["conexoes"]:
        if not ativa or a not in pos or b not in pos:
            continue
        cor = "#2f9e44" if w > 0 else "#e03131"
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color=cor, lw=min(5, 0.4 + abs(w)), alpha=0.7, zorder=1)

    for k in entradas:
        ax.scatter(*pos[k], s=220, c="#d0ebff", edgecolors="#1c7ed6", zorder=2)
        ax.text(pos[k][0] - 0.06, pos[k][1], nome_no(k), ha="right", va="center", fontsize=8)
    for k in ocultos:
        ax.scatter(*pos[k], s=180, c="#e5dbff", edgecolors="#7048e8", zorder=2)
    for k in saidas:
        ax.scatter(*pos[k], s=220, c="#fff3bf", edgecolors="#f08c00", zorder=2)
        ax.text(pos[k][0] + 0.06, pos[k][1], nome_no(k), ha="left", va="center", fontsize=8)

    ax.set_xlim(-0.45, 1.4)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")
    ax.set_title(titulo, fontsize=10)
