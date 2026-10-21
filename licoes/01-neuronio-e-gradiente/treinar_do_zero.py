"""
Lição 1 — Uma rede neural do zero, só com numpy.

Objetivo: ver com os próprios olhos o que é "treinar com gradiente".
Nada de bibliotecas de IA aqui: cada conta está escrita à mão.

O problema (inspirado no dino):
    Entradas: distância até o cacto (0..1) e velocidade do jogo (0..1)
    Saída:    probabilidade de "agora é hora de pular"

    Regra que queremos que a rede DESCUBRA sozinha:
    é hora de pular se o cacto está numa FAIXA de distância —
    nem perto demais (já era), nem longe demais (cedo demais).
    E quanto mais rápido o jogo, mais longe essa faixa fica.

Rode a partir da raiz do repositório:
    # treina e salva os gráficos
    python licoes/01-neuronio-e-gradiente/treinar_do_zero.py
    # treina e depois toca o treino numa janela, como um vídeo
    python licoes/01-neuronio-e-gradiente/treinar_do_zero.py --ao-vivo
    # grava o mesmo replay em resultados/treino.gif
    python licoes/01-neuronio-e-gradiente/treinar_do_zero.py --gif

No replay:  ESPAÇO = pausar/continuar,  ← → = voltar/avançar uma época,
            ↑ ↓ = mais rápido/mais devagar,  R = recomeçar

Saídas (em resultados/):
    historico_pesos.csv   — todos os pesos, a perda e a acurácia em cada época
    01_dados.png          — os exemplos de treino
    02_perda.png          — a curva de aprendizado
    03_pesos.png          — como cada peso foi mudando
    04_fronteiras.png     — o que a rede "pensa" em diferentes momentos do treino
    05_neuronio_unico.png — por que precisamos de uma camada oculta
"""
import csv
import os
import sys
import warnings

import numpy as np
import matplotlib

AO_VIVO = "--ao-vivo" in sys.argv
GIF = "--gif" in sys.argv
if not AO_VIVO:
    matplotlib.use("Agg")  # gera PNGs sem precisar de janela
import matplotlib.pyplot as plt

PASTA = os.path.dirname(os.path.abspath(__file__))
SAIDA = os.path.join(PASTA, "resultados")
os.makedirs(SAIDA, exist_ok=True)

# ---------------------------------------------------------------------------
# HIPERPARÂMETROS — as "regulagens" do treino. Mude e rode de novo!
# ---------------------------------------------------------------------------
SEMENTE = 42            # para o resultado ser reproduzível
N_EXEMPLOS = 600        # quantos exemplos de treino
N_OCULTOS = 6           # neurônios na camada oculta
TAXA_APRENDIZADO = 0.5  # tamanho do passo na direção contrária ao gradiente
EPOCAS = 4000           # quantas vezes a rede vê todos os exemplos

rng = np.random.default_rng(SEMENTE)


# ---------------------------------------------------------------------------
# 1) DADOS
# ---------------------------------------------------------------------------
def regra_verdadeira(dist, vel):
    """A 'resposta certa'. A rede nunca vê esta função — só os exemplos."""
    inicio = 0.10 + 0.25 * vel
    fim = 0.30 + 0.35 * vel
    return ((dist > inicio) & (dist < fim)).astype(float)


def gerar_dados(n):
    dist = rng.uniform(0, 1, n)
    vel = rng.uniform(0, 1, n)
    X = np.column_stack([dist, vel])
    y = regra_verdadeira(dist, vel).reshape(-1, 1)
    return X, y


def normalizar(X):
    """Leva 0..1 para -1..1. Redes treinam melhor com entradas centradas no zero."""
    return X * 2 - 1


# ---------------------------------------------------------------------------
# 2) A REDE: entrada(2) -> oculta(N, tanh) -> saída(1, sigmoide)
# ---------------------------------------------------------------------------
def sigmoide(z):
    return 1 / (1 + np.exp(-z))


def iniciar_pesos(n_entradas, n_ocultos):
    # Pesos pequenos e aleatórios. Se começassem todos iguais (ex.: zero),
    # todos os neurônios ocultos aprenderiam exatamente a mesma coisa.
    return {
        "W1": rng.normal(0, 1, (n_entradas, n_ocultos)),
        "b1": np.zeros((1, n_ocultos)),
        "W2": rng.normal(0, 1, (n_ocultos, 1)),
        "b2": np.zeros((1, 1)),
    }


def forward(p, X):
    """Passagem para frente: das entradas até a previsão."""
    z1 = X @ p["W1"] + p["b1"]     # soma ponderada na camada oculta
    h = np.tanh(z1)                # ativação (dá a não-linearidade)
    z2 = h @ p["W2"] + p["b2"]     # soma ponderada na saída
    y_hat = sigmoide(z2)           # vira probabilidade 0..1
    return y_hat, h


def perda(y_hat, y):
    """Entropia cruzada binária: castiga muito quem erra com convicção."""
    eps = 1e-9
    return float(-np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)))


def backward(p, X, y, y_hat, h):
    """
    Retropropagação (backpropagation): regra da cadeia, da saída para a entrada.
    Responde: 'se eu mexer um pouquinho neste peso, quanto a perda muda?'
    Isso é o GRADIENTE.
    """
    n = X.shape[0]
    dz2 = (y_hat - y) / n                  # derivada da perda em relação a z2
    dW2 = h.T @ dz2
    db2 = dz2.sum(axis=0, keepdims=True)
    dh = dz2 @ p["W2"].T                   # o erro "volta" para a camada oculta
    dz1 = dh * (1 - h ** 2)                # derivada da tanh
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0, keepdims=True)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def nomes_dos_pesos(p):
    nomes = []
    for k in ["W1", "b1", "W2", "b2"]:
        for idx in np.ndindex(p[k].shape):
            nomes.append(f"{k}[{','.join(map(str, idx))}]")
    return nomes


def achatar(p):
    return np.concatenate([p[k].ravel() for k in ["W1", "b1", "W2", "b2"]])


# ---------------------------------------------------------------------------
# 3) CONFERINDO O GRADIENTE (gradient check)
# ---------------------------------------------------------------------------
def checar_gradiente(p, X, y):
    """
    Compara o gradiente da fórmula (backward) com uma estimativa 'na força bruta':
    mexe o peso +eps e -eps e mede a diferença na perda. Se baterem, a
    retropropagação está certa. É assim que se depura backprop na vida real.
    """
    y_hat, h = forward(p, X)
    g = backward(p, X, y, y_hat, h)
    eps = 1e-5
    print("\n[Conferindo o gradiente]  fórmula  vs  força bruta")
    for k, idx in [("W1", (0, 0)), ("W1", (1, 2)), ("W2", (3, 0)), ("b2", (0, 0))]:
        original = p[k][idx]
        p[k][idx] = original + eps
        l_mais = perda(forward(p, X)[0], y)
        p[k][idx] = original - eps
        l_menos = perda(forward(p, X)[0], y)
        p[k][idx] = original
        numerico = (l_mais - l_menos) / (2 * eps)
        print(f"  {k}{list(idx)}:  {g[k][idx]:+.6f}  vs  {numerico:+.6f}")


# ---------------------------------------------------------------------------
# 4) TREINO
# ---------------------------------------------------------------------------
def treinar(X, y, n_ocultos, epocas, taxa, registrar=True, explicar=True):
    p = iniciar_pesos(X.shape[1], n_ocultos)
    historico = []
    fotos = {}  # cópias dos pesos em alguns momentos, para desenhar depois
    momentos = {0, 50, 300, epocas}

    for epoca in range(epocas + 1):
        y_hat, h = forward(p, X)
        L = perda(y_hat, y)
        acc = float(np.mean((y_hat > 0.5) == y))
        g = backward(p, X, y, y_hat, h)
        norma_grad = float(np.sqrt(sum((g[k] ** 2).sum() for k in g)))

        if registrar:
            historico.append([epoca, L, acc, norma_grad] + achatar(p).tolist())
        if epoca in momentos:
            fotos[epoca] = {k: v.copy() for k, v in p.items()}

        if explicar and epoca < 3:
            w, gw = p["W2"][0, 0], g["W2"][0, 0]
            print(f"\nÉpoca {epoca}: perda={L:.4f}  acurácia={acc:.0%}")
            print(f"  exemplo com um peso, W2[0,0]:")
            print(f"    valor atual = {w:+.5f}")
            print(f"    gradiente   = {gw:+.5f}   (sinal + => aumentar o peso aumentaria a perda)")
            print(f"    novo valor  = {w:+.5f} - {taxa} × {gw:+.5f} = {w - taxa * gw:+.5f}")
        elif explicar and epoca % 500 == 0:
            print(f"Época {epoca:5d}: perda={L:.4f}  acurácia={acc:.1%}  |gradiente|={norma_grad:.4f}")

        if epoca == epocas:
            break
        # O PASSO DE DESCIDA DE GRADIENTE: andar um pouco "morro abaixo"
        for k in p:
            p[k] -= taxa * g[k]

    return p, historico, fotos


# ---------------------------------------------------------------------------
# 5) GRÁFICOS
# ---------------------------------------------------------------------------
def grade_de_previsao(p, resolucao=200):
    d = np.linspace(0, 1, resolucao)
    v = np.linspace(0, 1, resolucao)
    D, V = np.meshgrid(d, v)
    Xg = normalizar(np.column_stack([D.ravel(), V.ravel()]))
    prob = forward(p, Xg)[0].reshape(D.shape)
    return D, V, prob


def plot_dados(X, y):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[y[:, 0] == 0, 0], X[y[:, 0] == 0, 1], s=10, c="#9aa5b1", label="não pular")
    ax.scatter(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], s=10, c="#e8590c", label="pular")
    ax.set_xlabel("distância até o cacto")
    ax.set_ylabel("velocidade do jogo")
    ax.set_title("Exemplos de treino: quando é hora de pular?")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(SAIDA, "01_dados.png"), dpi=110)
    plt.close(fig)


def plot_perda(hist):
    h = np.array(hist)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].plot(h[:, 0], h[:, 1], c="#1c7ed6")
    axs[0].set_title("Perda (erro) — quanto menor, melhor")
    axs[0].set_xlabel("época")
    axs[1].plot(h[:, 0], h[:, 2] * 100, c="#2f9e44")
    axs[1].set_title("Acurácia (%)")
    axs[1].set_xlabel("época")
    axs[1].set_ylim(0, 100)
    axs[2].plot(h[:, 0], h[:, 3], c="#ae3ec9")
    axs[2].set_yscale("log")
    axs[2].set_title("Tamanho do gradiente (escala log)")
    axs[2].set_xlabel("época")
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(SAIDA, "02_perda.png"), dpi=110)
    plt.close(fig)


def plot_pesos(hist, nomes):
    h = np.array(hist)
    pesos = h[:, 4:]
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
    for i, nome in enumerate(nomes):
        ax = axs[0] if nome.startswith(("W1", "b1")) else axs[1]
        ax.plot(h[:, 0], pesos[:, i], lw=1.2, label=nome)
    axs[0].set_title("Camada oculta (W1 e b1)")
    axs[1].set_title("Camada de saída (W2 e b2)")
    for ax in axs:
        ax.set_xlabel("época")
        ax.set_ylabel("valor do peso")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Cada linha é um peso. Repare: mudam rápido no começo e depois se acomodam.")
    fig.tight_layout()
    fig.savefig(os.path.join(SAIDA, "03_pesos.png"), dpi=110)
    plt.close(fig)


def plot_fronteiras(fotos, X, y):
    epocas = sorted(fotos)
    fig, axs = plt.subplots(1, len(epocas), figsize=(4.2 * len(epocas), 4))
    for ax, ep in zip(axs, epocas):
        D, V, prob = grade_de_previsao(fotos[ep])
        ax.contourf(D, V, prob, levels=20, cmap="RdBu_r", vmin=0, vmax=1, alpha=0.8)
        ax.contour(D, V, prob, levels=[0.5], colors="k", linewidths=1.5)
        ax.scatter(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], s=4, c="k", alpha=0.4, label="gabarito: exemplos de 'pular'")
        ax.set_title(f"época {ep}")
        ax.set_xlabel("distância")
    axs[0].set_ylabel("velocidade")
    axs[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Fundo = o que a rede pensa (vermelho = 'pule!', azul = 'espere'). "
                 "Pontos = gabarito, fixo desde o início.")
    fig.tight_layout()
    fig.savefig(os.path.join(SAIDA, "04_fronteiras.png"), dpi=110)
    plt.close(fig)


def plot_neuronio_unico(X, y, Xn):
    """Um neurônio só traça UMA linha reta. A faixa precisa de duas — não dá."""
    # 'rede' sem camada oculta = regressão logística
    w = rng.normal(0, 1, (2, 1))
    b = np.zeros((1, 1))
    for _ in range(EPOCAS):
        y_hat = sigmoide(Xn @ w + b)
        dz = (y_hat - y) / len(y)
        w -= TAXA_APRENDIZADO * Xn.T @ dz
        b -= TAXA_APRENDIZADO * dz.sum(axis=0, keepdims=True)
    acc = float(np.mean((sigmoide(Xn @ w + b) > 0.5) == y))

    d = np.linspace(0, 1, 200)
    D, V = np.meshgrid(d, d)
    prob = sigmoide(normalizar(np.column_stack([D.ravel(), V.ravel()])) @ w + b).reshape(D.shape)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.contourf(D, V, prob, levels=20, cmap="RdBu_r", vmin=0, vmax=1, alpha=0.8)
    ax.contour(D, V, prob, levels=[0.5], colors="k")
    ax.scatter(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], s=4, c="k", alpha=0.4, label="gabarito: exemplos de 'pular'")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"Neurônio único: acurácia {acc:.0%}\n(só consegue uma linha reta)")
    ax.set_xlabel("distância")
    ax.set_ylabel("velocidade")
    fig.tight_layout()
    fig.savefig(os.path.join(SAIDA, "05_neuronio_unico.png"), dpi=110)
    plt.close(fig)
    return acc


# ---------------------------------------------------------------------------
# 6) REPLAY — o treino gravado, tocado como um vídeo
# ---------------------------------------------------------------------------
VELOCIDADES = [1, 2, 5, 10, 25, 50, 100]  # épocas avançadas por quadro


def desachatar(v, modelo):
    """O contrário de achatar(): volta a lista de pesos para o formato da rede."""
    p, i = {}, 0
    for k in ["W1", "b1", "W2", "b2"]:
        n = modelo[k].size
        p[k] = v[i:i + n].reshape(modelo[k].shape)
        i += n
    return p


class Replay:
    """Desenha o estado do treino em qualquer época gravada no histórico."""

    def __init__(self, hist, modelo, X, y, Xn):
        self.h = np.array(hist)
        self.modelo = modelo
        self.y, self.Xn = y, Xn
        self.ultima = int(self.h[-1, 0])
        nomes = nomes_dos_pesos(modelo)

        self.fig, axs = plt.subplots(2, 2, figsize=(13, 8))
        self.ax_rede, self.ax_perda, self.ax_pesos, self.ax_grad = axs.ravel()

        self.D, self.V, prob = grade_de_previsao(self.pesos(0), resolucao=80)
        self.imagem = self.ax_rede.imshow(prob, origin="lower", extent=(0, 1, 0, 1), aspect="auto",
                                          cmap="RdBu_r", vmin=0, vmax=1, alpha=0.85)
        self.contorno = None
        self.ax_rede.scatter(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], s=4, c="k", alpha=0.4, label="gabarito: exemplos de 'pular'")
        self.ax_rede.legend(loc="lower right", fontsize=8)
        self.ax_rede.set_xlabel("distância")
        self.ax_rede.set_ylabel("velocidade")
        self.ax_rede.set_title("O que a rede pensa: vermelho = 'pule!', azul = 'espere'")

        self.linha_perda, = self.ax_perda.plot([], [], c="#1c7ed6", label="perda")
        self.linha_acc, = self.ax_perda.plot([], [], c="#2f9e44", label="acurácia")
        self.ax_perda.set_xlim(0, self.ultima)
        self.ax_perda.set_ylim(0, max(1.05, self.h[:, 1].max() * 1.05))
        self.ax_perda.set_xlabel("época")
        self.ax_perda.set_title("Perda caindo, acurácia subindo")
        self.ax_perda.legend(loc="center right")
        self.ax_perda.grid(alpha=0.3)

        pesos = self.h[:, 4:]
        self.linhas_pesos = [self.ax_pesos.plot([], [], lw=1)[0] for _ in nomes]
        self.ax_pesos.set_xlim(0, self.ultima)
        self.ax_pesos.set_ylim(pesos.min() - 0.5, pesos.max() + 0.5)
        self.ax_pesos.set_xlabel("época")
        self.ax_pesos.set_title(f"Os {len(nomes)} pesos se ajustando (cada linha é um peso)")
        self.ax_pesos.grid(alpha=0.3)
        self.marcas = [self.ax_perda.axvline(0, c="k", lw=0.8, alpha=0.4),
                       self.ax_pesos.axvline(0, c="k", lw=0.8, alpha=0.4)]

        self.barras = self.ax_grad.bar(range(len(nomes)), np.zeros(len(nomes)))
        self.ax_grad.set_xticks(range(len(nomes)), nomes, rotation=90, fontsize=7)
        self.ax_grad.axhline(0, c="k", lw=0.8)

        self.fig.text(0.5, 0.005, "ESPAÇO pausa  ·  ← → uma época  ·  ↑ ↓ velocidade  ·  R recomeça",
                      ha="center", fontsize=9, color="#555")
        self.fig.tight_layout(rect=(0, 0.02, 1, 0.96))

    def pesos(self, epoca):
        return desachatar(self.h[epoca, 4:], self.modelo)

    def desenhar(self, epoca, status=""):
        h = self.h
        p = self.pesos(epoca)

        _, _, prob = grade_de_previsao(p, resolucao=80)
        self.imagem.set_data(prob)
        if self.contorno is not None:
            self.contorno.remove()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # sem fronteira ainda (tudo azul) gera um aviso
            self.contorno = self.ax_rede.contour(self.D, self.V, prob, levels=[0.5],
                                                 colors="k", linewidths=1.5)

        ate = slice(0, epoca + 1)
        self.linha_perda.set_data(h[ate, 0], h[ate, 1])
        self.linha_acc.set_data(h[ate, 0], h[ate, 2])
        for i, linha in enumerate(self.linhas_pesos):
            linha.set_data(h[ate, 0], h[ate, 4 + i])
        for marca in self.marcas:
            marca.set_xdata([epoca, epoca])

        # O gradiente não fica no histórico, mas é só recalcular com os pesos da época.
        y_hat, oculta = forward(p, self.Xn)
        grad = achatar(backward(p, self.Xn, self.y, y_hat, oculta))
        for barra, g in zip(self.barras, grad):
            barra.set_height(g)
            barra.set_color("#e03131" if g > 0 else "#1971c2")
        lim = max(np.abs(grad).max() * 1.1, 1e-9)
        self.ax_grad.set_ylim(-lim, lim)
        self.ax_grad.set_title(f"Gradiente de cada peso (o peso anda no sentido CONTRÁRIO)\n"
                               f"tamanho total = {h[epoca, 3]:.5f} — escala ajustada a cada quadro",
                               fontsize=10)

        self.fig.suptitle(f"Época {epoca}/{self.ultima}   perda={h[epoca, 1]:.4f}   "
                          f"acurácia={h[epoca, 2]:.1%}   {status}")


def tocar_replay(replay):
    for k in ("keymap.back", "keymap.forward", "keymap.home"):
        plt.rcParams[k] = []  # libera ← → R, que o matplotlib usa para navegar no gráfico
    estado = {"epoca": 0, "vel": 1, "pausado": False}

    def tecla(evento):
        if evento.key == " ":
            estado["pausado"] = not estado["pausado"]
        elif evento.key == "right":
            estado["epoca"] = min(estado["epoca"] + 1, replay.ultima)
            estado["pausado"] = True
        elif evento.key == "left":
            estado["epoca"] = max(estado["epoca"] - 1, 0)
            estado["pausado"] = True
        elif evento.key in ("up", "+", "="):
            estado["vel"] = min(estado["vel"] + 1, len(VELOCIDADES) - 1)
        elif evento.key in ("down", "-"):
            estado["vel"] = max(estado["vel"] - 1, 0)
        elif evento.key == "r":
            estado["epoca"], estado["pausado"] = 0, False

    replay.fig.canvas.mpl_connect("key_press_event", tecla)
    while plt.fignum_exists(replay.fig.number):
        vel = VELOCIDADES[estado["vel"]]
        status = "[PAUSADO]" if estado["pausado"] else f"[{vel} época(s) por quadro]"
        replay.desenhar(estado["epoca"], status)
        plt.pause(1 / 30)
        if not estado["pausado"]:
            if estado["epoca"] >= replay.ultima:
                estado["pausado"] = True
            else:
                estado["epoca"] = min(estado["epoca"] + vel, replay.ultima)


def gravar_gif(replay, arquivo):
    from matplotlib.animation import FuncAnimation, PillowWriter

    # Devagar no começo, onde tudo acontece; rápido no fim, onde quase nada muda.
    quadros = (list(range(0, 60, 2)) + list(range(60, 400, 10))
               + list(range(400, replay.ultima, 100)) + [replay.ultima] * 15)
    anim = FuncAnimation(replay.fig, replay.desenhar, frames=quadros)
    anim.save(arquivo, writer=PillowWriter(fps=12), dpi=55)


# ---------------------------------------------------------------------------
def main():
    X, y = gerar_dados(N_EXEMPLOS)
    Xn = normalizar(X)
    print(f"{N_EXEMPLOS} exemplos, {int(y.sum())} deles são 'pular'.")

    p_teste = iniciar_pesos(2, N_OCULTOS)
    checar_gradiente(p_teste, Xn, y)

    print("\n[Treinando a rede 2 -> %d -> 1]" % N_OCULTOS)
    p, hist, fotos = treinar(Xn, y, N_OCULTOS, EPOCAS, TAXA_APRENDIZADO)

    nomes = nomes_dos_pesos(p)
    with open(os.path.join(SAIDA, "historico_pesos.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoca", "perda", "acuracia", "norma_gradiente"] + nomes)
        for linha in hist:
            w.writerow([linha[0]] + [f"{v:.6f}" for v in linha[1:]])

    # Teste com dados NOVOS, que a rede nunca viu
    Xt, yt = gerar_dados(2000)
    acc_teste = float(np.mean((forward(p, normalizar(Xt))[0] > 0.5) == yt))

    plot_dados(X, y)
    plot_perda(hist)
    plot_pesos(hist, nomes)
    plot_fronteiras(fotos, X, y)
    acc_unico = plot_neuronio_unico(X, y, Xn)

    print("\n=== Resultado ===")
    print(f"Acurácia no treino:          {hist[-1][2]:.1%}")
    print(f"Acurácia em dados novos:     {acc_teste:.1%}")
    print(f"Neurônio único (comparação): {acc_unico:.1%}")
    print(f"Gráficos e CSV em: {SAIDA}")
    if GIF:
        arquivo = os.path.join(SAIDA, "treino.gif")
        print("Gravando a animação...")
        gravar_gif(Replay(hist, p, X, y, Xn), arquivo)
        print(f"Animação salva em: {arquivo}")
    if AO_VIVO:
        print("Abrindo o replay do treino. Feche a janela para sair.")
        tocar_replay(Replay(hist, p, X, y, Xn))
    elif not GIF:
        print("Para assistir o treino como um vídeo, rode de novo com --ao-vivo")


if __name__ == "__main__":
    main()
