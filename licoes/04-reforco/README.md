# Lição 4 — Aprendizado por reforço (PPO)

> **Objetivo:** juntar o **gradiente** da lição 1 com a **nota** da lição 3. Um único dino joga milhares de partidas e ajusta os pesos da rede, por gradiente, para ganhar mais recompensa.
> É a mesma família de técnicas usada para ajustar LLMs depois do pré-treino.

![modelos jogando](resultados/modelos.gif)

*O mesmo dino com 0, 500 mil, 1 milhão e 2 milhões de passos de treino, na mesma partida. À direita, a chance que a rede dá para cada ação e o valor que ela espera ganhar.*

Rode um de cada vez, nesta ordem.

1. Treine (uns 2 minutos, sem janela):

```bash
python licoes/04-reforco/treinar.py
```

2. Gere os gráficos e o `RELATORIO.md`:

```bash
python licoes/04-reforco/relatorio.py
```

3. Assista ao modelo final jogando. As setas ← → trocam de modelo salvo:

```bash
python licoes/04-reforco/assistir.py
```

4. Compare com o modelo antes de treinar:

```bash
python licoes/04-reforco/assistir.py --modelo 0
```

Os modelos de um treino de exemplo já estão no repositório, então o passo 3 funciona mesmo sem treinar.

---

## 1. Três jeitos de aprender

| | Lição 1 — gradiente | Lição 3 — neuroevolução | Lição 4 — reforço |
|---|---|---|---|
| Quem joga | ninguém (exemplos prontos) | 50 dinos por geração | 1 dino, milhares de partidas |
| O que a IA recebe | a resposta certa de cada exemplo | uma nota no fim da partida | uma recompensa a **cada passo** |
| Como os pesos mudam | gradiente | mutação aleatória + seleção | gradiente |
| Precisa de alguém saber a resposta? | sim | não | não |

O reforço fica no meio: como na lição 3, ninguém diz qual é a jogada certa; como na lição 1, os pesos mudam por gradiente, sem sorteio de mutações.

---

## 2. O ambiente: observação, ação, recompensa

[`ambiente.py`](ambiente.py) embala o jogo no formato padrão **Gymnasium**, que qualquer algoritmo de reforço entende:

- **Observação:** os mesmos 6 sensores das lições anteriores.
- **Ação:** 0 = nada, 1 = pular, 2 = abaixar.
- **Recompensa:** `+0.1` por quadro vivo e `−1` ao bater. É o único "professor". O agente não sabe o que é um cacto; só sabe que alguns números na entrada antecedem a recompensa ruim.

Cada partida tem obstáculos sorteados de novo, então o agente precisa **generalizar** em vez de decorar uma fase.

**Repetição de ação:** cada decisão vale por 2 quadros do jogo (`REPETIR_ACAO = 2`). Com menos decisões por partida, fica mais fácil ligar uma ação à consequência dela. A DeepMind fazia o mesmo com 4 quadros nos jogos de Atari.

---

## 3. Ator e crítico: duas redes

O PPO treina **duas redes pequenas** ao mesmo tempo, as duas com 6 entradas e duas camadas de 32 neurônios:

- **Ator (1.379 pesos):** olha os sensores e dá a chance de cada ação. É quem joga.
- **Crítico (1.313 pesos):** olha os sensores e estima o **valor**: quanta recompensa ainda vem pela frente a partir dali.

No `assistir.py`, o valor do modelo treinado fica perto de **20**. Não é por acaso: cada passo vale 0.2 (dois quadros de 0.1) e o desconto `GAMMA = 0.99` faz o futuro distante valer cada vez menos, então a soma de tudo que vem pela frente dá 0.2 / (1 − 0.99) = 20. Quando um obstáculo difícil se aproxima e o crítico acha que pode dar errado, o valor cai.

---

## 4. Como o PPO aprende

Um ciclo, repetido umas 245 vezes num treino de 2 milhões de passos:

1. **Joga:** 8 partidas em paralelo, 1.024 passos cada. O ator **sorteia** as ações pelas chances que dá (é assim que ele explora).
2. **Compara com o esperado:** para cada passo, calcula a **vantagem**: deu mais ou menos recompensa do que o crítico esperava?
3. **Ajusta por gradiente:** aumenta a chance das ações com vantagem positiva e diminui a das com vantagem negativa. O crítico também é ajustado para errar menos na próxima.
4. **Freia:** o *clip* (o "P" de *Proximal*) impede que a chance de qualquer ação mude mais de ~20% numa atualização. Sem esse freio, um lote de partidas com sorte pode estragar uma política que já funcionava.

O passo 3 é exatamente a descida de gradiente da lição 1. A diferença é **o que** se tenta diminuir: lá era o erro contra a resposta certa; aqui é "menos recompensa do que poderia".

---

## 5. O que observar no relatório de exemplo

O treino de exemplo está em [`resultados/RELATORIO.md`](resultados/RELATORIO.md). Cada treino dá números um pouco diferentes, mas os padrões se repetem.

![pontos](resultados/01_pontos.png)

- **Platô, depois salto.** Por uns 600 mil passos o agente fica perto de 150 pontos: pular sem parar é uma estratégia "boa o bastante" e difícil de largar (um *ótimo local*). De repente ele descobre que esperar o cacto chegar perto rende muito mais, e a média dispara.
- **Treino × teste.** Durante o treino a média fica em ~4.300, abaixo do máximo, porque o ator ainda sorteia ações para explorar. Nos testes, o `relatorio.py` e o `assistir.py` escolhem sempre a ação mais provável, e o modelo final fez 5000 em todas as 10 partidas novas.

![ppo](resultados/02_ppo.png)

- **A entropia despenca junto com o salto dos pontos:** o agente deixa de escolher ao acaso e fica decidido.

![decisões](resultados/04_decisoes.png)

- **Em cima:** a rede aprendeu uma fronteira inclinada (pular quando o cacto chega perto, e mais cedo quanto mais rápido o jogo), parecida com a da lição 1. Só que aqui ninguém deu exemplos.
- **Embaixo:** com o pássaro na altura da cabeça, o agente fica abaixado enquanto ele está longe e **pula por cima** quando chega perto. Não é o que um humano faria, mas funciona. A recompensa diz se deu certo, nunca como fazer.

---

## 6. Onde fica guardado o aprendizado?

Num **arquivo de pesos**. Depois de treinado, o modelo é só isso: números. Carregar o arquivo = ter o dino que sabe jogar.

| Lição | Arquivo | O que tem dentro |
|---|---|---|
| 1 | `resultados/historico_pesos.csv` | os 25 pesos em cada época; a última linha é a rede treinada |
| 3 | `resultados/campeoes.pkl` | o genoma do campeão de cada geração (estrutura + pesos) |
| 4 | `resultados/modelos/ppo_02000k.zip` | `policy.pth` com os 2.692 pesos do ator e do crítico (~64 KB) |

Um LLM é **a mesma coisa em escala gigante**: um arquivo (geralmente `.safetensors`) com bilhões de pesos, que ocupa de alguns a centenas de gigabytes. A estrutura da rede é outra (um *transformer*), mas a ideia é igual: o treino ajusta os números, e depois o arquivo é carregado para usar.

E, como aqui, **usar não muda o arquivo**. Rodar o `assistir.py` mil vezes não ensina nada ao dino; só um novo `treinar.py` muda os pesos. Um LLM também não aprende enquanto conversa com você: o que ele "aprende" vem de novas rodadas de treino, que geram um novo arquivo de pesos, ou seja, uma nova versão do modelo.

---

## 7. Hiperparâmetros (topo de `treinar.py` e `ambiente.py`)

| Parâmetro | Valor | O que faz |
|---|---|---|
| `RECOMPENSA_VIVO` / `RECOMPENSA_BATER` | 0.1 / −1 | O que o agente tenta maximizar. Mudar isto muda o comportamento. |
| `REPETIR_ACAO` | 2 | Quadros por decisão. |
| `N_AMBIENTES` | 8 | Partidas jogadas em paralelo para coletar experiência. |
| `PASSOS_POR_COLETA` | 1024 | Passos por partida antes de cada atualização. |
| `TAXA_APRENDIZADO` | 0.0003 | Tamanho do passo do gradiente, como na lição 1. |
| `GAMMA` | 0.99 | Quanto o futuro vale. Perto de 1 = enxerga longe. |
| `ENTROPIA` | 0.03 | Bônus por continuar explorando. Ajudou a sair do platô de "pular sempre". |
| `CLIP` | 0.2 | Quanto a política pode mudar por atualização. |
| `CAMADAS` | [32, 32] | Tamanho das redes do ator e do crítico. |

---

## Exercícios

1. Em `ambiente.py`, coloque `CUSTO_PULO = 0.01`. O dino passa a pular menos? Olhe o `assistir.py`.
2. Coloque `REPETIR_ACAO = 1` e treine de novo. Quanto tempo o platô dura agora?
3. Coloque `ENTROPIA = 0`. O agente consegue sair do "pular sempre"?
4. Mude `GAMMA` para 0.9. O valor esperado no `assistir.py` vai ficar perto de quanto? (Dica: refaça a conta da seção 3.)
5. Tire o sensor de velocidade (em `dino/jogo.py`, `sensores()`, troque `vel` por `0`). A fronteira do `04_decisoes.png` continua inclinada?

---

## E os LLMs?

No ajuste de um LLM com reforço (RLHF e similares), as peças são as mesmas:

| Aqui | No LLM |
|---|---|
| observação: 6 sensores | o texto até agora |
| ação: nada, pular, abaixar | o próximo pedaço de palavra (*token*) |
| partida | uma resposta inteira |
| recompensa: sobreviver | nota de um modelo treinado com avaliações de pessoas, ou "o código passou nos testes" |
| ator | o próprio LLM |
| crítico | uma rede que estima a nota esperada |
| clip do PPO | um freio para o modelo não se afastar demais do original |

O PPO foi o algoritmo usado nas primeiras versões de RLHF. Hoje há variações, mas o ciclo continua o desta lição: **tentar, receber nota, ajustar os pesos por gradiente**.
