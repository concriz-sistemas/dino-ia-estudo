[English](README.md) · **Português** · [Español](README.es.md)

# Lição 4 — Aprendizado por reforço (PPO)

> **Objetivo:** juntar o **gradiente** da lição 1 com a **nota** da lição 3. Um único dino joga milhares de partidas e ajusta os pesos da rede, por gradiente, para ganhar mais recompensa.
> É a mesma família de técnicas usada para ajustar LLMs depois do pré-treino.

![modelos jogando](results/models.gif)

*O mesmo dino com 0, 500 mil, 1 milhão e 2 milhões de passos de treino, na mesma partida. O de 0k bate com 86 pontos e o de 500k com 129; os outros dois ainda estão vivos quando o clipe acaba (cada clipe para em 700 pontos). À direita, a chance que a rede dá para cada ação ("Chance of each action") e o valor que ela espera ganhar ("Expected value:").*

Rode um de cada vez, nesta ordem.

1. Treine (uns 80 segundos no Mac usado no exemplo, sem janela):

```bash
python lessons/04-reinforcement/train.py
```

2. Gere os gráficos e o `REPORT.md`:

```bash
python lessons/04-reinforcement/report.py
```

3. Assista ao modelo final jogando. As setas ← → trocam de modelo salvo, e R começa uma partida nova:

```bash
python lessons/04-reinforcement/watch.py
```

4. Compare com o modelo antes de treinar:

```bash
python lessons/04-reinforcement/watch.py --model 0
```

Os modelos de um treino de exemplo já estão no repositório, então o passo 3 funciona mesmo sem treinar.

Os gráficos, o relatório e a janela do jogo estão em inglês; os textos citados aqui aparecem como estão na tela, com a explicação em português.

---

## 1. Três jeitos de aprender

| | [Lição 1](../01-neuron-and-gradient/README.pt-BR.md) — gradiente | [Lição 3](../03-neuroevolution/README.pt-BR.md) — neuroevolução | Lição 4 — reforço |
|---|---|---|---|
| Quem joga | ninguém (exemplos prontos) | 50 dinos por geração | 1 dino, milhares de partidas |
| O que a IA recebe | a resposta certa de cada exemplo | uma nota no fim da partida | uma recompensa a **cada passo** |
| Como os pesos mudam | gradiente | mutação aleatória + seleção | gradiente |
| Precisa de alguém saber a resposta? | sim | não | não |

O reforço fica no meio: como na lição 3, ninguém diz qual é a jogada certa; como na lição 1, os pesos mudam por gradiente, sem sorteio de mutações.

---

## 2. O ambiente: observação, ação, recompensa

[`environment.py`](environment.py) embala o jogo no formato padrão **Gymnasium**, que qualquer algoritmo de reforço entende:

- **Observação:** os mesmos 6 sensores das lições anteriores.
- **Ação:** 0 = nada (`nothing`), 1 = pular (`jump`), 2 = abaixar (`duck`).
- **Recompensa:** `+0.1` por quadro vivo e `−1` ao bater. É o único "professor". O agente não sabe o que é um cacto; só sabe que alguns números na entrada antecedem a recompensa ruim.

Cada partida tem obstáculos sorteados de novo, então o agente precisa **generalizar** em vez de decorar uma fase.

**Repetição de ação:** cada decisão vale por 2 quadros do jogo (`ACTION_REPEAT = 2`). Com menos decisões por partida, fica mais fácil ligar uma ação à consequência dela. A DeepMind fazia o mesmo com 4 quadros nos jogos de Atari.

---

## 3. Ator e crítico: duas redes

O PPO treina **duas redes pequenas** ao mesmo tempo, as duas com 6 entradas e duas camadas de 32 neurônios:

- **Ator (1.379 pesos):** olha os sensores e dá a chance de cada ação. É quem joga.
- **Crítico (1.313 pesos):** olha os sensores e estima o **valor**: quanta recompensa ainda vem pela frente a partir dali.

No `watch.py`, o valor do modelo treinado ("Expected value:", valor esperado, no painel) fica por volta de **19.5**. Não é por acaso: cada passo vale 0.2 (dois quadros de 0.1) e o desconto `GAMMA = 0.99` faz o futuro distante valer cada vez menos, então a soma de tudo que vem pela frente dá 0.2 / (1 − 0.99) = 20. Esse é o teto, e o valor mostrado fica logo abaixo dele. Quando um obstáculo difícil se aproxima e o crítico acha que pode dar errado, o valor cai.

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

O treino de exemplo está em [`results/REPORT.md`](results/REPORT.md). Cada treino dá números um pouco diferentes, mas os padrões se repetem.

![pontos](results/01_scores.png)

- **Platô, depois salto.** Por uns 600 mil passos o agente fica entre 100 e 250 pontos: pular sem parar é uma estratégia "boa o bastante" e difícil de largar (um *ótimo local*). De repente ele descobre que esperar o cacto chegar perto rende muito mais, e a média dispara (a média móvel, "mean of the last 50" no gráfico, passa de 300 pela primeira vez em ~595 mil passos, de 1.000 em ~686 mil e de 4.000 em ~867 mil).
- **Treino × teste.** Durante o treino os pontos ficam entre ~4.200 e ~4.900 (média das últimas 200 partidas: 4.329), abaixo do máximo, porque o ator ainda sorteia ações para explorar. Nos testes, o `report.py` e o `watch.py` escolhem sempre a ação mais provável.

![avaliação](results/03_evaluation.png)

- **Cada modelo salvo em 10 partidas novas.** Os modelos de 0k, 250k e 500k passos fazem em média 86, 116 e 145 pontos. O primeiro modelo salvo com média de 4.000 ou mais é o de 750k, e do 750k em diante todos fizeram 5000 nas 10 partidas novas (5000 é o `SCORE_LIMIT`, onde a partida termina).

![ppo](results/02_ppo.png)

- **A entropia ("Entropy") despenca junto com o salto dos pontos:** o agente deixa de escolher ao acaso e fica decidido.

![decisões](results/04_decisions.png)

- **Em cima:** a rede aprendeu uma fronteira inclinada (pular quando o cacto chega perto, e mais cedo quanto mais rápido o jogo), parecida com a da lição 1. Só que aqui ninguém deu exemplos.
- **Embaixo:** com o pássaro na altura da cabeça, o agente fica abaixado (`duck`) enquanto ele está longe e **pula por cima** (`jump`) quando chega perto. Não é o que um humano faria, mas funciona. A recompensa diz se deu certo, nunca como fazer.

---

## 6. Onde fica guardado o aprendizado?

Num **arquivo de pesos**. Depois de treinado, o modelo é só isso: números. Carregar o arquivo = ter o dino que sabe jogar.

| Lição | Arquivo | O que tem dentro |
|---|---|---|
| [1](../01-neuron-and-gradient/README.pt-BR.md) | `results/weight_history.csv` | os 25 pesos em cada época; a última linha é a rede treinada |
| [3](../03-neuroevolution/README.pt-BR.md) | `results/champions.pkl` | o genoma do campeão de cada geração (estrutura + pesos) |
| 4 | `results/models/ppo_02000k.zip` | `policy.pth` com os 2.692 pesos do ator e do crítico (uns 62 KB) |

Um LLM é **a mesma coisa em escala gigante**: um arquivo (geralmente `.safetensors`) com bilhões de pesos, que ocupa de alguns a centenas de gigabytes. A estrutura da rede é outra (um *transformer*), mas a ideia é igual: o treino ajusta os números, e depois o arquivo é carregado para usar.

E, como aqui, **usar não muda o arquivo**. Rodar o `watch.py` mil vezes não ensina nada ao dino; só um novo `train.py` muda os pesos. Um LLM também não aprende enquanto conversa com você: o que ele "aprende" vem de novas rodadas de treino, que geram um novo arquivo de pesos, ou seja, uma nova versão do modelo.

---

## 7. Hiperparâmetros (topo de `train.py` e `environment.py`)

| Parâmetro | Valor | O que faz |
|---|---|---|
| `REWARD_ALIVE` / `REWARD_CRASH` | 0.1 / −1 | O que o agente tenta maximizar. Mudar isto muda o comportamento. |
| `ACTION_REPEAT` | 2 | Quadros por decisão. |
| `N_ENVS` | 8 | Partidas jogadas em paralelo para coletar experiência. |
| `STEPS_PER_ROLLOUT` | 1024 | Passos por partida antes de cada atualização. |
| `LEARNING_RATE` | 0.0003 | Tamanho do passo do gradiente, como na lição 1. |
| `GAMMA` | 0.99 | Quanto o futuro vale. Perto de 1 = enxerga longe. |
| `ENTROPY_COEF` | 0.03 | Bônus por continuar explorando. Ajudou a sair do platô de "pular sempre". |
| `CLIP` | 0.2 | Quanto a política pode mudar por atualização. |
| `LAYERS` | [32, 32] | Tamanho das redes do ator e do crítico. |

---

## Exercícios

1. Em `environment.py`, coloque `JUMP_COST = 0.01`. O dino passa a pular menos? Olhe o `watch.py`.
2. Coloque `ACTION_REPEAT = 1` e treine de novo. Quanto tempo o platô dura agora?
3. Coloque `ENTROPY_COEF = 0`. O agente consegue sair do "pular sempre"?
4. Mude `GAMMA` para 0.9. O valor esperado no `watch.py` vai ficar perto de quanto? (Dica: refaça a conta da seção 3.)
5. Tire o sensor de velocidade: em `dino/game.py`, em `sensors()`, troque `speed = self.speed / MAX_SPEED` por `speed = 0`. A fronteira do `04_decisions.png` continua inclinada?

---

## Qual a relação com os LLMs?

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

---

**Lição anterior:** [neuroevolução](../03-neuroevolution/README.pt-BR.md) · **Início do estudo:** [todas as lições](../../README.pt-BR.md) · **Próxima lição:** [imitação](../05-imitation/README.pt-BR.md)
