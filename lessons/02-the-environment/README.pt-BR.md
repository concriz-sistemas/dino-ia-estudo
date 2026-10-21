[English](README.md) · **Português** · [Español](README.es.md)

# Lição 2 — O ambiente: o jogo que a IA vai jogar

> **Objetivo:** entender o "mundo" da IA — o que ela vê, o que ela pode fazer
> e como sabemos se ela foi bem.

```bash
python lessons/02-the-environment/play.py
```

Jogue algumas partidas. No canto da tela, sob "What the AI sees:" (o que a IA vê), aparecem os **6 números que a IA vai receber**. Repare como eles mudam quando um cacto se aproxima.

---

## O que é o pygame?

É uma biblioteca de **Python** para fazer jogos 2D. Ela abre uma janela, desenha retângulos e imagens, lê o teclado e controla o tempo (60 quadros por segundo). Instala com `pip install pygame`.

Nosso jogo inteiro está em [`dino/game.py`](../../dino/game.py) — pouco mais de 200 linhas, todas comentadas.

---

## O ciclo do jogo

Todo jogo (e todo ambiente de IA) é um laço que se repete:

```
┌──────────────────────────────────────────────┐
│ 1. OBSERVAR   o estado atual  → sensors()    │
│ 2. DECIDIR    uma ação        → humano ou IA │
│ 3. AGIR       avançar um quadro → step()     │
│ 4. AVALIAR    bateu? +1 ponto?               │
└──────────────── repete ──────────────────────┘
```

Esse mesmo ciclo — **observação → ação → recompensa** — é a base do aprendizado por reforço, a técnica usada também no pós-treinamento de LLMs.

---

## Os três ingredientes que a IA precisa

### 1. Observação: o que ela "vê"

A IA não vê a tela. Ela recebe 6 números, calculados em `Game.sensors()`. Na tela, os nomes aparecem em inglês:

| # | Sensor | Significado | Faixa |
|---|---|---|---|
| 1 | distance | Quão longe está o próximo obstáculo (distância) | 0 (colado) a 1 (borda da tela) |
| 2 | width | Largura do obstáculo | ~0,3 a ~0,9 |
| 3 | height | Altura do obstáculo | ~0,4 a ~0,9 |
| 4 | elevation | Distância do obstáculo até o chão (elevação) | 0 = cacto; > 0 = pássaro |
| 5 | dino height | Altura do dino (está no ar?) | 0 = no chão |
| 6 | speed | Velocidade atual do jogo | 0,4 no início, 1 no máximo |

Todos os números estão **normalizados** (perto de 0 a 1). Redes neurais aprendem mal quando um número vale 800 e outro vale 0,02 — foi o mesmo motivo de usarmos `normalize()` na Lição 1.

> 💡 **Escolher boas entradas é metade do trabalho.** Se esquecermos a velocidade, a IA não teria como saber que precisa pular mais cedo quando o jogo acelera.

### 2. Ação: o que ela pode fazer

Três opções: `NOTHING` (nada), `JUMP` (pular) ou `DUCK` (abaixar). A rede terá **3 saídas**, e a maior vence.

### 3. Recompensa (ou *fitness*): como sabemos se foi bem

**+1 ponto por quadro sobrevivido.** Simples, e não diz *como* jogar — só o que é bom resultado. A IA vai ter que descobrir o resto.

> ⚠️ A recompensa define o que a IA aprende. Se déssemos ponto por pulo, ela aprenderia a pular sem parar. Esse problema — a IA otimizar a nota em vez do objetivo real — tem nome: *reward hacking*. Também acontece com LLMs.

---

## Por que vários dinos ao mesmo tempo?

O `Game` aceita `n_dinos`. Todos enfrentam **os mesmos obstáculos** na mesma partida, o que torna justa a comparação entre eles. Isso será essencial na próxima lição, em que 50 dinos competem para passar seus "genes" adiante.

## Exercícios

1. Quantos pontos você consegue? (Spoiler: a IA vai chegar a 5000.)
2. Quando o primeiro pássaro aparece, qual o valor do sensor "elevation"? E para cada um dos 3 tipos de pássaro (`bird_low`, `bird_mid`, `bird_high`)?
3. No `dino/game.py`, mude `JUMP_FORCE` para `-12`. O jogo continua possível?

**Próxima lição:** [neuroevolução — 50 dinos evoluindo](../03-neuroevolution/README.pt-BR.md)

---

## Trocando as figuras

O dino, os cactos e o pássaro são desenhados em [`dino/sprites.py`](../../dino/sprites.py) como **texto**: `#` é um pixel pintado, `w` é branco e `.` é transparente. Mude os desenhos e crie o personagem que quiser.

Isso é **só visual**. Para o jogo, cada coisa continua sendo um retângulo (`rect` em `game.py`), e a colisão e os sensores usam esses retângulos. A IA nunca vê as figuras, só os 6 números. Por isso trocar o desenho não muda nada no aprendizado.
