# Relatório do treino — aprendizado por reforço (PPO)

> Gerado automaticamente por `relatorio.py` a partir de `episodios.csv`, `atualizacoes.csv` e dos modelos em `modelos/`.
> Rode o treino de novo e este relatório muda.

## Resumo

| | |
|---|---|
| Passos de treino | 2.002.960 |
| Partidas jogadas no treino | 8.206 |
| Média de pontos: 200 primeiras → 200 últimas partidas | 115 → 4329 |
| Primeiro modelo salvo com média ≥ 4000 em partidas novas | 750k passos |
| Modelo final em 10 partidas novas | média 5000, pior 5000, melhor 5000 |

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
| 0k passos | 86 | 86 | 86 |
| 250k passos | 116 | 88 | 179 |
| 500k passos | 145 | 96 | 354 |
| 750k passos | 5000 | 5000 | 5000 |
| 1000k passos | 5000 | 5000 | 5000 |
| 1500k passos | 5000 | 5000 | 5000 |
| 2000k passos | 5000 | 5000 | 5000 |

## 4. O que a rede "pensa"

![decisões](04_decisoes.png)

Cada ponto é uma pergunta feita à rede: "com o obstáculo a esta distância e o jogo nesta velocidade, o que você faz?".

- **Em cima, o cacto:** no começo a rede responde igual para tudo. No fim aparece uma **fronteira inclinada**: pular quando o cacto chega perto, e mais cedo quanto mais rápido o jogo. É a mesma ideia da faixa da lição 1, mas lá a rede recebeu exemplos prontos; aqui, só a recompensa.
- **Embaixo, o pássaro na altura da cabeça:** a cor mostra a ação mais provável. Repare que a rede não precisa escolher o que nós escolheríamos: pular por cima do pássaro também funciona. A recompensa só diz se deu certo, não como.
