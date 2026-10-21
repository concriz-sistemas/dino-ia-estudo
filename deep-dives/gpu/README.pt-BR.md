[English](README.md) · **Português** · [Español](README.es.md)

# Mergulho profundo: por que GPUs?

> Por que todo mundo treina redes neurais em placas de vídeo? Que cálculos estão acontecendo de verdade,
> e por que eles precisam de tanto poder de computação? Com uma medição real no Mac usado neste estudo.

Para repetir a medição no seu próprio computador (cerca de 1 minuto):

```bash
python deep-dives/gpu/benchmark.py
```

Depois, para redesenhar os gráficos que transformam os números em tempo (alguns segundos):

```bash
python deep-dives/gpu/scale.py
```

---

## 1. O que uma rede neural calcula de verdade

Quase tudo é uma única operação, repetida um número enorme de vezes: **multiplicar dois números e somar o resultado a um total acumulado** (um *multiply-add*, ou "multiplica e soma").

Pegue a primeira camada da [lição 1](../../lessons/01-neuron-and-gradient/README.pt-BR.md). Para cada um dos 600 exemplos, cada um dos 6 neurônios ocultos calcula `peso₁ × distância + peso₂ × velocidade + viés`. São 600 × 6 = 3.600 pequenas somas, cada uma com 2 "multiplica e soma". Escrito como uma tabela de números, isso é uma **multiplicação de matrizes**:

```
exemplos (600 × 2)  ×  pesos (2 × 6)  =  somas (600 × 6)
```

Toda rede deste estudo funciona assim. Redes maiores só têm tabelas maiores: uma camada de um LLM multiplica tabelas com milhares de linhas e colunas.

Duas coisas importam nesses "multiplica e soma":

- **São muitos.** Multiplicar duas tabelas de 4096 × 4096 exige 4096³ ≈ 69 bilhões de "multiplica e soma".
- **Eles não dependem uns dos outros.** A soma do exemplo 1, neurônio 1 não precisa da soma do exemplo 2, neurônio 5. Todos podem ser feitos **ao mesmo tempo**.

Treinar custa cerca de 3 vezes mais do que só usar a rede, porque o passo para trás (a retropropagação, lição 1) precisa de mais umas duas multiplicações de matrizes para cada uma do passo para a frente.

---

## 2. CPU vs GPU: poucos chefs experientes ou um exército de ajudantes

**Analogia.** Uma **CPU** é uma cozinha com poucos chefs experientes (nesta medição, o PyTorch usou 4 deles). Cada um é muito rápido e consegue seguir qualquer receita complicada, passo a passo, decidindo o que fazer em seguida. Uma **GPU** é uma cozinha com **milhares de ajudantes**. Cada um é mais simples e mais lento, e todos precisam fazer o mesmo tipo de tarefa, mas eles trabalham ao mesmo tempo.

Para preparar um prato elaborado, os chefs ganham. Para picar 69 bilhões de cebolas, os ajudantes ganham, desde que haja cebola suficiente para manter todos ocupados.

![poucos chefs vs muitos ajudantes](charts/fill.gif)

*Preenchendo a tabela de resultado de uma multiplicação de matrizes. Cada célula é uma linha vezes uma coluna, e nenhuma célula precisa esperar outra ficar pronta. Com 4 trabalhadores, 256 células levam 64 passos; com 64 trabalhadores, 4 passos. Um ajudante sozinho é mais lento do que um chef, mas eles são tantos que, numa tabela grande o bastante, ganham.*

Multiplicação de matrizes é picar cebola: bilhões de "multiplica e soma" idênticos e independentes. É por isso que as GPUs, projetadas para calcular milhões de pixels ao mesmo tempo nos jogos, se revelaram perfeitas para redes neurais.

---

## 3. Medido neste Mac

A máquina: um **Mac mini com chip Apple M4** (10 núcleos, sendo 4 de alto desempenho) e 24 GB de memória.

![velocidade](charts/02_speed.png)

| Tamanho da matriz | CPU | GPU | Mais rápida |
|---|---|---|---|
| 32 × 32 | 0,001 ms | 0,22 ms | CPU, cerca de 200× |
| 256 × 256 | 0,024 ms | 0,24 ms | CPU, cerca de 10× |
| 1024 × 1024 | 1,15 ms | 1,68 ms | CPU, um pouco |
| 2048 × 2048 | 9,5 ms | 5,3 ms | GPU, 1,8× |
| 4096 × 4096 | 75 ms | 41,5 ms | GPU, 1,8× |

No seu melhor, a GPU deste Mac fez **3,3 trilhões de operações por segundo**, e a CPU **1,8 trilhão**.

Duas surpresas:

- **Multiplicações pequenas são muito mais rápidas na CPU.** Todo trabalho enviado à GPU tem um custo fixo, cerca de 0,2 ms aqui, para prepará-lo, enviá-lo e esperar a resposta. Para uma tabela pequena, esse custo é muito maior do que o próprio trabalho. É como chamar a equipe inteira da cozinha para picar uma cebola.
- **Neste Mac, a GPU é "só" 1,8× mais rápida, mesmo em tabelas grandes.** Os chips da Apple têm uma unidade especial para contas com matrizes ao lado da CPU, então a CPU deles é excepcionalmente boa nisso. Num computador típico, a CPU é bem mais lenta em multiplicação de matrizes, e a diferença para uma placa de vídeo é muito maior.

![passos minúsculos](charts/03_tiny_steps.png)

Isso também explica as escolhas do estudo. As lições 3, 4 e 5 rodam na CPU: as redes delas são minúsculas (a da lição 4 tem 2.692 pesos), cada passo é uma multiplicação pequena, e o próprio jogo roda na CPU. No gráfico acima, 2.000 passos minúsculos seguidos levam mais ou menos o mesmo tempo nas duas (25 ms na CPU, 29 ms na GPU). As lições 6 e 7 usam a GPU, porque lotes de imagens e de texto são grandes o bastante para mantê-la ocupada.

---

## 4. Quanta computação cada lição precisou?

Contando os "multiplica e soma" de cada treino (um "multiplica e soma" = 2 operações):

| Treino | Operações aproximadas | Como é contado |
|---|---|---|
| Lição 1: a rede de 25 pesos | 0,26 bilhão | 36 por exemplo × 3 (treino) × 600 exemplos × 4.000 épocas |
| Lição 4: PPO | cerca de 0,3 trilhão | 2.692 pesos, 2 milhões de passos, 10 passadas pelos dados a cada atualização |
| Lição 7: o transformer minúsculo | 5,5 trilhões | 6 × 111.029 pesos × 8,2 milhões de tokens lidos |
| Lição 6: os olhos (CNN) | cerca de 76 trilhões | 20 milhões por imagem × 3 × 1,3 milhão de imagens vistas no treino |
| Um LLM grande (exemplo) | cerca de 6 trilhões de trilhões (6 × 10²⁴) | 6 × 70 bilhões de pesos × 15 trilhões de tokens |

A regra "6 × pesos × tokens" é um jeito comum de estimar o custo de treinar um modelo de linguagem: 2 operações por peso para cada token lido, vezes 3 para o treino. A linha do LLM grande usa tamanhos parecidos com os publicados para modelos abertos com 70 bilhões de pesos.

Números tão grandes são difíceis de imaginar, então aqui estão eles como tempo. Primeiro, **se uma pessoa fizesse à mão**, um "multiplica e soma" por segundo, sem nunca parar:

![à mão](charts/04_by_hand.png)

Até os 25 pesos da lição 1 levariam cerca de **4 anos** para uma pessoa. O transformer minúsculo da lição 7, cerca de **87.000 anos**.

E em máquinas de verdade, nas melhores velocidades medidas acima:

| Treino | CPU deste Mac | GPU deste Mac |
|---|---|---|
| Lição 1 | menos de um milissegundo | menos de um milissegundo |
| Lição 4 | 0,18 segundos | 0,1 segundos |
| Lição 7 | 3 segundos | 1,7 segundos |
| Lição 6 | 41 segundos | 23 segundos |

Esses são tempos *ideais*, em velocidade máxima o tempo todo. Os treinos reais levaram mais (a lição 7 levou cerca de 26 segundos; a lição 6, cerca de 22 minutos, porque também joga o jogo e seus lotes são pequenos). O que importa é a ordem de grandeza.

Agora o LLM grande, numa única escala compartilhada:

![quanto tempo](charts/05_how_long.png)

- Na GPU deste Mac: cerca de **60.000 anos**.
- Numa GPU atual de centro de dados: elas fazem da ordem de mil trilhões de operações por segundo (10¹⁵) nos formatos de número usados no treino, mas na prática rodam bem abaixo do seu pico; supondo 40% dele, cerca de **500 anos**.
- Em **10.000 delas ao mesmo tempo**: cerca de **18 dias**.

É por isso que os modelos grandes são treinados em **milhares de GPUs ao mesmo tempo**, durante semanas ou meses, e por isso custa tanto: não é um cálculo difícil, e sim um número astronômico de cálculos fáceis.

---

## 5. E quando o modelo responde?

Usar um LLM treinado também custa cerca de **2 operações por peso para cada token que ele escreve**. Para um modelo com 70 bilhões de pesos, isso dá cerca de 140 bilhões de operações por token, e é por isso que escrever uma resposta leva um tempo perceptível e que os serviços de chat precisam de grandes servidores de GPU. Escrever também é mais difícil de dividir entre os ajudantes do que treinar: cada token depende do anterior, então o modelo precisa escrever um de cada vez (o `generate.py` da lição 7 faz exatamente isso).

---

## Resumo

| | CPU | GPU |
|---|---|---|
| Analogia | poucos chefs experientes | milhares de ajudantes |
| Boa em | trabalho complicado, passo a passo; tarefas pequenas | a mesma tarefa simples em quantidades enormes de dados |
| Neste estudo | lições 1-5: redes pequenas, o jogo | lições 6-7: imagens e texto em lotes grandes |
| Custo fixo por tarefa | quase nenhum | cerca de 0,2 ms aqui: não compensa para tarefas minúsculas |

**Próximo:** [chips, memória e quantização](../hardware/README.pt-BR.md) · **Voltar para:** [a página inicial do estudo](../../README.pt-BR.md)
