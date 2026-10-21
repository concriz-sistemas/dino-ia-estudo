[English](README.md) · **Português** · [Español](README.es.md)

# Lição 3 — Neuroevolução: 50 dinos evoluindo até jogar bem

> **Objetivo:** treinar redes neurais para jogar **sem gradiente**, só com seleção natural,
> e registrar cada geração para ver os pesos mudando.

![uma geração a cada 3, acelerado](results/evolution.gif)

*O treino de exemplo acelerado, mostrando uma geração a cada 3 (0, 3, 6 … 27 e a última, 29): 50 dinos por geração, os primeiros 480 pontos de cada partida. No canto de cima, à direita, o melhor resultado de cada geração até ali, inclusive das que o GIF pula; a barra fica verde quando um dino chega a 5000.*

![campeões de várias gerações](results/champions.gif)

*O campeão das gerações 0, 7, 14 e 29 jogando a mesma partida. O da geração 0 bate logo (129 pontos); os outros ainda estão vivos quando o trecho deles termina (o GIF para cada trecho em 600 pontos).*

Rode um de cada vez, nesta ordem.

1. Assista à evolução. As setas ↑ ↓ mudam a velocidade (de 1x a 64x), V pula para a velocidade máxima e volta, e ESC para e salva o que já foi registrado:

```bash
python lessons/03-neuroevolution/train.py
```

2. Gere os gráficos e o `REPORT.md`:

```bash
python lessons/03-neuroevolution/report.py
```

3. Veja o campeão da geração 0:

```bash
python lessons/03-neuroevolution/watch.py --generation 0
```

4. E compare com o campeão da última geração. As setas ← → trocam de geração, e R começa uma partida nova:

```bash
python lessons/03-neuroevolution/watch.py
```

Para treinar sem janela (uns 3 segundos): `train.py --fast`. Para mudar o número de gerações (30 por padrão): `train.py --generations 50`.

Para gravar de novo o GIF da evolução (ele treina sem janela e grava uma geração a cada 3):

```bash
python lessons/03-neuroevolution/train.py --gif
```

Para gravar de novo o GIF dos campeões, depois de treinar:

```bash
python lessons/03-neuroevolution/watch.py --gif
```

📊 **[Veja o relatório de um treino de exemplo](results/REPORT.md)** (em inglês)

---

## A ideia: evolução em vez de gradiente

Na [Lição 1](../01-neuron-and-gradient/README.pt-BR.md), cada peso era ajustado **calculando** para que lado a perda diminuía. Aqui não tem cálculo nenhum:

```
1. Crie 50 redes com pesos aleatórios
2. Coloque todas para jogar a mesma partida
3. Nota de cada uma = quantos quadros sobreviveu
4. As melhores viram "pais"; os filhos herdam os pesos com pequenas mutações aleatórias
5. Volte ao passo 2
```

Ninguém programou "pule quando o cacto estiver perto". Essa regra **surge** porque, por acaso, alguns dinos pulam na hora certa, sobrevivem e passam os pesos adiante.

| | Gradiente (Lição 1) | Neuroevolução (esta lição) |
|---|---|---|
| Como decide mudar um peso | Calcula a derivada da perda | Mutação aleatória + seleção |
| Precisa de "resposta certa"? | Sim, exemplos rotulados | Não, só uma nota no final |
| Muda a estrutura da rede? | Não, é fixa | Sim, cria e remove neurônios |
| Escala para bilhões de pesos? | Sim (é como LLMs são treinados) | Não, fica lento demais |
| Bom para | Muitos dados, redes grandes | Problemas pequenos, sem gradiente disponível |

---

## O NEAT

**NEAT** (*NeuroEvolution of Augmenting Topologies*, 2002) é o algoritmo usado aqui, pela biblioteca `neat-python`. Ele tem três truques:

1. **Começa mínimo:** só entradas ligadas às saídas, sem camada oculta. Neurônios e conexões são **adicionados por mutação** quando ajudam.
2. **Genes com histórico:** cada conexão tem um número de "inovação", para que dois pais possam cruzar redes de formatos diferentes sem embaralhar tudo.
3. **Espécies:** redes parecidas competem entre si. Uma inovação recém-criada (que no começo costuma piorar a rede) ganha tempo para amadurecer sem ser eliminada pelos campeões já estabelecidos.

---

## O que é registrado

O `train.py` tem um `Recorder` (registrador) que, ao fim de **cada geração**, salva em `results/history.json`:

- melhor nota, média, mediana e desvio da população;
- as espécies e seus tamanhos;
- o **campeão completo**: cada neurônio com seu viés e cada conexão com seu peso;
- **todos os pesos** de toda a população.

O `history.json` é texto puro. Abra num editor e procure `"champion"` (campeão) para ver uma rede inteira em números.

O `report.py` transforma isso em gráficos e no [`REPORT.md`](results/REPORT.md):

| Gráfico | O que mostra |
|---|---|
| `01_fitness.png` | Melhor, média e mediana por geração |
| `02_species.png` | Como a população se divide em espécies |
| `03_complexity.png` | Quantos neurônios e conexões o campeão tem |
| `04_champion_weights.png` | O valor de cada conexão do campeão final ao longo das gerações |
| `05_weight_distribution.png` | Todos os pesos de todos os dinos, geração a geração |
| `06_networks.png` | Desenho da rede do campeão em 4 momentos |

Os títulos e legendas dos gráficos estão em inglês: *best dino* = melhor dino, *population average* = média da população, *median* = mediana.

---

## O que observar no relatório de exemplo

O NEAT sorteia mutações, mas o `train.py` fixa o ponto de partida desse sorteio com `SEED = 0` (a semente): ele inicializa o `random` do Python antes de criar a população. Por isso **o seu treino dá exatamente os mesmos números** do treino de exemplo que está em [`results/REPORT.md`](results/REPORT.md) (30 gerações, 50 dinos por geração). Os valores entre parênteses são dele. Se você mudar a `SEED`, sai outra evolução, com outros números, mas os padrões se repetem:

- **O melhor sobe aos saltos, a média sobe devagar.** Um dino com sorte chega a 5000 cedo (geração 6 no exemplo), mas a população inteira leva muito mais tempo para melhorar (a média vai de 121 na geração 0 para 1084 na última).
- **O melhor de uma geração pode ser pior que o da anterior** (669 → 316 → 235 nas gerações 0, 1 e 2 do exemplo; e depois de chegar a 5000 na geração 6, a geração 7 caiu para 1477, e aí deu 5000 em todas as gerações de 8 a 29). Cada geração joga uma partida **diferente**, com outra sequência de obstáculos, e às vezes a partida é mais difícil. Isso é proposital: se todas jogassem a mesma fase, os dinos decorariam a fase em vez de aprender a jogar (*overfitting*).
- **A mediana fica baixa** mesmo com a média subindo. A maioria dos filhos sofre mutações ruins e morre cedo. Evolução é desperdício: muitos tentam, poucos dão certo.
- **O tamanho da rede varia de treino para treino.** Às vezes as mutações cortam conexões e o campeão fica mais simples do que começou, com 18 conexões (6 conexões e 1 neurônio oculto no exemplo); às vezes ele ganha neurônios ocultos. Veja `03_complexity.png` e `06_networks.png`. O NEAT não busca a rede "mais bonita", só uma que funcione.
- **A regra principal sempre aparece: obstáculo longe = não pule.** No exemplo ela vem direto, como uma conexão `distance` (distância) → `jump` (pular) com peso **−5,204**, o mais forte do campeão final. Com outras sementes ela pode passar por um neurônio oculto: `distance` → neurônio oculto com peso positivo, e neurônio oculto → `jump` com peso negativo. Positivo vezes negativo dá negativo, ou seja, a mesma regra por outro caminho. Ninguém ensinou isso aos dinos.

Os outros pesos fortes do campeão final do exemplo: `speed` (velocidade) → `duck` (abaixar) (−1,921), `speed` → `nothing` (não fazer nada) (−0,993), `hidden 178` (oculto 178) → `jump` (+0,632) e `elevation` (elevação) → `hidden 178` (+0,585). A tabela completa está na seção 6 do relatório.

---

## Os parâmetros (`neat-config.txt`)

Os mais importantes para experimentar (a lista completa, com explicação, sai no relatório):

| Parâmetro | Valor | O que faz |
|---|---|---|
| `pop_size` | 50 | Dinos por geração. Mais = mais chances por geração, mas cada geração demora mais. |
| `weight_mutate_rate` | 0.8 | Chance de cada peso sofrer mutação. |
| `weight_mutate_power` | 0.5 | Tamanho da mutação. É o "equivalente" à taxa de aprendizado. |
| `node_add_prob` | 0.2 | Chance de nascer um neurônio novo. |
| `conn_add_prob` | 0.5 | Chance de nascer uma conexão nova. |
| `compatibility_threshold` | 3.0 | Menor = mais espécies. |
| `elitism` | 2 | Os 2 melhores passam intactos para a próxima geração (uma rede boa nunca se perde por mutação). |
| `survival_threshold` | 0.2 | Só os 20% melhores de cada espécie se reproduzem. |

## Exercícios

1. Rode `watch.py --generation 0` e depois `watch.py`. Use ← → para passar pelas gerações e ver em que momento o dino "entende" os pássaros.
2. Mude `pop_size` para 10 e depois 200. Rode `train.py --fast` e `report.py`. Em qual geração ele atinge 5000? E em quanto tempo real?
3. Coloque `weight_mutate_power = 3.0`. O que acontece com o gráfico de média?
4. Em `train.py`, troque `seed=self.n` por `seed=1` (todas as gerações jogam a mesma fase). Treine e use o `watch.py`, que sempre sorteia uma fase nova. O campeão ainda joga bem?
5. Mude a nota: em `eval_genomes`, dê `genome.fitness = dino.score - 10 * (número de pulos)`. O estilo de jogo muda? (Você vai precisar contar os pulos.)
6. Em `train.py`, troque `SEED = 0` por outro número e rode `train.py --fast` e `report.py`. Compare o seu novo `REPORT.md` com o do exemplo: em qual geração ele atinge 5000? Qual o tamanho do campeão final? A regra `distance` → `jump` aparece direto ou por um neurônio oculto? (Isso sobrescreve a pasta `results/`; volte para `SEED = 0` e treine de novo para recuperar o exemplo.)

**Próxima lição:** [aprendizado por reforço com gradiente (PPO)](../04-reinforcement/README.pt-BR.md)
