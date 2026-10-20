# Lição 3 — Neuroevolução: 50 dinos evoluindo até jogar bem

> **Objetivo:** treinar redes neurais para jogar **sem gradiente**, só com seleção natural,
> e registrar cada geração para ver os pesos mudando.

![campeões de várias gerações](resultados/campeoes.gif)

*O campeão das gerações 0, 7, 14 e 29 jogando a mesma partida. O da geração 0 bate logo; os seguintes já sabem pular e abaixar.*

Rode um de cada vez, nesta ordem.

1. Assista à evolução. A tecla V acelera:

```bash
python licoes/03-neuroevolucao/treinar.py
```

2. Gere os gráficos e o `RELATORIO.md`:

```bash
python licoes/03-neuroevolucao/relatorio.py
```

3. Veja o campeão da geração 0:

```bash
python licoes/03-neuroevolucao/assistir.py --geracao 0
```

4. E compare com o campeão da última geração. As setas ← → trocam de geração:

```bash
python licoes/03-neuroevolucao/assistir.py
```

Para treinar sem janela (uns 10 segundos): `treinar.py --rapido`.

Para gravar de novo o GIF dos campeões, depois de treinar:

```bash
python licoes/03-neuroevolucao/assistir.py --gif
```

📊 **[Veja o relatório de um treino de exemplo](resultados/RELATORIO.md)**

---

## A ideia: evolução em vez de gradiente

Na Lição 1, cada peso era ajustado **calculando** para que lado a perda diminuía. Aqui não tem cálculo nenhum:

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

O `treinar.py` tem um `Registrador` que, ao fim de **cada geração**, salva em `resultados/historico.json`:

- melhor nota, média, mediana e desvio da população;
- as espécies e seus tamanhos;
- o **campeão completo**: cada neurônio com seu viés e cada conexão com seu peso;
- **todos os pesos** de toda a população.

O `historico.json` é texto puro. Abra num editor e procure `"campeao"` para ver uma rede inteira em números.

O `relatorio.py` transforma isso em gráficos e no [`RELATORIO.md`](resultados/RELATORIO.md):

| Gráfico | O que mostra |
|---|---|
| `01_fitness.png` | Melhor, média e mediana por geração |
| `02_especies.png` | Como a população se divide em espécies |
| `03_complexidade.png` | Quantos neurônios e conexões o campeão tem |
| `04_pesos_campeao.png` | O valor de cada conexão do campeão final ao longo das gerações |
| `05_distribuicao_pesos.png` | Todos os pesos de todos os dinos, geração a geração |
| `06_redes.png` | Desenho da rede do campeão em 4 momentos |

---

## O que observar no relatório de exemplo

O NEAT sorteia mutações, então **cada treino dá números diferentes**. Os valores entre parênteses são do treino de exemplo que está em [`resultados/RELATORIO.md`](resultados/RELATORIO.md). Os do seu treino vão ser outros, mas os padrões se repetem:

- **O melhor sobe aos saltos, a média sobe devagar.** Um dino com sorte chega a 5000 cedo (geração 14 no exemplo), mas a população inteira leva muito mais tempo para melhorar.
- **O melhor de uma geração pode ser pior que o da anterior** (300 → 881 → 232 nas gerações 0, 1 e 2 do exemplo). Cada geração joga uma partida **diferente**, com outra sequência de obstáculos, e às vezes a partida é mais difícil. Isso é proposital: se todas jogassem a mesma fase, os dinos decorariam a fase em vez de aprender a jogar (*overfitting*).
- **A mediana fica baixa** mesmo com a média subindo. A maioria dos filhos sofre mutações ruins e morre cedo. Evolução é desperdício: muitos tentam, poucos dão certo.
- **O tamanho da rede varia de treino para treino.** Às vezes as mutações cortam conexões e o campeão fica mais simples do que começou, com 18 conexões; às vezes ele ganha neurônios ocultos (14 conexões e 4 ocultos no exemplo). Veja `03_complexidade.png` e `06_redes.png`. O NEAT não busca a rede "mais bonita", só uma que funcione.
- **A regra principal sempre aparece: obstáculo longe = não pule.** Às vezes ela vem direto, como uma conexão distância → pular com peso negativo. No exemplo, ela passa por um neurônio oculto: distância → oculto 144 (**+4,45**) e oculto 144 → pular (**−5,10**). Positivo vezes negativo dá negativo, ou seja, a mesma regra por outro caminho. Ninguém ensinou isso aos dinos.

---

## Os parâmetros (`config-neat.txt`)

Os mais importantes para experimentar (a lista completa, com explicação, sai no relatório):

| Parâmetro | Valor | O que faz |
|---|---|---|
| `pop_size` | 50 | Dinos por geração. Mais = mais chances por geração, mas cada geração demora mais. |
| `weight_mutate_rate` | 0.8 | Chance de cada peso sofrer mutação. |
| `weight_mutate_power` | 0.5 | Tamanho da mutação. É o "equivalente" à taxa de aprendizado. |
| `node_add_prob` | 0.2 | Chance de nascer um neurônio novo. |
| `conn_add_prob` | 0.5 | Chance de nascer uma conexão nova. |
| `compatibility_threshold` | 3.0 | Menor = mais espécies. |
| `elitism` | 2 | Os 2 melhores passam intactos (garante que nunca piora). |
| `survival_threshold` | 0.2 | Só os 20% melhores de cada espécie se reproduzem. |

## Exercícios

1. Rode `assistir.py --geracao 0` e depois `assistir.py`. Use ← → para passar pelas gerações e ver em que momento o dino "entende" os pássaros.
2. Mude `pop_size` para 10 e depois 200. Rode `treinar.py --rapido` e `relatorio.py`. Em qual geração ele atinge 5000? E em quanto tempo real?
3. Coloque `weight_mutate_power = 3.0`. O que acontece com o gráfico de média?
4. Em `treinar.py`, troque `semente=self.n` por `semente=1` (todas as gerações jogam a mesma fase). Treine e use o `assistir.py`, que sempre sorteia uma fase nova. O campeão ainda joga bem?
5. Mude a nota: em `avaliar_genomas`, dê `genoma.fitness = dino.pontos - 10 * (número de pulos)`. O estilo de jogo muda? (Você vai precisar contar os pulos.)

**Próxima lição:** aprendizado por reforço com gradiente (PPO) — em breve.
