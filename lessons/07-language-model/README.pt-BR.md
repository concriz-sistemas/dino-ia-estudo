[English](README.md) · **Português** · [Español](README.es.md)

# Lição 7 — Um modelo de linguagem minúsculo

> **Objetivo:** construir a menor versão possível de um LLM e olhar por dentro dele.
> Ele lê texto e aprende a prever o próximo caractere, com o mesmo projeto dos grandes
> (um *transformer*), só que minúsculo. Ele treina com os READMEs deste estudo, em uns 30 segundos.

![o modelo escrevendo](results/generation.gif)

*O modelo escreve um caractere de cada vez. Embaixo, a chance que ele dá aos 8 próximos caracteres mais prováveis; o laranja é o caractere sorteado.*

O modelo foi treinado com texto em inglês, então escreve em inglês.

Rode um de cada vez, nesta ordem.

1. Treine o modelo (uns 30 segundos num Mac recente e mais tempo num computador sem uma placa de vídeo que o PyTorch consiga usar):

```bash
python lessons/07-language-model/train.py
```

2. Faça ele escrever:

```bash
python lessons/07-language-model/generate.py
```

3. Veja como ele escolhe cada caractere:

```bash
python lessons/07-language-model/generate.py --explain
```

4. Olhe por dentro dele: *embeddings*, similaridade de cosseno e atenção:

```bash
python lessons/07-language-model/inside.py
```

O modelo treinado já está no repositório, então os passos 2 a 4 funcionam sem treinar.

---

## 1. O trabalho todo: adivinhar o próximo caractere

Um modelo de linguagem faz uma coisa só: dado o texto até agora, ele dá uma **chance a cada pedaço de texto que pode vir em seguida**. Para escrever, ele sorteia um pedaço de acordo com essas chances, acrescenta ao texto e pergunta de novo.

Aqui cada pedaço (cada **token**) é um único caractere: 53 diferentes, entre letras, dígitos, pontuação, espaço e quebra de linha. Os LLMs de verdade usam pedaços de palavras no lugar ("dino", "saur", "jump", "ing"), com vocabulários de dezenas de milhares a uns duzentos mil tokens.

O texto de treino é o README em inglês do início do estudo e das lições 1 a 6, limpo (sem tabelas, sem código, sem links), em minúsculas: **46.605 caracteres**. Ele está congelado em [`data/study.txt`](data/study.txt). Os primeiros 90% são usados no treino; os últimos 10% são **texto não visto**, para conferir como o modelo se sai com um texto com o qual ele nunca treinou.

A **perda** é a mesma ideia da lição 1: o quanto o modelo se surpreende com o próximo caractere de verdade. Um modelo que chuta ao acaso tem perda de 3,97. Quanto menor, melhor.

---

## 2. Modelo 1: contando pares

O modelo mais simples possível, o **bigrama**, só conta que caractere vem depois de qual no texto de treino: depois de "q", quase sempre "u"; depois de um espaço, muitas vezes "t". Sem rede, sem treino.

A perda dele no texto não visto é **2,44**, bem melhor que o acaso. Mas ele só olha para o único caractere anterior, então escreve assim:

```
the dino ond, mpenche me fond ps, --oen. one min torith, 3v/7;qoomat5'sschinletchtsth
```

---

## 3. Modelo 2: um *transformer* minúsculo

[`tiny_lm.py`](tiny_lm.py) monta o mesmo tipo de rede de um LLM, só que minúscula:

1. ***Embeddings*.** Cada caractere vira um vetor de 64 números (o seu *embedding*), mais um vetor para a sua posição (1º, 2º, 3º... caractere). Esses números são aprendidos.
2. **Atenção.** Cada posição olha para os caracteres anteriores (até 64) e decide quais importam para o que vem a seguir. As posições nunca podem olhar para o futuro: isso seria espiar a resposta.
3. **Uma rede pequena** (como a da lição 1) processa o que a atenção juntou.
4. Os passos 2 e 3 formam um **bloco**; o modelo empilha 2 deles. No final, uma pontuação para cada um dos 53 caracteres, transformada em chances.

No total, **111.029 pesos**, treinados por descida de gradiente como todas as redes deste estudo.

![perda](results/01_loss.png)

| Passo | Perda no texto de treino | Perda no texto não visto | O que ele escreve depois de "the dino " |
|---|---|---|---|
| 0 | 4,15 | 4,16 | `o'9,sqj'/9+r%-62)'m=3%nw(-)zet(o60i` |
| 250 | 2,25 | 2,28 | `on the en mote the ands weiget. the min tory` |
| 500 | 1,70 | 1,87 | `only step. ritep (the screace work + in to it` |
| 1000 | 1,38 | 1,68 | `only much mutes for 300 muches. the minutions` |
| 2000 | 1,14 | 1,64 | `only matches teaches path. in the explore,` |
| 4000 | 0,95 | 1,72 | `only model steps to find -1 expected value` |

Primeiro ele aprende quais caracteres são comuns, depois palavras curtas ("the", "on"), depois palavras mais longas ("network", "matches"), e depois algo que soa como inglês mas não quer dizer nada. [`results/samples.md`](results/samples.md) tem as amostras completas.

### Aprendendo ou decorando?

Olhe as duas curvas. A perda no **texto de treino** continua caindo até o fim. A perda no **texto não visto** para de melhorar por volta do passo 1400 (melhor: **1,64**) e depois piora devagar. Dali em diante, o modelo está **decorando** este texto em particular em vez de aprender inglês: é *overfitting*, a mesma armadilha da lição 1 com 20 exemplos.

O `train.py` guarda o modelo do seu melhor momento no texto não visto. Um *dropout* de 10% (desligar sinais ao acaso durante o treino) também dificulta decorar.

A cura de verdade é **muito mais texto**. 47 mil caracteres cabem em poucas páginas; os LLMs treinam com trilhões de tokens, então não conseguem decorar tudo e precisam aprender os padrões.

---

## 4. Escrevendo: acaso e temperatura

`generate.py --explain` mostra cada escolha. Depois de "the dino sco", o modelo dá 97% para "r": ele aprendeu a palavra "score". Depois de um espaço, as chances se espalham, porque muitas palavras podem vir em seguida:

```
                       text so far   the 5 most likely next characters         drawn
                       'the dino '   a  15%  s  14%  t   8%  p   7%  i   7%   -> 's'
                      'the dino s'   t  42%  a  22%  c  17%  e   6%  h   4%   -> 'c'
                     'the dino sc'   o  63%  r  30%  a   3%  e   2%  h   1%   -> 'o'
                    'the dino sco'   r  97%  n   1%  v   1%  m   0%  s   0%   -> 'r'
```

A **temperatura** muda como as chances são usadas. Temperatura baixa deixa os caracteres prováveis ainda mais prováveis; temperatura alta dá mais chance aos improváveis. Experimente:

```bash
python lessons/07-language-model/generate.py --compare
```

| Temperatura | O que ele escreve depois de "the dino " |
|---|---|
| 0,3 | `steps the same a good step of the example step over action 50 and 2000 and the step and the dino and the and the same the same and the` |
| 0,8 | `only matches teacher's when example find of the 292020000 15 is in lotic still it learning:` |
| 1,5 | `only much. rule for 300,5-4zen. the minutoring, 24bunce, at5k scompletes fipt)? helthile togree varixelys` |

Com 0,3 ele fica preso repetindo as palavras mais seguras. Com 1,5 ele inventa palavras. Os assistentes de chat costumam escrever em algum ponto no meio.

---

## 5. *Embeddings* e similaridade de cosseno, de verdade

Na lição 1, a similaridade de cosseno era só uma analogia. Aqui ela é real: cada caractere tem um vetor de 64 números, e a similaridade de cosseno diz o quanto dois deles apontam para o mesmo lado.

![mapa dos embeddings](results/03_embeddings.png)

*Os 64 números de cada caractere espremidos em 2, para caberem num mapa.*

Os **dígitos** formam um grupo próprio, longe das letras. Ninguém disse ao modelo que "3" e "4" são números. Ele percebeu que eles aparecem nos mesmos tipos de lugar (do lado de outros dígitos, antes do "k" em "500k", depois de "lesson "), então deu a eles vetores que apontam para o mesmo lado.

![similaridade de cosseno](results/02_cosine.png)

| Grupo | Similaridade de cosseno média entre si | Com o resto |
|---|---|---|
| dígitos | +0,18 | −0,02 |
| vogais | +0,09 | −0,04 |
| consoantes | +0,03 | −0,02 |
| pontuação e espaço | +0,02 | −0,02 |

Os caracteres mais próximos de "3" são "4" (+0,45), "7" (+0,38) e "0" (+0,24). Os mais próximos de "." incluem ";" e "?", outros sinais de pontuação.

Esses sinais são fracos para a maioria dos caracteres: uma letra sozinha carrega pouco significado. Nos LLMs, cujos tokens são palavras e pedaços de palavras, o mesmo mecanismo coloca "dog" (cachorro) e "puppy" (filhote) perto um do outro, porque eles são usados nos mesmos lugares. É a mesma ideia, com mais significado para capturar.

---

## 6. Atenção: para onde cada posição olha

![atenção](results/04_attention.png)

Cada painel é uma das 4 cabeças de atenção da última camada, lendo "the dino jumps over the cactus.". Cada linha é uma posição, e as células escuras são os caracteres anteriores para os quais ela olha ao adivinhar o que vem a seguir. Nada aparece acima da diagonal: o futuro fica escondido.

Nesta frase, algumas cabeças olham principalmente para o começo da palavra atual (na cabeça 1, o "u", o "m" e o "p" de "jumps" olham para o "j"); outras olham um ou dois caracteres para trás. Ninguém programou essas regras; elas saíram do treino, porque ajudam a adivinhar o próximo caractere.

---

## 7. Este modelo vs um LLM de verdade

| | Esta lição | Um LLM grande |
|---|---|---|
| Token | 1 caractere | um pedaço de palavra |
| Vocabulário | 53 tokens | de dezenas de milhares a ~200.000 tokens |
| Contexto (até onde ele consegue olhar para trás) | 64 caracteres | de milhares a milhões de tokens |
| Pesos | 111.029 | de bilhões a centenas de bilhões |
| Texto de treino | 46.605 caracteres | trilhões de tokens |
| Treino | uns 30 segundos num notebook | semanas ou meses em milhares de GPUs |

O projeto é o mesmo: *embeddings*, blocos de atenção, uma pontuação para cada próximo token possível, descida de gradiente na perda. A diferença é a escala, e com a escala vem a capacidade de escrever coisas que fazem sentido.

E esta lição é só a primeira etapa de um LLM, o **pré-treinamento**. Depois dele, os modelos de verdade são ajustados imitando boas respostas ([lição 5](../05-imitation/README.pt-BR.md)) e por aprendizado por reforço ([lição 4](../04-reinforcement/README.pt-BR.md)).

---

## Exercícios

1. Treine com qualquer outro arquivo de texto, como um livro em texto puro: `train.py --text book.txt`. Ele aprende mais rápido com mais texto? Ainda decora?
2. Em `tiny_lm.py`, coloque `DROPOUT = 0` e treine de novo. Onde a perda no texto não visto para de melhorar agora?
3. Coloque `CONTEXT = 8`. O que acontece com a perda e com as palavras que ele escreve?
4. Coloque `LAYERS = 1`. Um bloco basta?
5. Rode `generate.py --prompt "the gradient "` com temperaturas 0,3 e 1,5 (use `--temperature`). Qual soa mais como o estudo?

**Lição anterior:** [aprender a enxergar](../06-vision/README.pt-BR.md) · **Início do estudo:** [todas as lições](../../README.pt-BR.md)
