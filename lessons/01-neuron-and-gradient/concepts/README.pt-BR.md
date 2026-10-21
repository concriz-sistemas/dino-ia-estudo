[English](README.md) · **Português** · [Español](README.es.md)

# Mergulho profundo: peso, viés, ativação e perda

> Um complemento mais calmo e mais visual da [lição 1](../README.pt-BR.md). Cada ideia vem com uma analogia do dia a dia,
> um gráfico, um pequeno exemplo resolvido à mão e o lugar onde ela aparece no código.

![girando os botões](charts/knobs.gif)

*Um neurônio é uma maquininha cheia de botões. Aqui os botões são girados um de cada vez: primeiro o peso, depois o viés. Repare no que cada um faz com a curva.*

Todos os gráficos desta página usam o neurônio mais simples possível: **uma entrada**, a distância até o cacto (`x`, de 0 = encostado nele a 1 = bem longe), e uma saída, o quanto ele "quer pular". Para desenhar os gráficos você mesmo:

```bash
python lessons/01-neuron-and-gradient/concepts/concepts.py
```

---

## 1. Peso: um botão de volume

**Analogia.** Imagine uma mesa de som, com um botão de volume para cada instrumento. Um **peso** é um desses botões, um por entrada: ele diz o quanto aquela entrada aparece na mixagem final. Aumente o volume e aquela entrada passa a importar mais. Gire até o zero e a entrada fica muda. Um peso *negativo* é um botão que funciona ao contrário: quanto mais alta a entrada, mais ela empurra para o outro lado.

**Fórmula.** Um neurônio multiplica cada entrada pelo seu peso e soma tudo:

```
soma = peso × entrada
```

![peso](charts/c1_weight.png)

Com uma entrada só, o peso é simplesmente a **inclinação** da reta. Positivo: quanto mais longe o cacto, maior a soma. Negativo: quanto mais longe o cacto, *menor* a soma. Zero: o neurônio ignora a distância por completo.

**Exemplo resolvido, duas entradas.** Um neurônio com peso −3 para a distância e +1 para a velocidade:

| Distância | Velocidade | Soma |
|---|---|---|
| 0,2 (perto) | 0,5 | −3 × 0,2 + 1 × 0,5 = **−0,1** |
| 0,8 (longe) | 0,5 | −3 × 0,8 + 1 × 0,5 = **−1,9** |

O peso −3 diz "a distância importa muito, e quanto mais longe, menos vontade eu tenho de pular". O peso +1 diz "a velocidade importa um pouco, e quanto mais rápido, mais vontade eu tenho de pular".

**No código:** `W1` (das 2 entradas para os 6 neurônios ocultos) e `W2` (dos neurônios ocultos para a saída). São 18 dos 25 números da rede.

---

## 2. Viés: um termostato

**Analogia.** Um termostato não muda *como* o aquecedor funciona; ele muda *quando* o aquecedor liga. O **viés** é a mesma coisa: um número somado à soma, que muda o ponto em que o neurônio vira de "não" para "sim".

**Fórmula.**

```
soma = peso × entrada + viés
```

![viés](charts/c2_bias.png)

Aqui o peso fica fixo em −10 e só o viés muda. O neurônio diz "pule" (chance acima de 50%) onde a soma é positiva, então ele vira em `x = viés / 10`: com viés 3, ele começa a querer pular quando o cacto está mais perto que 0,3; com viés 7, já em 0,7.

Sem viés, todo neurônio seria obrigado a virar exatamente em `x = 0`. O viés dá a cada neurônio o seu próprio ponto de virada.

**No código:** `b1` (um por neurônio oculto) e `b2` (um para a saída): os outros 7 dos 25 números. Eles começam em zero, e é por isso que, na época 0 da lição 1, todas as retas de [`06_hidden_lines.png`](../results/06_hidden_lines.png) se cruzam no mesmo ponto.

---

## 3. Ativação: um dimmer

**Analogia.** Um interruptor comum está desligado ou ligado. Um **dimmer** vai suavemente do apagado até o brilho máximo. Uma função de ativação é um dimmer aplicado à soma do neurônio: ela entorta a reta e a transforma numa curva.

![ativações](charts/c3_activation.png)

- **sigmoid** espreme qualquer soma em 0..1, o que funciona como uma **chance**. A lição 1 usa a sigmoide na saída: "chance de pular".
- **tanh** espreme em −1..1. A lição 1 usa a tanh nos neurônios ocultos.
- **ReLU** mantém as somas positivas e corta as negativas para zero. Ela e suas primas mais suaves (como a GELU, usada pelo modelo de linguagem da [lição 7](../../07-language-model/README.pt-BR.md)) são as mais comuns nas redes grandes de hoje.

**Por que isso importa.** Sem ativação, empilhar neurônios não adianta nada: uma reta de uma reta continua sendo uma reta. Dois neurônios lineares em sequência, o primeiro `2x + 1` e o segundo `−3 × (primeiro) + 0.5`, dão:

```
−3 × (2x + 1) + 0.5 = −6x − 2.5        (só mais uma linha reta)
```

Não importa quantos você empilhe, o resultado é uma reta só. É a curva que permite à rede desenhar formas.

---

## 4. Dois dimmers formam uma faixa

**Analogia.** Pense numa sala de espera com uma placa: "entrada a partir das 14h, mas não depois das 17h". São duas regras, "pelo menos 14h" e "ainda não deu 17h", e você só entra quando as duas concordam. A faixa de "pular" da lição 1 é exatamente isso: "o cacto está pelo menos tão perto assim, mas não perto demais".

![faixa](charts/c4_band.png)

- O **neurônio A** liga quando a distância passa de 0,2. O **neurônio B** liga quando ela passa de 0,5.
- O neurônio de saída soma os dois com pesos opostos, `4 × A − 4 × B − 3`: só entre 0,2 e 0,5 o A está ligado enquanto o B ainda está desligado.
- A sigmoide transforma esse calombo numa **chance de pular**.

| Distância | A | B | Soma 4A − 4B − 3 | Chance de pular |
|---|---|---|---|---|
| 0,1 | −0,99 | −1,00 | −2,95 | 5% |
| 0,35 | +1,00 | −1,00 | +4,99 | **99%** |
| 0,8 | +1,00 | +1,00 | −3,00 | 5% |

É isso que a rede treinada da lição 1 faz, com 2 entradas em vez de 1: dois dos seus neurônios ocultos acabam nas duas bordas da faixa ([`06_hidden_lines.png`](../results/06_hidden_lines.png)). Ninguém escolheu esses números à mão. O treino encontrou.

---

## 5. Perda: uma aposta

**Analogia.** Pense em cada previsão como uma **aposta**. Quando a rede diz "90% pular", ela está apostando 90% em pular. A **perda** é quanto a aposta custa depois que a resposta certa é revelada: uma aposta confiante e certa quase não custa nada, uma aposta tímida custa um pouco, e uma aposta confiante e *errada* custa caro.

**Fórmula** (entropia cruzada, a perda da lição 1), com `p` = a chance de pular que a rede deu:

```
se a resposta certa é PULAR:        loss = −log(p)
se a resposta certa é NÃO pular:    loss = −log(1 − p)
```

![perda](charts/c5_loss.png)

**Exemplo resolvido, três apostas:**

| Resposta certa | A rede disse | Perda |
|---|---|---|
| pular | 90% pular | 0,11 (confiante e certa: quase de graça) |
| não pular | 20% pular | 0,22 (bem segura e certa) |
| pular | 5% pular | **3,00** (confiante e errada: caro) |
| | **média** | **1,11** |

A perda da rede é a **média** sobre todos os exemplos. Treinar significa deixar essa média a menor possível.

**Por que não contar os erros e pronto?** "Certo ou errado" não distingue uma aposta de 51% de uma de 99%, então não consegue dizer *como* melhorar. A perda muda suavemente com cada peso, e é exatamente disso que o gradiente precisa.

**Por que não o erro quadrático?** O gráfico da direita mostra que o erro quadrático quase não castiga uma resposta errada e confiante (ele nunca passa de 1). A entropia cruzada dispara, então esses são os erros que a rede corrige primeiro.

---

## 6. Gradiente: descendo um morro na neblina

**Analogia.** Você está num morro, numa neblina fechada, e quer chegar lá embaixo. Não dá para ver o vale, mas seus pés sentem para que lado o chão está inclinado. Então você dá um passo morro abaixo, sente de novo, dá outro passo. O **gradiente** é essa inclinação, sentida no ponto onde você está. A **taxa de aprendizado** é o tamanho do seu passo.

![gradiente](charts/c6_gradient.png)

Aqui a perda aparece como função de um único peso (o peso da distância), com o viés fixo. As linhas laranja são a inclinação em cada passo:

| Passo | Peso | Perda | Inclinação | Novo peso = peso − 12 × inclinação |
|---|---|---|---|---|
| 0 | +2,00 | 2,690 | +0,4025 | −2,83 |
| 1 | −2,83 | 0,930 | +0,2721 | −6,10 |
| 2 | −6,10 | 0,427 | +0,0527 | −6,73 |
| 9 | −7,54 | 0,390 | +0,0021 | −7,57 |

- Uma **inclinação positiva** significa "aumentar o peso aumenta a perda", então o peso desce. Uma inclinação negativa o empurraria para cima.
- Longe do fundo, a inclinação é forte e os passos são grandes. Perto do fundo, o terreno fica plano e os passos ficam minúsculos: as duas fases do replay da lição 1.

A lição 1 faz isso com os 25 pesos ao mesmo tempo, e a retropropagação calcula as 25 inclinações de uma vez só. A seção 8 da lição 1 mostra a mesma caminhada num mapa com 2 pesos, e o que acontece quando os passos são pequenos demais ou grandes demais.

---

## Resumo

| Ideia | Analogia | O que faz | No código da lição 1 |
|---|---|---|---|
| Peso | um botão de volume | o quanto uma entrada importa, e em que direção | `W1`, `W2` |
| Viés | um termostato | onde o neurônio liga | `b1`, `b2` |
| Ativação | um dimmer | entorta a soma numa curva, para a rede conseguir desenhar formas | `np.tanh`, `sigmoid` |
| Perda | uma aposta | quanto as respostas da rede custam, em média | `loss()` |
| Gradiente | a inclinação sob seus pés na neblina | para que lado, e quanto, mexer cada peso | `backward()` |

**Voltar para:** [lição 1](../README.pt-BR.md)
