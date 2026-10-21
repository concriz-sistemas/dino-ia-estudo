[English](README.md) · [Português](README.pt-BR.md) · **Español**

# Lección 1 — Neurona, pérdida y gradiente (desde cero, solo con numpy)

> **Objetivo:** entender qué pasa *por dentro* cuando una red neuronal "aprende".
> Sin bibliotecas de IA: cada cálculo está escrito a mano en
> [`train_from_scratch.py`](train_from_scratch.py).

```bash
python lessons/01-neuron-and-gradient/train_from_scratch.py --live
```

El script entrena la red (unos 5 segundos), guarda el valor de cada peso en cada época y después **reproduce el entrenamiento en una ventana, como un video**: lo que la red piensa (con la línea recta de cada neurona oculta en morado), la pérdida bajando, cada uno de los 25 pesos ajustándose y el gradiente de cada peso en ese instante.

| Tecla | Qué hace |
|---|---|
| ESPACIO | pausa / continúa |
| ← → | retrocede / avanza una época (para ver un paso de gradiente a la vez) |
| ↑ ↓ | más rápido / más lento (de 1 a 100 épocas por cuadro) |
| R | vuelve a empezar desde cero |

La parte de abajo de la ventana recuerda las teclas, en inglés: "SPACE pause · ← → one epoch · ↑ ↓ speed · R restart".

![repetición del entrenamiento](results/training.gif)

Para grabar esta animación de nuevo:

```bash
python lessons/01-neuron-and-gradient/train_from_scratch.py --gif
```

Sin `--live`, el script entrena en unos 5 segundos, explica los primeros cálculos en la terminal y guarda las gráficas y el CSV en [`results/`](results/), que son los que se usan en el resto de esta página.

---

## 1. El problema

Antes de poner una IA en el juego, vamos a resolver una parte de él por separado:

> Dada la **distancia hasta el cactus** y la **velocidad del juego**, ¿es momento de saltar?

La respuesta correcta es una **franja**: si el cactus está demasiado cerca, ya es tarde; si está demasiado lejos, aterrizas antes de llegar a él. Y cuanto más rápido va el juego, antes hay que saltar.

![datos](results/01_data.png)

La red **nunca ve esta regla**. Solo recibe 600 ejemplos (los puntos de arriba) y tiene que descubrir el patrón por su cuenta. La terminal lo confirma: "600 examples, 137 of them are 'jump'." (600 ejemplos, 137 de ellos son "saltar").

### Qué son los puntos en las gráficas

Son la **clave de respuestas**. Antes de entrenar, el script sortea 600 situaciones y usa la regla verdadera para responder cada una. De ahí en adelante, en las gráficas de la red, los puntos negros son los 137 ejemplos marcados como "saltar" (los 463 de "no saltar" se dejan fuera para no saturar la gráfica). En la leyenda aparecen como "answer key: 'jump' examples" (clave de respuestas: ejemplos de "saltar").

Los puntos **nunca cambian**. Lo que cambia durante el entrenamiento es el **fondo de color**: la respuesta de la red para cada lugar del plano. Entrenar es hacer que el fondo se vuelva rojo donde están los puntos y azul en el resto.

Esto se llama **aprendizaje supervisado**: las respuestas de los ejemplos se conocen desde el principio, y la red aprende comparando lo que respondió con la clave de respuestas. En las lecciones 3 y 4 no hay clave de respuestas; el dino solo recibe una nota o una recompensa.

---

## 2. La red

```
 entradas          capa oculta           salida
 distancia  ──┐   ┌─ neurona 1 ─┐
              ├───┼─ neurona 2 ─┼───  P(saltar)
 velocidad  ──┘   └─ ...      6 ┘
```

Cada neurona hace siempre lo mismo:

```
salida = activación( peso₁·entrada₁ + peso₂·entrada₂ + ... + sesgo )
```

| Término | Qué es | En el código |
|---|---|---|
| **Peso** | Cuánto importa una entrada (puede ser negativo) | `W1`, `W2` |
| **Sesgo (bias)** | Un "empujón" fijo, desplaza el umbral de la neurona | `b1`, `b2` |
| **Activación** | Una curva que dobla la suma. Sin ella, toda la red sería solo una línea recta | `tanh`, `sigmoid` |
| **Forward** | Pasar las entradas por la red y obtener la predicción | `forward()` |

> 🔍 **¿Son nuevas para ti estas palabras?** La [explicación a fondo](concepts/README.es.md) presenta peso, sesgo, activación, pérdida y gradiente uno por uno, con analogías de la vida cotidiana, gráficas, una animación y pequeños ejemplos resueltos a mano.

La suma dentro de la activación, `peso₁·entrada₁ + peso₂·entrada₂ + sesgo`, es exactamente la fórmula de una **regresión lineal**. La sección 6 muestra qué significa eso en la práctica.

Esta red tiene 2×6 + 6 + 6×1 + 1 = **25 números ajustables**. Entrenar = encontrar buenos valores para esos 25 números. (Un LLM grande tiene **cientos de miles de millones** de ellos — pero la idea es la misma.)

---

## 3. Pérdida: midiendo cuánto se equivoca la red

Para mejorar, la red necesita una nota. La **pérdida** (*loss*) es un número que se hace grande cuando la red se equivoca y pequeño cuando acierta. Usamos la *entropía cruzada binaria*, que castiga mucho a quien se equivoca **con convicción** (decir 99% "salta" cuando era "no saltes").

---

## 4. Gradiente: ¿hacia qué lado mover cada peso?

Imagina la pérdida como un terreno montañoso, donde cada posición es una combinación de los 25 pesos. Queremos llegar al valle.

El **gradiente** responde, para cada peso: *"si aumento este peso un poquito, ¿la pérdida sube o baja? ¿Y cuánto?"*

- Gradiente **positivo** → aumentar el peso empeora → entonces lo **disminuimos**.
- Gradiente **negativo** → aumentar el peso mejora → entonces lo **aumentamos**.

Toda la regla del aprendizaje es esta línea (peso = peso − tasa de aprendizaje × gradiente):

```python
weight = weight - learning_rate * gradient
```

El script muestra esto ocurriendo de verdad en las primeras épocas. La salida está en inglés: `loss` es la pérdida, `accuracy` la exactitud, `current value` el valor actual, `new value` el nuevo valor, y el comentario entre paréntesis dice que el signo + significa que aumentar el peso aumentaría la pérdida:

```
Epoch 0: loss=0.9365  accuracy=46%
  example with one weight, W2[0,0]:
    current value = +1.93379
    gradient      = -0.12817   (+ sign => increasing the weight would increase the loss)
    new value     = +1.93379 - 0.5 × -0.12817 = +1.99788
```

### ¿Cómo se calcula el gradiente? Retropropagación

La función `backward()` aplica la **regla de la cadena** del cálculo, de atrás hacia adelante: primero el error en la salida, después cuánto contribuyó a él cada neurona oculta, después cuánto contribuyó cada peso de entrada a la neurona. Por eso el nombre *backpropagation*: el error **regresa** por la red.

### ¿Cómo saber si el cálculo está bien? Verificación numérica

El script compara la fórmula con una estimación por fuerza bruta (mueve el peso +0.00001 y −0.00001 y mide la diferencia en la pérdida). Si coinciden, la retropropagación es correcta. En la terminal esto aparece bajo "[Checking the gradient]  formula  vs  brute force" (verificando el gradiente: fórmula vs fuerza bruta):

```
  W1[0, 0]:  -0.084432  vs  -0.084432
  W2[3, 0]:  +0.259212  vs  +0.259212
```

---

## 5. Viendo el entrenamiento

### La curva de aprendizaje

![pérdida](results/02_loss.png)

- La **pérdida** cae en picada en los primeros cientos de épocas y después baja despacio.
- La **exactitud** sube de ~46% (adivinar) a 100%.
- El **gradiente** va disminuyendo: cerca del valle, el terreno se vuelve plano y los pasos se hacen pequeños.

### Los pesos cambiando

![pesos](results/03_weights.png)

Cada línea es uno de los 25 números. El CSV [`results/weight_history.csv`](results/weight_history.csv) tiene el valor exacto de todos ellos en **cada época** — ábrelo en Excel y explora.

### Lo que la red "piensa" a lo largo del entrenamiento

![fronteras](results/04_boundaries.png)

El título de la gráfica lo resume: "Background = what the network thinks (red = 'jump!', blue = 'wait'). Dots = answer key, fixed from the start." (fondo = lo que la red piensa, rojo = "¡salta!", azul = "espera"; puntos = clave de respuestas, fija desde el principio).

- **`epoch 0` (época 0):** pesos aleatorios, la red adivina.
- **`epoch 50`:** empezó a notar que "lejos = no saltar".
- **`epoch 300`:** ya encontró la franja, todavía borrosa.
- **`epoch 4000`:** franja nítida, prácticamente igual a la regla verdadera.

### Dos fases: descenso rápido, después ajuste fino

En la repetición (`--live`) se ve que la franja surge en los primeros cientos de épocas y después casi no cambia. Son dos fases:

1. **Descenso rápido (épocas 0 a ~300).** Con pesos aleatorios, la red se equivoca mucho, así que el gradiente es grande. Como el paso es `tasa × gradiente`, los pasos también son grandes, y la forma aparece. La pérdida cae en picada.
2. **Ajuste fino (de ~300 en adelante).** La red ya está cerca del fondo del valle de la pérdida. El terreno se vuelve plano, el gradiente se hace pequeño (mira el "total size", el tamaño total, en el panel del gradiente bajar de ~1 en la época 0 a ~0.03 en la 300 y ~0.002 en la 4000) y cada paso mueve muy poco los pesos. Esto se llama **convergencia**.

La forma no se queda totalmente quieta: los pesos siguen creciendo despacio (panel de pesos), y eso hace que los bordes de la franja sean más **nítidos**. La red se está volviendo más *confiada*. Con la entropía cruzada, siempre se puede bajar la pérdida un poquito más apostando con más certeza, así que los pesos crecen sin parar, cada vez más despacio.

La sección 8 muestra estas dos fases vistas desde arriba, en un mapa de la pérdida.

---

## 6. Cada neurona es una regresión lineal

Cada neurona calcula `w₁·distancia + w₂·velocidad + b`, la fórmula de una **regresión lineal**. Donde esa suma es cero, la neurona cambia de "no" a "sí", y esos puntos forman una **línea recta** en la gráfica. La activación (`tanh` o `sigmoid`) solo dobla la respuesta cerca de la recta; no cambia dónde está la recta.

### Una neurona sola: una recta no alcanza

![neurona única](results/05_single_neuron.png)

Una neurona con sigmoide y sin capa oculta es una **regresión logística**: solo puede trazar **una línea recta**. La franja necesita dos bordes, así que se rinde y aprende "nunca saltes" — lo que acierta el 77% solo porque la mayoría de los ejemplos son "no saltar". Esta es una trampa clásica: **una exactitud alta no significa que aprendió**.

### Seis neuronas: seis rectas, combinadas en una franja

![rectas de las neuronas ocultas](results/06_hidden_lines.png)

Cada recta morada es donde cambia una neurona oculta. Cuanto más gruesa la recta, más importa esa neurona para la salida.

- **`epoch 0` (época 0):** las seis rectas se cruzan en el medio de la gráfica. Los sesgos empiezan en cero, así que todas las rectas pasan por el punto central; solo los ángulos son aleatorios.
- **`epoch 50` y `epoch 300`:** las rectas se separan y giran hacia los puntos.
- **`epoch 4000`:** dos neuronas quedan exactamente sobre los dos bordes de la franja, con pesos de salida fuertes y de signos opuestos (+9.7 y −9.4): una dice "el cactus está lo bastante cerca", la otra dice "pero no demasiado cerca". Las otras cuatro movieron sus rectas fuera del área de juego (a distancias mayores que 2) y ya casi no importan.

Es decir, la red solo necesitó 2 de sus 6 neuronas, lo que coincide con lo que `N_HIDDEN = 2` muestra en los ejercicios. La capa de salida combina las dos rectas en la franja. **Más neuronas = más rectas para combinar = formas más complicadas.**

Puedes ver estas rectas moverse en la repetición (`--live`).

---

## 7. Hiperparámetros: las perillas del entrenamiento

Están al principio del script. No se aprenden — **tú** los eliges.

| Parámetro | Valor | Qué pasa si lo cambias |
|---|---|---|
| `LEARNING_RATE` | 0.5 | Muy pequeña (0.01): después de 4000 épocas todavía no salió del "nunca saltes" (78%). Muy grande (20): pasos enormes, la pérdida da saltos de hasta 10× antes de estabilizarse. |
| `N_HIDDEN` | 6 | Con 1: no logra la franja. Con 2 ya lo logra (2 rectas = 2 bordes). Más neuronas dan margen para patrones más complicados. |
| `EPOCHS` | 4000 | Pocas: se queda a medio camino (mira la época 300). |
| `N_EXAMPLES` | 600 | Con 20: 100% en el entrenamiento, pero solo ~91% con datos nuevos — la red "memorizó" los ejemplos (*overfitting*). |
| `SEED` | 42 | Cambia los pesos iniciales y los datos. Resultados finales parecidos, caminos diferentes. |

---

## 8. El valle de la pérdida: tasa de aprendizaje y programación coseno

Los 25 pesos forman un "terreno" de pérdida con 25 dimensiones, imposible de dibujar. Para ver el descenso de gradiente desde arriba, [`loss_valley.py`](loss_valley.py) reduce el problema a la neurona más simple que existe: **sin ninguna activación**, una regresión lineal pura con solo dos pesos:

```
jump distance = w · speed + b
```

(distancia del salto = w · velocidad + b.) La tarea sigue viniendo del juego: cuanto más rápido va el juego, antes hay que saltar. El script genera 200 ejemplos con ruido y baja la colina cuatro veces desde el mismo punto de partida, cambiando solo la tasa de aprendizaje.

```bash
python lessons/01-neuron-and-gradient/loss_valley.py
```

![valle de la pérdida](results/loss_valley.gif)

Cada punto del mapa es una recta posible `(w, b)`; más oscuro significa menor pérdida. La estrella dorada es el fondo del valle, la mejor recta. Después de 60 pasos:

| Recorrido | Tasa de aprendizaje | Qué pasa | Pérdida después de 60 pasos |
|---|---|---|---|
| gris | 0.05, fija | Seguro, pero demasiado lento: todavía lejos del fondo | 0.069 |
| verde | 0.5, fija | Baja y se asienta en el fondo | 0.00097 |
| rojo | 0.78, fija | Se pasa del fondo, rebota de un lado a otro, cada vez más lejos, y explota | 4.5 |
| morado | 0.78 → 0, coseno | Empieza con la misma tasa "demasiado grande" y rebota, pero la tasa se reduce y se calma en el fondo | 0.0011 |

El fondo del valle tiene una pérdida de 0.00095.

Tres cosas para notar:

- **El valle es un tazón largo y angosto.** Cruzarlo es empinado, a lo largo es suave. Un paso grande se pasa en el lado empinado y rebota; ese es el zigzag. Las redes más grandes pueden tener valles así en muchas dimensiones a la vez.
- **Las dos fases de la sección 5 aparecen aquí también.** El descenso rápido son los primeros pasos, cuando los caminos caen hacia el valle; el ajuste fino es el avance largo y lento por el fondo.
- **La programación coseno** (*cosine schedule*) toma lo mejor de los dos: pasos grandes al principio, cuando la red está lejos del fondo, y pasos pequeños al final, cuando está cerca. La tasa de aprendizaje sigue media onda de coseno, de 0.78 a 0 (gráfica de abajo a la derecha). Hasta más o menos el paso 50 el camino morado tiene menor pérdida que el verde, y termina prácticamente empatado con él. Los modelos grandes, incluidos los LLM, suelen entrenarse con una programación así en lugar de una tasa fija.

Una neurona con activación (como las de `train_from_scratch.py`) tiene un valle más curvo que este tazón perfecto, pero el recorrido es el mismo: `weight = weight - learning_rate * gradient`, una y otra vez.

---

## Ejercicios

1. Pon `LEARNING_RATE = 20` y mira el `02_loss.png`. Después prueba `0.01`. ¿Qué pasó en cada caso?
2. Prueba `N_HIDDEN = 1` y después `2`. ¿Cuál es el mínimo de neuronas para dibujar la franja? ¿Por qué? Mira el `04_boundaries.png`.
3. Pon `N_EXAMPLES = 20`. Compara la exactitud en el entrenamiento con la exactitud con datos nuevos.
4. Abre el CSV y encuentra el peso que más cambió. ¿A qué entrada está conectado?
5. En `loss_valley.py`, cambia la tasa "demasiado grande" de 0.78 a 0.75. ¿Todavía explota? Después prueba 0.8. Alrededor de 0.76 está el límite entre "rebota pero se asienta" y "explota".

---

## ¿Qué relación tiene con los LLM?

Un LLM se entrena **exactamente** con este ciclo — forward, pérdida, gradiente, paso — solo que:

- la entrada es un fragmento de texto y la salida es la probabilidad de cada palabra siguiente;
- son cientos de miles de millones de pesos en lugar de 25;
- los datos son billones de palabras en lugar de 600 puntos;
- usan variantes más ingeniosas del descenso de gradiente (como *Adam*), en miles de GPUs.

---

## Más allá de esta lección: similitud coseno

> **No se usa en esta lección.** Está aquí como analogía, porque "coseno" aparece mucho cuando se habla de LLM, y no tiene nada que ver con la programación coseno de la sección 8.

Un **vector** es solo una lista de números. Cada neurona oculta de esta lección tiene uno: sus dos pesos de entrada, `(peso de la distancia, peso de la velocidad)`. Ese vector define la dirección de la recta de la neurona.

La **similitud coseno** mide cuánto apuntan dos vectores hacia el mismo lado, sin importar su tamaño: 1 = misma dirección, 0 = sin relación, −1 = direcciones opuestas.

En la red entrenada:

- Las dos neuronas de los bordes de la franja tienen los vectores (−8.91, 3.07) y (−8.15, 1.97). Su similitud coseno es **0.996**: apuntan casi hacia el mismo lado, y por eso los dos bordes de la franja son casi paralelos.
- Otra neurona, con pesos diminutos (−0.17, 0.08), también obtiene **0.995** frente a la primera. Apunta hacia el mismo lado, pero es 50 veces más débil. La similitud coseno solo mira la dirección, no el tamaño.

Los LLM usan la misma herramienta a una escala mucho mayor. Convierten cada palabra o texto en un vector con cientos o miles de números (un *embedding*). Los textos con significado parecido terminan apuntando en direcciones parecidas, así que "perro" y "cachorro" tienen una similitud coseno alta, y "perro" y "hoja de cálculo", una baja. Así es como los buscadores y los asistentes encuentran textos con un significado parecido, incluso cuando las palabras son diferentes. La [lección 7](../07-language-model/README.es.md) lo hace de verdad, con un mini modelo de lenguaje.

---

**Próxima lección:** [el entorno — el juego que la IA va a jugar](../02-the-environment/README.es.md)
