[English](README.md) · [Português](README.pt-BR.md) · **Español**

# Lección 4 — Aprendizaje por refuerzo (PPO)

> **Objetivo:** juntar el **gradiente** de la lección 1 con la **nota** de la lección 3. Un solo dino juega miles de partidas y ajusta los pesos de la red, por gradiente, para ganar más recompensa.
> Es la misma familia de técnicas que se usa para ajustar los LLM después del preentrenamiento.

![modelos jugando](results/models.gif)

*El mismo dino con 0, 500 mil, 1 millón y 2 millones de pasos de entrenamiento, en la misma partida. El de 0k choca con 86 puntos y el de 500k con 129; los otros dos siguen vivos cuando termina el clip (cada clip se detiene en 700 puntos). A la derecha, la probabilidad que la red le da a cada acción ("Chance of each action") y el valor que espera ganar ("Expected value:").*

Ejecuta uno a la vez, en este orden.

1. Entrena (unos 80 segundos en la Mac usada para el ejemplo, sin ventana):

```bash
python lessons/04-reinforcement/train.py
```

2. Genera las gráficas y el `REPORT.md`:

```bash
python lessons/04-reinforcement/report.py
```

3. Mira al modelo final jugando. Las flechas ← → cambian de modelo guardado, y R empieza una partida nueva:

```bash
python lessons/04-reinforcement/watch.py
```

4. Compara con el modelo antes de entrenar:

```bash
python lessons/04-reinforcement/watch.py --model 0
```

Los modelos de un entrenamiento de ejemplo ya están en el repositorio, así que el paso 3 funciona aunque no entrenes.

Las gráficas, el reporte y la ventana del juego están en inglés; los textos citados aquí aparecen tal como están en pantalla, con la explicación en español.

---

## 1. Tres formas de aprender

| | [Lección 1](../01-neuron-and-gradient/README.es.md) — gradiente | [Lección 3](../03-neuroevolution/README.es.md) — neuroevolución | Lección 4 — refuerzo |
|---|---|---|---|
| Quién juega | nadie (ejemplos listos) | 50 dinos por generación | 1 dino, miles de partidas |
| Qué recibe la IA | la respuesta correcta de cada ejemplo | una nota al final de la partida | una recompensa en **cada paso** |
| Cómo cambian los pesos | gradiente | mutación aleatoria + selección | gradiente |
| ¿Alguien tiene que saber la respuesta? | sí | no | no |

El refuerzo queda en medio: como en la lección 3, nadie dice cuál es la jugada correcta; como en la lección 1, los pesos cambian por gradiente, sin sorteo de mutaciones.

---

## 2. El entorno: observación, acción, recompensa

[`environment.py`](environment.py) envuelve el juego en el formato estándar **Gymnasium**, que cualquier algoritmo de refuerzo entiende:

- **Observación:** los mismos 6 sensores de las lecciones anteriores.
- **Acción:** 0 = nada (`nothing`), 1 = saltar (`jump`), 2 = agacharse (`duck`).
- **Recompensa:** `+0.1` por cuadro vivo y `−1` al chocar. Es el único "profesor". El agente no sabe qué es un cactus; solo sabe que algunos números en la entrada vienen antes de la recompensa mala.

Cada partida tiene obstáculos sorteados de nuevo, así que el agente tiene que **generalizar** en lugar de memorizar un nivel.

**Repetición de acción:** cada decisión dura 2 cuadros del juego (`ACTION_REPEAT = 2`). Con menos decisiones por partida, es más fácil relacionar una acción con su consecuencia. DeepMind hacía lo mismo con 4 cuadros en los juegos de Atari.

---

## 3. Actor y crítico: dos redes

PPO entrena **dos redes pequeñas** al mismo tiempo, las dos con 6 entradas y dos capas de 32 neuronas:

- **Actor (1379 pesos):** mira los sensores y da la probabilidad de cada acción. Es el que juega.
- **Crítico (1313 pesos):** mira los sensores y estima el **valor**: cuánta recompensa falta por venir a partir de ahí.

En `watch.py`, el valor del modelo entrenado ("Expected value:", valor esperado, en el panel) se queda alrededor de **19.5**. No es casualidad: cada paso vale 0.2 (dos cuadros de 0.1) y el descuento `GAMMA = 0.99` hace que el futuro lejano valga cada vez menos, así que la suma de todo lo que viene da 0.2 / (1 − 0.99) = 20. Ese es el techo, y el valor que se muestra queda justo por debajo. Cuando se acerca un obstáculo difícil y el crítico cree que algo puede salir mal, el valor baja.

---

## 4. Cómo aprende PPO

Un ciclo, repetido unas 245 veces en un entrenamiento de 2 millones de pasos:

1. **Juega:** 8 partidas en paralelo, 1024 pasos cada una. El actor **sortea** las acciones según las probabilidades que da (así es como explora).
2. **Compara con lo esperado:** para cada paso, calcula la **ventaja**: ¿dio más o menos recompensa de lo que el crítico esperaba?
3. **Ajusta por gradiente:** aumenta la probabilidad de las acciones con ventaja positiva y disminuye la de las que tienen ventaja negativa. El crítico también se ajusta para equivocarse menos la próxima vez.
4. **Frena:** el *clip* (la "P" de *Proximal*) impide que la probabilidad de cualquier acción cambie más de ~20% en una actualización. Sin ese freno, un lote de partidas con suerte podría arruinar una política que ya funcionaba.

El paso 3 es exactamente el descenso de gradiente de la lección 1. La diferencia es **qué** se intenta disminuir: allá era el error contra la respuesta correcta; aquí es "menos recompensa de la que podría haber".

---

## 5. Qué observar en el reporte de ejemplo

El entrenamiento de ejemplo está en [`results/REPORT.md`](results/REPORT.md). Cada entrenamiento da números un poco distintos, pero los patrones se repiten.

![puntos](results/01_scores.png)

- **Meseta, después salto.** Durante unos 600 mil pasos el agente se queda entre 100 y 250 puntos: saltar sin parar es una estrategia "suficientemente buena" y difícil de abandonar (un *óptimo local*). De repente descubre que esperar a que el cactus se acerque rinde mucho más, y el promedio se dispara (el promedio móvil, "mean of the last 50" en la gráfica, llega a 300 por primera vez en ~595 mil pasos, a 1000 en ~686 mil y a 4000 en ~867 mil).
- **Entrenamiento vs. prueba.** Durante el entrenamiento los puntos quedan entre ~4200 y ~4900 (promedio de las últimas 200 partidas: 4329), por debajo del máximo, porque el actor todavía sortea acciones para explorar. En las pruebas, `report.py` y `watch.py` siempre eligen la acción más probable.

![evaluación](results/03_evaluation.png)

- **Cada modelo guardado en 10 partidas nuevas.** Los modelos de 0k, 250k y 500k pasos promedian 86, 116 y 145 puntos. El primer modelo guardado con promedio de 4000 o más es el de 750k, y del 750k en adelante todos hicieron 5000 en las 10 partidas nuevas (5000 es el `SCORE_LIMIT`, donde termina la partida).

![ppo](results/02_ppo.png)

- **La entropía ("Entropy") se desploma junto con el salto de los puntos:** el agente deja de elegir al azar y se vuelve decidido.

![decisiones](results/04_decisions.png)

- **Arriba:** la red aprendió una frontera inclinada (saltar cuando el cactus se acerca, y antes cuanto más rápido va el juego), parecida a la de la lección 1. Solo que aquí nadie le dio ejemplos.
- **Abajo:** con el pájaro a la altura de la cabeza, el agente se queda agachado (`duck`) mientras está lejos y **salta por encima** (`jump`) cuando se acerca. No es lo que haría un humano, pero funciona. La recompensa dice si salió bien, nunca cómo hacerlo.

---

## 6. ¿Dónde queda guardado el aprendizaje?

En un **archivo de pesos**. Después de entrenado, el modelo es solo eso: números. Cargar el archivo = tener el dino que sabe jugar.

| Lección | Archivo | Qué tiene adentro |
|---|---|---|
| [1](../01-neuron-and-gradient/README.es.md) | `results/weight_history.csv` | los 25 pesos en cada época; la última fila es la red entrenada |
| [3](../03-neuroevolution/README.es.md) | `results/champions.pkl` | el genoma del campeón de cada generación (estructura + pesos) |
| 4 | `results/models/ppo_02000k.zip` | `policy.pth` con los 2692 pesos del actor y del crítico (unos 62 KB) |

Un LLM es **lo mismo a escala gigante**: un archivo (generalmente `.safetensors`) con miles de millones de pesos, que ocupa desde unos pocos hasta cientos de gigabytes. La estructura de la red es otra (un *transformer*), pero la idea es la misma: el entrenamiento ajusta los números, y después el archivo se carga para usarlo.

Y, como aquí, **usarlo no cambia el archivo**. Ejecutar `watch.py` mil veces no le enseña nada al dino; solo un nuevo `train.py` cambia los pesos. Un LLM tampoco aprende mientras conversa contigo: lo que "aprende" viene de nuevas rondas de entrenamiento, que generan un nuevo archivo de pesos, es decir, una nueva versión del modelo.

---

## 7. Hiperparámetros (al inicio de `train.py` y `environment.py`)

| Parámetro | Valor | Qué hace |
|---|---|---|
| `REWARD_ALIVE` / `REWARD_CRASH` | 0.1 / −1 | Lo que el agente intenta maximizar. Cambiar esto cambia el comportamiento. |
| `ACTION_REPEAT` | 2 | Cuadros por decisión. |
| `N_ENVS` | 8 | Partidas jugadas en paralelo para recolectar experiencia. |
| `STEPS_PER_ROLLOUT` | 1024 | Pasos por partida antes de cada actualización. |
| `LEARNING_RATE` | 0.0003 | Tamaño del paso del gradiente, como en la lección 1. |
| `GAMMA` | 0.99 | Cuánto vale el futuro. Cerca de 1 = ve lejos. |
| `ENTROPY_COEF` | 0.03 | Bono por seguir explorando. Ayudó a salir de la meseta de "saltar siempre". |
| `CLIP` | 0.2 | Cuánto puede cambiar la política por actualización. |
| `LAYERS` | [32, 32] | Tamaño de las redes del actor y del crítico. |

---

## Ejercicios

1. En `environment.py`, pon `JUMP_COST = 0.01`. ¿El dino salta menos? Míralo con `watch.py`.
2. Pon `ACTION_REPEAT = 1` y entrena de nuevo. ¿Cuánto dura la meseta ahora?
3. Pon `ENTROPY_COEF = 0`. ¿El agente logra salir de "saltar siempre"?
4. Cambia `GAMMA` a 0.9. ¿Cerca de cuánto va a quedar el valor esperado en `watch.py`? (Pista: rehaz la cuenta de la sección 3.)
5. Quita el sensor de velocidad: en `dino/game.py`, en `sensors()`, cambia `speed = self.speed / MAX_SPEED` por `speed = 0`. ¿La frontera de `04_decisions.png` sigue inclinada?

---

## ¿Qué relación tiene con los LLM?

En el ajuste de un LLM con refuerzo (RLHF y similares), las piezas son las mismas:

| Aquí | En el LLM |
|---|---|
| observación: 6 sensores | el texto hasta ahora |
| acción: nada, saltar, agacharse | el siguiente pedazo de palabra (*token*) |
| partida | una respuesta completa |
| recompensa: sobrevivir | la nota de un modelo entrenado con evaluaciones de personas, o "el código pasó las pruebas" |
| actor | el propio LLM |
| crítico | una red que estima la nota esperada |
| clip de PPO | un freno para que el modelo no se aleje demasiado del original |

PPO fue el algoritmo usado en las primeras versiones de RLHF. Hoy hay variaciones, pero el ciclo sigue siendo el de esta lección: **intentar, recibir una nota, ajustar los pesos por gradiente**.

---

**Lección anterior:** [neuroevolución](../03-neuroevolution/README.es.md) · **Inicio del estudio:** [todas las lecciones](../../README.es.md) · **Siguiente lección:** [imitación](../05-imitation/README.es.md)
