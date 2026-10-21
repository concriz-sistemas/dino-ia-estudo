[English](README.md) · [Português](README.pt-BR.md) · **Español**

# Lección 3 — Neuroevolución: 50 dinos evolucionando hasta jugar bien

> **Objetivo:** entrenar redes neuronales para jugar **sin gradiente**, solo con selección natural,
> y registrar cada generación para ver cómo cambian los pesos.

![una generación de cada 3, acelerado](results/evolution.gif)

*El entrenamiento de ejemplo acelerado, mostrando una generación de cada 3 (0, 3, 6 … 27 y la última, 29): 50 dinos por generación, los primeros 480 puntos de cada partida. Arriba a la derecha, el mejor resultado de cada generación hasta ese momento, incluidas las que el GIF salta; la barra se pone verde cuando un dino llega a 5000.*

![campeones de varias generaciones](results/champions.gif)

*El campeón de las generaciones 0, 7, 14 y 29 jugando la misma partida. El de la generación 0 choca enseguida (129 puntos); los demás siguen vivos cuando termina su fragmento (el GIF corta cada fragmento en 600 puntos).*

Ejecuta uno a la vez, en este orden.

1. Mira la evolución. Las flechas ↑ ↓ cambian la velocidad (de 1x a 64x), V salta a la velocidad máxima y vuelve, y ESC la detiene y guarda lo que ya se registró:

```bash
python lessons/03-neuroevolution/train.py
```

2. Genera las gráficas y el `REPORT.md`:

```bash
python lessons/03-neuroevolution/report.py
```

3. Mira al campeón de la generación 0:

```bash
python lessons/03-neuroevolution/watch.py --generation 0
```

4. Y compáralo con el campeón de la última generación. Las flechas ← → cambian de generación, y R empieza una partida nueva:

```bash
python lessons/03-neuroevolution/watch.py
```

Para entrenar sin ventana (unos 3 segundos): `train.py --fast`. Para cambiar el número de generaciones (30 por defecto): `train.py --generations 50`.

Para volver a grabar el GIF de la evolución (entrena sin ventana y graba una generación de cada 3):

```bash
python lessons/03-neuroevolution/train.py --gif
```

Para volver a grabar el GIF de los campeones, después de entrenar:

```bash
python lessons/03-neuroevolution/watch.py --gif
```

📊 **[Mira el informe de un entrenamiento de ejemplo](results/REPORT.md)** (en inglés)

---

## La idea: evolución en lugar de gradiente

En la [Lección 1](../01-neuron-and-gradient/README.es.md), cada peso se ajustaba **calculando** hacia qué lado bajaba la pérdida. Aquí no hay ningún cálculo:

```
1. Crea 50 redes con pesos aleatorios
2. Pon a todas a jugar la misma partida
3. Nota de cada una = cuántos cuadros sobrevivió
4. Las mejores se vuelven "padres"; los hijos heredan los pesos con pequeñas mutaciones aleatorias
5. Vuelve al paso 2
```

Nadie programó "salta cuando el cactus esté cerca". Esa regla **surge** porque, por casualidad, algunos dinos saltan en el momento justo, sobreviven y pasan sus pesos a la siguiente generación.

| | Gradiente (Lección 1) | Neuroevolución (esta lección) |
|---|---|---|
| Cómo decide cambiar un peso | Calcula la derivada de la pérdida | Mutación aleatoria + selección |
| ¿Necesita la "respuesta correcta"? | Sí, ejemplos etiquetados | No, solo una nota al final |
| ¿Cambia la estructura de la red? | No, es fija | Sí, crea y elimina neuronas |
| ¿Escala a miles de millones de pesos? | Sí (así se entrenan los LLMs) | No, se vuelve demasiado lento |
| Bueno para | Muchos datos, redes grandes | Problemas pequeños, sin gradiente disponible |

---

## NEAT

**NEAT** (*NeuroEvolution of Augmenting Topologies*, 2002) es el algoritmo que se usa aquí, a través de la biblioteca `neat-python`. Tiene tres trucos:

1. **Empieza mínimo:** solo entradas conectadas a las salidas, sin capa oculta. Las neuronas y conexiones se **agregan por mutación** cuando ayudan.
2. **Genes con historial:** cada conexión tiene un número de "innovación", para que dos padres puedan cruzar redes de formas distintas sin revolver todo.
3. **Especies:** las redes parecidas compiten entre sí. Una innovación recién creada (que al principio suele empeorar la red) gana tiempo para madurar sin que la eliminen los campeones ya establecidos.

---

## Qué se registra

`train.py` tiene un `Recorder` (registrador) que, al final de **cada generación**, guarda en `results/history.json`:

- la mejor nota, el promedio, la mediana y la desviación de la población;
- las especies y sus tamaños;
- el **campeón completo**: cada neurona con su sesgo y cada conexión con su peso;
- **todos los pesos** de toda la población.

`history.json` es texto plano. Ábrelo en un editor y busca `"champion"` (campeón) para ver una red entera en números.

`report.py` convierte esto en gráficas y en [`REPORT.md`](results/REPORT.md):

| Gráfica | Qué muestra |
|---|---|
| `01_fitness.png` | Mejor, promedio y mediana por generación |
| `02_species.png` | Cómo se divide la población en especies |
| `03_complexity.png` | Cuántas neuronas y conexiones tiene el campeón |
| `04_champion_weights.png` | El valor de cada conexión del campeón final a lo largo de las generaciones |
| `05_weight_distribution.png` | Todos los pesos de todos los dinos, generación por generación |
| `06_networks.png` | Dibujo de la red del campeón en 4 momentos |

Los títulos y leyendas de las gráficas están en inglés: *best dino* = mejor dino, *population average* = promedio de la población, *median* = mediana.

---

## Qué observar en el informe de ejemplo

NEAT sortea mutaciones, pero `train.py` fija el punto de partida de ese sorteo con `SEED = 0` (la semilla): inicializa el `random` de Python antes de crear la población. Por eso **tu entrenamiento da exactamente los mismos números** que el entrenamiento de ejemplo que está en [`results/REPORT.md`](results/REPORT.md) (30 generaciones, 50 dinos por generación). Los valores entre paréntesis son de ese ejemplo. Si cambias la `SEED`, sale otra evolución, con otros números, pero los patrones se repiten:

- **El mejor sube a saltos, el promedio sube despacio.** Un dino con suerte llega a 5000 pronto (generación 6 en el ejemplo), pero a la población entera le toma mucho más tiempo mejorar (el promedio va de 121 en la generación 0 a 1084 en la última).
- **El mejor de una generación puede ser peor que el de la anterior** (669 → 316 → 235 en las generaciones 0, 1 y 2 del ejemplo; y después de llegar a 5000 en la generación 6, la generación 7 bajó a 1477, y luego dio 5000 en todas las generaciones de la 8 a la 29). Cada generación juega una partida **distinta**, con otra secuencia de obstáculos, y a veces la partida es más difícil. Esto es a propósito: si todas jugaran el mismo nivel, los dinos memorizarían el nivel en lugar de aprender a jugar (*overfitting*).
- **La mediana se queda baja** aunque el promedio suba. La mayoría de los hijos sufre mutaciones malas y muere pronto. La evolución es desperdicio: muchos lo intentan, pocos lo logran.
- **El tamaño de la red varía de un entrenamiento a otro.** A veces las mutaciones cortan conexiones y el campeón queda más simple de lo que empezó, con 18 conexiones (6 conexiones y 1 neurona oculta en el ejemplo); a veces gana neuronas ocultas. Mira `03_complexity.png` y `06_networks.png`. NEAT no busca la red "más bonita", solo una que funcione.
- **La regla principal siempre aparece: obstáculo lejos = no saltes.** En el ejemplo aparece directa, como una conexión `distance` (distancia) → `jump` (saltar) con peso **−5.204**, el más fuerte del campeón final. Con otras semillas puede pasar por una neurona oculta: `distance` → neurona oculta con peso positivo, y neurona oculta → `jump` con peso negativo. Positivo por negativo da negativo, o sea, la misma regla por otro camino. Nadie les enseñó esto a los dinos.

Los otros pesos fuertes del campeón final del ejemplo: `speed` (velocidad) → `duck` (agacharse) (−1.921), `speed` → `nothing` (no hacer nada) (−0.993), `hidden 178` (oculta 178) → `jump` (+0.632) y `elevation` (elevación) → `hidden 178` (+0.585). La tabla completa está en la sección 6 del informe.

---

## Los parámetros (`neat-config.txt`)

Los más importantes para experimentar (la lista completa, con explicación, sale en el informe):

| Parámetro | Valor | Qué hace |
|---|---|---|
| `pop_size` | 50 | Dinos por generación. Más = más oportunidades por generación, pero cada generación tarda más. |
| `weight_mutate_rate` | 0.8 | Probabilidad de que cada peso sufra una mutación. |
| `weight_mutate_power` | 0.5 | Tamaño de la mutación. Es el "equivalente" a la tasa de aprendizaje. |
| `node_add_prob` | 0.2 | Probabilidad de que nazca una neurona nueva. |
| `conn_add_prob` | 0.5 | Probabilidad de que nazca una conexión nueva. |
| `compatibility_threshold` | 3.0 | Menor = más especies. |
| `elitism` | 2 | Los 2 mejores pasan intactos a la siguiente generación (una red buena nunca se pierde por mutación). |
| `survival_threshold` | 0.2 | Solo el 20% mejor de cada especie se reproduce. |

## Ejercicios

1. Ejecuta `watch.py --generation 0` y después `watch.py`. Usa ← → para recorrer las generaciones y ver en qué momento el dino "entiende" a los pájaros.
2. Cambia `pop_size` a 10 y después a 200. Ejecuta `train.py --fast` y `report.py`. ¿En qué generación llega a 5000? ¿Y en cuánto tiempo real?
3. Pon `weight_mutate_power = 3.0`. ¿Qué pasa con la gráfica del promedio?
4. En `train.py`, cambia `seed=self.n` por `seed=1` (todas las generaciones juegan el mismo nivel). Entrena y usa `watch.py`, que siempre sortea un nivel nuevo. ¿El campeón sigue jugando bien?
5. Cambia la nota: en `eval_genomes`, pon `genome.fitness = dino.score - 10 * (número de saltos)`. ¿Cambia el estilo de juego? (Vas a tener que contar los saltos.)
6. En `train.py`, cambia `SEED = 0` por otro número y ejecuta `train.py --fast` y `report.py`. Compara tu nuevo `REPORT.md` con el del ejemplo: ¿en qué generación llega a 5000? ¿Qué tamaño tiene el campeón final? ¿La regla `distance` → `jump` aparece directa o a través de una neurona oculta? (Esto sobrescribe la carpeta `results/`; vuelve a `SEED = 0` y entrena de nuevo para recuperar el ejemplo.)

**Siguiente lección:** [aprendizaje por refuerzo con gradiente (PPO)](../04-reinforcement/README.es.md)
