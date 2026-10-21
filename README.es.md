[English](README.md) · [Português](README.pt-BR.md) · **Español**

# 🦖 Dino AI — aprendiendo redes neuronales paso a paso

Un estudio práctico: desde la primera neurona escrita a mano hasta una población de dinosaurios que aprende sola a jugar el juego del dinosaurio de Chrome.

Cada lección es una carpeta con un **README que explica los conceptos**, un **script corto y comentado** y los **gráficos que genera** el propio script, para que veas el aprendizaje en acción. El código, los gráficos y los mensajes de la terminal están en inglés; los README existen en inglés, portugués y español.

![campeones de varias generaciones](lessons/03-neuroevolution/results/champions.gif)

---

## Hoja de ruta

| # | Lección | Lo que aprendes | Tiempo de ejecución |
|---|---|---|---|
| 1 | [Neurona y gradiente](lessons/01-neuron-and-gradient/README.es.md) | Peso, sesgo, activación, pérdida, **gradiente**, retropropagación, hiperparámetros, sobreajuste — todo desde cero con numpy | ~5 s |
| 2 | [El entorno](lessons/02-the-environment/README.es.md) | El juego en pygame, lo que la IA "ve" (sensores), acciones y recompensa | juegas tú |
| 3 | [Neuroevolución (NEAT)](lessons/03-neuroevolution/README.es.md) | Entrenar **sin gradiente**, por selección y mutación. Registro de cada generación, gráficos de los pesos e informe automático | ~3 s sin ventana |
| 4 | [Aprendizaje por refuerzo (PPO)](lessons/04-reinforcement/README.es.md) | Un dino aprende por gradiente a partir de recompensas: actor, crítico, ventaja, clip. Dónde queda guardado el aprendizaje y la relación con el ajuste de los LLM | ~2 min sin ventana |
| 5 | [Imitación](lessons/05-imitation/README.es.md) | Juegas tú y una red aprende a copiarte, o copia al agente de la lección 4. Aprendizaje supervisado a partir de demostraciones, y por qué los LLM imitan antes de usar refuerzo | ~2 s de entrenamiento |
| 6 | [Aprender a ver](lessons/06-vision/README.es.md) | Una red convolucional (CNN) lee la pantalla en lugar de los 6 números: filtros, capas, y por qué una red falla en escenas que nunca vio (DAgger lo corrige) | ~20 min de entrenamiento; ojos entrenados incluidos |
| 7 | [Un mini modelo de lenguaje](lessons/07-language-model/README.es.md) | Un *transformer* diminuto, la arquitectura de un LLM, aprende a escribir prediciendo el siguiente carácter: tokens, atención, temperatura, *embeddings* y similitud coseno de verdad, y memorizar vs. aprender | ~30 s de entrenamiento |

Las dudas comunes, como "¿alguien le enseña la regla a la IA?" y "¿dónde queda guardado el aprendizaje?", están en [Preguntas que surgen en el camino](#preguntas-que-surgen-en-el-camino).

---

## Instalación

Necesitas **Python 3.9+**.

Copia y ejecuta un comando a la vez:

```bash
git clone https://github.com/remofurlanetto/dino-ai-study.git
```

```bash
cd dino-ai-study
```

```bash
python3 -m venv .venv
```

Activa el entorno. En Mac o Linux:

```bash
source .venv/bin/activate
```

En Windows: `.venv\Scripts\activate`

```bash
pip install -r requirements.txt
```

**Todos los comandos se ejecutan desde la raíz del repositorio**, por ejemplo:

```bash
python lessons/01-neuron-and-gradient/train_from_scratch.py --live
```

Cada vez que abras una terminal nueva, ejecuta antes `source .venv/bin/activate`, o Python no encontrará las bibliotecas.

## Paso a paso

Haz las lecciones en orden. Cada paso tiene un comando, lo que vas a ver y lo que aprendes con él. Antes que nada, en cada terminal nueva:

```bash
source .venv/bin/activate
```

### Paso 1: una red neuronal aprendiendo con gradiente

```bash
python lessons/01-neuron-and-gradient/train_from_scratch.py --live
```

La terminal muestra los primeros cálculos del gradiente, peso por peso. Después se abre la repetición del entrenamiento:

![repetición de la lección 1](lessons/01-neuron-and-gradient/results/training.gif)

Teclas: **ESPACIO** pausa, **← →** avanza o retrocede una época, **↑ ↓** cambia la velocidad, **R** reinicia.

Fíjate en que la franja roja aparece rápido y después solo se vuelve más nítida, mientras las barras del gradiente se encogen. La [lección 1](lessons/01-neuron-and-gradient/README.es.md) explica por qué.

### Paso 2: el juego que la IA va a jugar

```bash
python lessons/02-the-environment/play.py
```

Juegas tú (ESPACIO salta, ↓ se agacha) y ves en la esquina los 6 números que la IA recibe en lugar de ojos. [Lección 2](lessons/02-the-environment/README.es.md).

### Paso 3: 50 dinos evolucionando

```bash
python lessons/03-neuroevolution/train.py
```

En cada generación, 50 dinos con redes distintas juegan juntos; los mejores tienen hijos con mutaciones. Las flechas ↑ ↓ cambian la velocidad (de 1x a 64x) y V salta a la velocidad máxima:

![una generación de cada 3, acelerado](lessons/03-neuroevolution/results/evolution.gif)

Después, genera los gráficos y el informe:

```bash
python lessons/03-neuroevolution/report.py
```

Y mira al campeón de cada generación jugando, con su red dibujada al lado (← → cambia de generación):

```bash
python lessons/03-neuroevolution/watch.py
```

![campeones de varias generaciones](lessons/03-neuroevolution/results/champions.gif)

[Lección 3](lessons/03-neuroevolution/README.es.md).

### Paso 4: aprendizaje por refuerzo

```bash
python lessons/04-reinforcement/train.py
```

Un solo dino juega miles de partidas y ajusta sus pesos por gradiente para ganar más recompensa (`+0.1` por cuadro vivo, `−1` al chocar). Junta el gradiente del paso 1 con la nota del paso 3. Tarda unos 2 minutos, sin ventana. Después, genera los gráficos:

```bash
python lessons/04-reinforcement/report.py
```

Y mira al dino jugando, con la probabilidad de cada acción y el valor esperado al lado (← → cambia de modelo guardado):

```bash
python lessons/04-reinforcement/watch.py
```

![modelos de la lección 4](lessons/04-reinforcement/results/models.gif)

La lección también responde dónde queda guardado el aprendizaje (en un archivo de pesos, como un LLM) y muestra la relación con el ajuste de los LLM. [Lección 4](lessons/04-reinforcement/README.es.md).

### Paso 5: imitación

```bash
python lessons/05-imitation/train.py
```

Una red alumna aprende a jugar copiando a un profesor: aquí, 10 partidas grabadas del agente de la lección 4. Una sola partida grabada y 2 segundos de entrenamiento bastan para un jugador perfecto. Después, mira al alumno jugando junto a las elecciones del profesor:

```bash
python lessons/05-imitation/watch.py
```

![el alumno de la lección 5](lessons/05-imitation/results/student.gif)

Y enséñale tú mismo: juega algunas partidas y se graba cada cuadro.

```bash
python lessons/05-imitation/record.py
```

[Lección 5](lessons/05-imitation/README.es.md).

### Paso 6: aprender a ver

```bash
python lessons/06-vision/watch.py
```

El dino juega solo a partir de la pantalla. Una red convolucional (los ojos) lee la imagen y estima los 6 números; el agente de la lección 4 (el cerebro) decide a partir de ellos. Los ojos entrenados ya están en el repositorio; entrenarlos tú mismo tarda unos 20 a 25 minutos:

```bash
python lessons/06-vision/train_eyes.py
```

![ojos + cerebro jugando a partir de la pantalla](lessons/06-vision/results/eyes.gif)

[Lección 6](lessons/06-vision/README.es.md).

### Paso 7: un mini modelo de lenguaje

```bash
python lessons/07-language-model/train.py
```

Un *transformer* diminuto, la misma arquitectura de un LLM, lee los README de este estudio y aprende a predecir el siguiente carácter. En unos 30 segundos pasa de letras al azar a un texto que suena a inglés. Después, haz que escriba:

```bash
python lessons/07-language-model/generate.py
```

![el mini modelo de lenguaje escribiendo](lessons/07-language-model/results/generation.gif)

[Lección 7](lessons/07-language-model/README.es.md).

### Grabar las animaciones

Los GIF de esta página los generan los propios scripts:

```bash
python lessons/01-neuron-and-gradient/train_from_scratch.py --gif
```

```bash
python lessons/03-neuroevolution/train.py --gif
```

```bash
python lessons/03-neuroevolution/watch.py --gif
```

```bash
python lessons/04-reinforcement/watch.py --gif
```

```bash
python lessons/06-vision/watch.py --gif
```

---

## Estructura

```
dino-ai-study/
├── dino/game.py                      # el juego (usado en las lecciones 2, 3 y 4)
├── dino/sprites.py                   # las figuras en pixel art, editables como texto
└── lessons/
    ├── 01-neuron-and-gradient/
    │   ├── README.md                 # la teoría, con los gráficos
    │   ├── train_from_scratch.py     # red neuronal solo con numpy
    │   └── results/                  # gráficos + CSV con los pesos de cada época
    ├── 02-the-environment/
    │   ├── README.md
    │   └── play.py                   # juegas tú y ves los sensores de la IA
    ├── 03-neuroevolution/
    │   ├── README.md
    │   ├── neat-config.txt           # los parámetros de la evolución, comentados
    │   ├── train.py                  # entrena y registra cada generación
    │   ├── report.py                 # genera gráficos + REPORT.md
    │   ├── watch.py                  # mira al campeón de cualquier generación jugando
    │   └── results/                  # history.json, gráficos, REPORT.md
    ├── 04-reinforcement/
    │   ├── README.md
    │   ├── environment.py            # el juego en formato Gymnasium (observación, acción, recompensa)
    │   ├── train.py                  # entrena con PPO y guarda el modelo en varios momentos
    │   ├── report.py                 # gráficos, mapa de decisión + REPORT.md
    │   ├── watch.py                  # el dino jugando, con su "cabeza" al lado
    │   └── results/models/           # el aprendizaje: archivos con los pesos de la red
    ├── 05-imitation/
    │   ├── README.md
    │   ├── record.py                 # juegas tú (o juega el agente de la lección 4) y se graba cada cuadro
    │   ├── train.py                  # una red alumna aprende a imitar las grabaciones
    │   ├── experiments.py            # profesores fuerte, débil y lento: ¿el alumno queda igual de bien?
    │   ├── watch.py                  # el alumno jugando, junto a las elecciones del profesor
    │   └── results/demos/            # las demostraciones grabadas (CSV)
    ├── 06-vision/
    │   ├── README.md
    │   ├── vision.py                 # la cámara, los ojos (CNN) y el agente de la lección 4 como cerebro
    │   ├── train_eyes.py             # entrena los ojos, con rondas de DAgger
    │   ├── pixels_to_keys.py         # el otro enfoque: de los píxeles directo a las teclas
    │   ├── watch.py                  # ojos + cerebro jugando, con lo que ven los ojos
    │   └── results/eyes.pt           # los ojos entrenados
    └── 07-language-model/
        ├── README.md
        ├── tiny_lm.py                # el texto, el bigrama y el transformer diminuto
        ├── train.py                  # lo entrena y guarda el mejor modelo en texto nuevo
        ├── generate.py               # hace que escriba, con temperatura
        ├── inside.py                 # embeddings, similitud coseno y atención
        └── data/study.txt            # el texto de entrenamiento: los README de este estudio
```

Cada README existe en tres versiones: `README.md` (inglés), `README.pt-BR.md` (portugués) y `README.es.md` (español).

---

## Glosario rápido

| Término | En inglés | En una frase |
|---|---|---|
| **Peso** | weight | Número que dice cuánto importa una entrada para una neurona. Es lo que se ajusta en el entrenamiento. |
| **Sesgo** | bias | Número que se suma a la neurona y desplaza el punto en que se "enciende". |
| **Activación** | activation | Curva aplicada a la neurona (tanh, sigmoide...). Sin ella, la red solo podría trazar rectas. |
| **Pérdida** | loss | Número que mide cuánto se equivoca la red. El entrenamiento intenta reducirlo. |
| **Gradiente** | gradient | Para cada peso: hacia qué lado y cuánto moverlo para reducir la pérdida. |
| **Retropropagación** | backpropagation | La forma eficiente de calcular el gradiente, de la salida hacia la entrada. |
| **Tasa de aprendizaje** | learning rate | Tamaño del paso que se da en la dirección del gradiente. |
| **Época** | epoch | Una pasada por todos los ejemplos de entrenamiento. |
| **Hiperparámetro** | hyperparameter | Un ajuste que eliges tú, no que se aprende (tasa, tamaño de la red...). |
| **Sobreajuste** | overfitting | Memorizar los ejemplos de entrenamiento en lugar de aprender el patrón general. |
| **Fitness / recompensa** | fitness / reward | La nota de qué tan bien le fue a la IA en el entorno. |
| **Generación** | generation | Una ronda de la evolución: todos juegan, los mejores se reproducen. |
| **Mutación** | mutation | Un cambio aleatorio en un peso o en la estructura de la red. |
| **Especie** | species | Grupo de redes parecidas que compiten entre sí en NEAT. |

Los cinco primeros (peso, sesgo, activación, pérdida y gradiente) se explican con calma, con analogías, gráficos y una animación, en la [explicación a fondo de la lección 1](lessons/01-neuron-and-gradient/concepts/README.es.md).

---

## Preguntas que surgen en el camino

**¿Alguien le enseña la regla a la IA?**
No. En ninguna lección la IA recibe la regla "salta cuando el cactus esté cerca". En la lección 1, la regla existe en el código solo para generar 600 ejemplos ya respondidos, y la red aprende de los ejemplos sin ver nunca la regla. En las lecciones 3 y 4 ni siquiera eso: la IA solo recibe una nota o una recompensa, que dice si salió bien, nunca cómo hacerlo.

**¿El dino aprende solo a ver dónde están los obstáculos?**
No, y es a propósito. En ninguna lección el dino ve la pantalla. El código del juego lo mide todo y entrega 6 números listos: la distancia, el ancho, la altura y la elevación del próximo obstáculo, la altura del dino y la velocidad. Es como darle un sensor de distancia en lugar de ojos. Lo que aprende solo es **qué hacer** con esos números: cuándo saltar, cuándo agacharse, cuándo esperar. Aprender a ver a partir de los píxeles de la pantalla, como hizo DeepMind con los juegos de Atari, requiere una red convolucional y mucho más entrenamiento. Es la [lección 6](lessons/06-vision/README.es.md): allí, una CNN aprende a estimar esos 6 números a partir de la pantalla, y el dino juega solo con los píxeles.

**¿Y la lección 1, entonces?**
En la lección 1 ni siquiera hay un dino jugando. Es un ejercicio con ejemplos listos (las *respuestas correctas*, los puntos de los gráficos) para ver el gradiente funcionando de cerca. Los que de verdad aprenden a jugar, sin que nadie sepa la respuesta, son los dinos de las lecciones 3 y 4.

**Entonces, ¿el aprendizaje es aleatorio?**
Solo en parte. Los pesos empiezan sorteados en todas las lecciones. En las lecciones 1 y 4, a partir de ahí el gradiente calcula hacia qué lado mover cada peso, sin sorteos. En la lección 3, las mutaciones son aleatorias, pero la selección no: los dinos que llegan más lejos tienen hijos. El azar propone y la nota decide. Por eso cada semilla da números distintos, pero la regla principal ("lejos = no saltes") siempre aparece.

**¿Por qué en la repetición la forma aparece rápido y después casi no cambia?**
Al principio la red se equivoca mucho, el gradiente es grande y cada paso mueve bastante los pesos. Cerca del fondo del "valle" de la pérdida, el gradiente se vuelve diminuto (de ~1 a ~0,002 en la lección 1) y los pesos casi se detienen. Esto se llama convergencia. La [lección 1](lessons/01-neuron-and-gradient/README.es.md) tiene la explicación completa, incluida la relación con la regresión lineal.

**¿Por qué el entrenamiento de la lección 4 se estanca y después da un salto?**
Durante unos 600 mil pasos, el dino queda atrapado en "saltar sin parar", que da unos 150 puntos y es difícil de abandonar (un *óptimo local*). Cuando descubre que esperar a que el cactus se acerque rinde mucho más, el promedio se dispara hasta casi 5000. En las primeras pruebas, sin ajustes, no salió de la meseta en 500 mil pasos y, con 2 millones, solo salió cerca del final. Lo que adelantó el salto fue hacer que cada acción durara 2 cuadros y dar un bono mayor por explorar.

**¿La IA hace lo que haría un humano?**
No siempre. Frente a un pájaro a la altura de la cabeza, el dino de la lección 4 se queda agachado mientras el pájaro está lejos y salta por encima cuando se acerca. Nadie le enseñó eso, y funciona. La recompensa premia el resultado, no el método.

**¿Por qué el dino juega mejor en la prueba que en el entrenamiento?**
Durante el entrenamiento, el agente de la lección 4 todavía sortea acciones para explorar, así que a veces se equivoca a propósito. En la prueba siempre elige la acción más probable. Por eso el promedio del entrenamiento queda en ~4300, mientras que el modelo final hizo 5000 en las 10 partidas nuevas.

**Una vez entrenado, ¿el aprendizaje queda guardado?**
Sí, en un archivo de pesos. El dino de la lección 4 es el archivo `lessons/04-reinforcement/results/models/ppo_02000k.zip`: unos 62 KB con 2692 números. Cargar ese archivo es tener el dino que sabe jugar, sin volver a entrenar. En la lección 1, la última fila de `weight_history.csv` tiene los pesos finales; en la lección 3, `champions.pkl` guarda al campeón de cada generación.

**¿Esto se parece a un LLM?**
Es la misma idea a una escala gigantesca. Un LLM también es un archivo de pesos, solo que con miles de millones de números y de unos pocos a cientos de gigabytes. Y, como aquí, usarlo no cambia el archivo: ejecutar `watch.py` mil veces no le enseña nada al dino, y un LLM no aprende mientras conversa contigo. Solo cambia cuando un nuevo entrenamiento genera un archivo nuevo, es decir, una nueva versión del modelo.

**¿Qué es el overfitting?**
Es cuando un modelo **memoriza sus ejemplos de entrenamiento en lugar de aprender el patrón que hay detrás**. Le va muy bien con lo que ya vio y mal con cualquier cosa nueva, como un alumno que memoriza las respuestas de la guía de ejercicios y después reprueba el examen porque las preguntas cambiaron. Aparece dos veces en este estudio: en la lección 1, una red entrenada con solo 20 ejemplos acertó el 100 % de ellos, pero solo cerca del 91 % de situaciones nuevas; en la lección 7, el modelo de lenguaje siguió mejorando en su texto de entrenamiento mientras empeoraba en texto que nunca había visto. Por eso cada lección aparta una parte de los datos: es la única forma de distinguir aprender de memorizar.

**Entonces, ¿un LLM "piensa" de verdad, o solo repite lo que memorizó?**
Memoriza un poco, pero en su mayor parte no. Un LLM se entrena con mucho más texto del que sus pesos podrían guardar palabra por palabra; para que le vaya bien, tiene que quedarse con **patrones**: gramática, datos comunes, cómo suele ser una explicación o un programa. El modelo diminuto de la lección 7 hizo lo mismo a su escala: nadie le dijo que "3" y "4" son números, pero los agrupó porque eso le ayudaba a predecir. Y la copia se puede medir: la [lección 7](lessons/07-language-model/README.es.md) compara cada carácter que escribe el modelo con el texto de entrenamiento y pinta las partes copiadas. En su mejor momento, el modelo escribe casi solo combinaciones nuevas; entrenado demasiado tiempo sin *dropout*, la mitad de lo que escribe está pegado. Los LLM grandes también muestran los dos comportamientos: pueden repetir algunos textos palabra por palabra, sobre todo los que vieron muchas veces, y empeoran lejos de lo que vieron, como los ojos de la lección 6. Si eso cuenta como "pensar" es una pregunta abierta, que depende de lo que entendamos por la palabra. Lo que se puede medir es que generalizan mucho más allá de memorizar, con aciertos impresionantes y errores extraños.

---

## ¿Qué relación tiene con los LLM?

Usan los mismos bloques de estas lecciones, a una escala enorme:

1. **Preentrenamiento** — el ciclo de la **Lección 1** (forward, pérdida, gradiente, paso), con miles de millones de pesos aprendiendo a predecir la siguiente palabra en billones de textos. En cierto modo ya es imitación: el modelo copia cómo escriben las personas. La [lección 7](lessons/07-language-model/README.es.md) construye uno así, en miniatura.
2. **Ajuste supervisado** (*supervised fine-tuning*) — la **Lección 5**: personas escriben ejemplos de buenas respuestas y el modelo aprende a imitarlos.
3. **Aprendizaje por refuerzo** — el ciclo de la **Lección 4** (actuar, recibir una recompensa, ajustar por gradiente): el modelo genera respuestas, recibe una nota (de personas o de pruebas automáticas) y se ajusta para sacar mejores notas.

La neuroevolución de la **Lección 3** no se usa para entrenar LLM (sería demasiado lenta con miles de millones de pesos), pero es la forma más visual de entender que **"aprender" es ajustar números hasta que una nota mejore**.

---

## Bibliotecas usadas

- [numpy](https://numpy.org/) — cálculos con matrices
- [matplotlib](https://matplotlib.org/) — gráficos
- [pygame](https://www.pygame.org/) — el juego (ventana, dibujo, teclado)
- [neat-python](https://github.com/CodeReclaimers/neat-python) — el algoritmo NEAT
- [Gymnasium](https://gymnasium.farama.org/) — el formato estándar de entornos de refuerzo
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — el algoritmo PPO (usa [PyTorch](https://pytorch.org/), que es una descarga grande)
