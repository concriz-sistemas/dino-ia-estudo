[English](README.md) · [Português](README.pt-BR.md) · **Español**

# Lección 5 — Imitación: tú le enseñas al dino jugando

> **Objetivo:** entrenar una red **copiando a un profesor** en lugar de perseguir una recompensa.
> El profesor puedes ser tú (juegas y se graba cada cuadro) o el agente de la lección 4.
> Es aprendizaje supervisado, como en la lección 1, con las teclas del profesor como clave de respuestas.

![el alumno jugando](results/student.gif)

*El alumno jugando solo. A la derecha, la probabilidad que le da a cada acción, lo que el profesor presionaría en la misma situación y con qué frecuencia coinciden.*

Ejecuta uno a la vez, en este orden.

1. Entrena al alumno con las demostraciones de ejemplo (10 partidas jugadas por el agente de la lección 4, que ya están en el repositorio):

```bash
python lessons/05-imitation/train.py
```

2. Mira al alumno jugando. R empieza una partida nueva:

```bash
python lessons/05-imitation/watch.py
```

3. Ejecuta los cuatro experimentos: ¿el alumno es tan bueno como su profesor?

```bash
python lessons/05-imitation/experiments.py
```

4. Ahora enséñale tú. Juega algunas partidas: ESPACIO o ↑ salta, ↓ se agacha, R empieza la siguiente partida y ESC guarda y sale:

```bash
python lessons/05-imitation/record.py
```

Después la terminal muestra el comando para entrenar a un alumno con tus propias demostraciones.

---

## 1. Un profesor en lugar de una recompensa

| | Lección 1 — gradiente | Lección 4 — refuerzo | Lección 5 — imitación |
|---|---|---|---|
| Qué recibe la red | la respuesta correcta de cada ejemplo | una recompensa en cada paso | lo que el profesor presionó en cada cuadro |
| ¿Quién sabe la respuesta? | el código (la regla verdadera) | nadie | el profesor |
| Cómo cambian los pesos | gradiente | gradiente | gradiente |
| Costo de aprender a jugar | — | 2 millones de pasos, unos 80 s | 1 partida grabada, unos 2 s |

La imitación vuelve a ser aprendizaje supervisado: el mismo ciclo de la lección 1 (ida, pérdida, gradiente, paso). La diferencia es de dónde sale la clave de respuestas. En la lección 1 la escribió una regla; aquí la escribe un jugador, solo con jugar.

---

## 2. Las demostraciones

[`record.py`](record.py) graba una fila por cuadro del juego: los 6 sensores que ve la IA y la tecla que se presionó en ese cuadro. El archivo de ejemplo, [`results/demos/teacher_02000k.csv`](results/demos/teacher_02000k.csv), tiene 10 partidas jugadas por el agente final de la lección 4: 50 000 filas.

Para grabar al profesor tú mismo (otro profesor, un retraso, más partidas), mira las opciones al inicio de `record.py`. Por ejemplo:

```bash
python lessons/05-imitation/record.py --teacher
```

![acciones en las demostraciones](results/teacher_02000k/01_actions.png)

Aquí el profesor presionó "saltar" (`jump`) en el 40% de los cuadros, "nada" (`nothing`) en el 33% y "agacharse" (`duck`) en el 28%. Es otra vez la **trampa de la exactitud** de la lección 1: un alumno que siempre respondiera "saltar" ya tendría 40% de exactitud, y chocaría con el primer cactus. Con tus propias demostraciones la trampa suele ser mucho mayor, porque las personas pasan la mayor parte del tiempo sin presionar nada.

---

## 3. Entrenando al alumno

El alumno es una red pequeña escrita solo con numpy, como en la lección 1: 6 sensores → 16 neuronas ocultas → 3 acciones. Con 3 respuestas posibles en lugar de 2, la salida usa una *softmax*, que convierte tres puntajes en tres probabilidades que suman 100%.

La última partida de las demostraciones se deja aparte: el alumno nunca entrena con ella, así que muestra cómo le va al alumno en situaciones que nunca vio.

![entrenamiento](results/teacher_02000k/02_training.png)

Con las demostraciones de ejemplo:

- el alumno presiona la misma tecla que el profesor en el **93.7%** de los cuadros de entrenamiento y en el **93.5%** de los cuadros de la partida apartada;
- jugando solo, hace **5000 en las 10 partidas nuevas**.

El alumno completo es un archivo con 163 números, `results/models/student_teacher_02000k.npz`. Como el agente de la lección 4, ese archivo *es* el aprendizaje.

---

## 4. 93% de coincidencia, juego perfecto

Si el alumno no coincide con el profesor en el 6% de los cuadros, ¿cómo es que nunca choca? Porque la mayoría de las diferencias ocurren cuando la tecla casi no importa:

- el **56%** de las diferencias ocurre mientras el dino está en el aire, donde presionar "saltar" otra vez no hace nada;
- otro **24%** son "nada" contra "agacharse" mientras el obstáculo todavía está lejos.

La exactitud mide qué tan bien **copia** el alumno; el puntaje mide qué tan bien **juega**. No son lo mismo, y el objetivo es lo segundo.

---

## 5. Qué aprendió el alumno

![decisiones](results/teacher_02000k/03_decisions.png)

Compárala con [`04_decisions.png`](../04-reinforcement/results/04_decisions.png) de la lección 4: es prácticamente la misma imagen. El alumno aprendió la línea inclinada del profesor para el cactus y hasta su extraña estrategia para el pájaro a la altura de la cabeza (quedarse agachado mientras está lejos y saltar por encima cuando se acerca). El alumno nunca vio una recompensa; solo copió, y lo copió todo, incluidas las manías.

---

## 6. ¿El alumno es tan bueno como su profesor?

[`experiments.py`](experiments.py) graba cuatro profesores distintos, entrena un alumno con cada uno y pone a profesor y alumno a jugar las mismas 10 partidas nuevas. Los profesores "con retraso" son el agente final de la lección 4 presionando cada tecla unos cuadros después de ver la situación, como el tiempo de reacción de una persona.

![experimentos](results/experiments.png)

| Profesor | Ejemplos | Puntaje promedio del profesor | Puntaje promedio del alumno |
|---|---|---|---|
| Fuerte, solo 1 partida grabada | 5000 | 5000 | **5000** |
| Débil (lección 4 después de 500k pasos) | 1803 | 151 | 198 |
| Fuerte, 8 cuadros de retraso | 37 313 | 2870 | **5000** |
| Fuerte, 12 cuadros de retraso | 3757 | 752 | 172 |

- **Imitar es barato.** Una partida grabada y 2 segundos de entrenamiento dieron un jugador perfecto. La lección 4 necesitó 2 millones de pasos para descubrir lo mismo por su cuenta.
- **Un profesor débil hace un alumno débil.** El alumno copia lo que ve. No tiene forma de saber que el profesor es malo, porque nunca ve un puntaje.
- **El alumno puede superar a un profesor descuidado.** El profesor con 8 cuadros de retraso a veces salta demasiado tarde y choca; el alumno aprende su comportamiento *promedio*, que es bueno, y pierde el desliz ocasional. Hizo 5000 donde el profesor promedió 2870.
- **Pero no a uno malo.** Con 12 cuadros de retraso, el retraso es el comportamiento, no un desliz. El alumno lo copia y le va todavía peor.

---

## 7. Tu turno: enséñale tú

Graba algunas partidas con `record.py` y entrena un alumno con ellas usando el comando que muestra el script. Después compara tu alumno con el de ejemplo. Algunas cosas que suelen pasar con demostraciones humanas:

- **La trampa se hace más grande.** La mayoría de tus cuadros son "nada", así que el alumno "siempre nada" parece tener una exactitud muy alta y juega terrible. Mira `01_actions.png` y la línea punteada en `02_training.png`.
- **Tu tiempo de reacción está en los datos.** Como los profesores "con retraso" de arriba, el alumno aprende cuándo presionaste *tú*, no cuándo deberías haber presionado.
- **Más partidas ayudan**, sobre todo partidas en las que llegaste a la parte rápida del juego, con pájaros. El alumno solo puede copiar situaciones que están en las demostraciones.

---

## Ejercicios

1. Entrena solo con la primera partida de las demostraciones de ejemplo (`train.py --matches 1`). ¿El alumno sigue haciendo 5000?
2. En `experiments.py`, prueba retrasos de 4, 10 y 16 cuadros. ¿Dónde está la línea entre "el alumno limpia los errores del profesor" y "el alumno los copia"?
3. Graba tus propias demostraciones y entrena un alumno. ¿Es mejor o peor que tú? Míralo con `watch.py --student` y la ruta que muestra el entrenamiento.
4. Mezcla tus demostraciones con las del profesor: `train.py --demos` acepta varios archivos. ¿Qué le pasa al alumno?

---

## ¿Qué relación tiene con los LLM?

Los LLM se entrenan en el mismo orden que este estudio:

1. El **preentrenamiento** es imitación a escala gigante: el modelo aprende a predecir la siguiente palabra de billones de palabras escritas por personas. Copia texto humano, bueno y malo.
2. El ***supervised fine-tuning*** (ajuste fino supervisado) es esta lección: personas escriben ejemplos de buenas respuestas, y el modelo aprende a imitarlos. Es barato y rápido, como el alumno de 1 partida, pero el modelo solo puede ser tan bueno como los ejemplos, como el alumno del profesor débil.
3. El **aprendizaje por refuerzo** (lección 4) viene al final: el modelo intenta, recibe un puntaje y se ajusta para puntuar más alto. Así es como puede ir más allá de sus ejemplos.

Imitar primero y después usar refuerzo combina lo mejor de los dos: la imitación da un buen punto de partida a bajo costo, y el refuerzo lo mejora donde los ejemplos se quedan cortos.

**Lección anterior:** [aprendizaje por refuerzo](../04-reinforcement/README.es.md) · **Inicio del estudio:** [todas las lecciones](../../README.es.md)
