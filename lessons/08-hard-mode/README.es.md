[English](README.md) · [Português](README.pt-BR.md) · **Español**

# Lección 8 — Modo difícil: cuando el mundo cambia

> **Objetivo:** cambiar el juego y ver qué pasa con todo lo que se entrenó en el juego viejo.
> El juego nuevo tiene formaciones de pájaros en las que saltar es fatal. ¿Quién todavía puede jugar, quién puede aprenderlo
> y qué hace falta?

![antes y después](results/before_after.gif)

*La misma partida dos veces. Arriba: el agente de la lección 4 salta contra una escalera de pájaros y choca. Abajo: el alumno de la lección 8 se agacha por debajo y sigue adelante.*

El juego original (`dino/game.py`) no cambió: las lecciones 1 a 7 lo siguen usando. El juego difícil es una versión aparte, en [`hard_game.py`](hard_game.py).

Juégalo tú mismo:

```bash
python lessons/08-hard-mode/hard_game.py
```

Después ejecuta los experimentos uno a la vez, en este orden. El alumno entrenado ya está en el repositorio, así que puedes saltarte el paso 3 y ejecutar igual los demás.

1. ¿Quién todavía puede jugar? Todos los agentes del estudio, en los dos juegos (unos 4 minutos):

```bash
python lessons/08-hard-mode/test_agents.py --student
```

2. ¿Puede el aprendizaje por refuerzo aprender el juego nuevo por sí solo? Cuatro intentos (unos 10 minutos):

```bash
python lessons/08-hard-mode/rl_attempts.py
```

3. Un alumno que imita a un entrenador, con DAgger (unos 20 minutos):

```bash
python lessons/08-hard-mode/train_student.py
```

4. Mira al agente de la lección 4 y al alumno en la misma partida. R empieza una partida nueva:

```bash
python lessons/08-hard-mode/watch.py
```

---

## 1. Las formaciones nuevas

Cuando empiezan a aparecer los pájaros (alrededor de 700 puntos), a veces aparece una **formación** en lugar de un obstáculo solo:

| Formación | Qué es | ¿Saltar? | ¿Agacharse? |
|---|---|---|---|
| **Túnel** | una fila larga de pájaros a la altura de la cabeza, demasiado larga para saltarla a cualquier velocidad | ❌ | ✅ la única forma |
| **Escalera** | pájaros que bajan: alto (75), medio (50) y después tres bajos (26) | ❌ un salto choca con uno | ✅ la única forma: quedarse de pie bajo los dos primeros y agacharse bajo el resto |
| **Par** | dos pájaros a la altura de la cabeza, muy juntos | ✅ | ✅ las dos funcionan |

Los cactus y los pájaros solos del juego original también siguen apareciendo. Un cactus solo se puede saltar, y el pájaro a la altura de la cabeza se puede saltar o esquivar agachándose.

Nada de esto se adivinó: [`hard_game.py`](hard_game.py) prueba todas las secuencias de teclas contra todas las formaciones, a todas las velocidades de 10.5 a 20, y comprueba que hay una forma de pasar, y qué teclas necesita:

```bash
python lessons/08-hard-mode/hard_game.py --verify
```

---

## 2. El mundo cambió: los agentes viejos fallan

![quién puede jugar](results/01_who_can_play.png)

| Agente | Juego original | Juego difícil | Contra qué chocó (10 partidas difíciles) |
|---|---|---|---|
| Lección 3: campeón de la neuroevolución | 5000 | 930 | 5 túneles, 4 escaleras, 1 par |
| Lección 4: aprendizaje por refuerzo | 5000 | 902 | 5 túneles, 3 escaleras, 2 pares |
| Lección 5: alumno por imitación | 5000 | 946 | 6 túneles, 4 escaleras |

Los tres eran perfectos en el juego que conocían, y los tres chocan en las primeras formaciones, justo después de que empiezan a aparecer los pájaros. Aprendieron "pájaro a la altura de la cabeza: agáchate mientras está lejos, salta cuando está cerca", que era una buena regla en el juego viejo. En un túnel o en una escalera, ese salto es fatal.

Los agentes no tienen nada de malo. **El mundo cambió** y ellos nunca lo habían visto.

---

## 3. El aprendizaje por refuerzo solo no aprende el juego nuevo

La solución obvia es entrenar de nuevo en el juego nuevo. [`rl_attempts.py`](rl_attempts.py) lo intenta cuatro veces, y cada intento corrige lo que pudo haber salido mal en el anterior:

![intentos de aprendizaje por refuerzo](results/02_rl_attempts.png)

| Intento | Entrenamiento | Juego original | Juego difícil |
|---|---|---|---|
| 1. Desde cero | 3 millones de pasos, solo el juego difícil | 5000 | 951 |
| 2. Ajuste fino de la lección 4 | 1 millón de pasos más en el juego difícil | 5000 | 950 |
| 3. + currículo | 2 millones de pasos; en el entrenamiento, las formaciones aparecen desde el principio, en la mitad de los obstáculos | 5000 | 953 |
| 4. + más exploración | como el 3, con un bono de exploración 3× más grande, 3 millones de pasos | 4294 | 955 |

Ninguno logra pasar. ¿Por qué?

![por qué falla el aprendizaje por refuerzo](results/03_why_rl_fails.png)

Este es el intento 4 acercándose a una escalera. Mientras los pájaros están lejos, le da cerca de 60% a "agacharse". Cuando se acercan, cambia a "saltar" (casi 90%), su viejo hábito con los pájaros, salta contra la escalera y choca. Para pasar, tendría que seguir agachado durante muchas decisiones seguidas, todas correctas, justo en el momento en que su hábito dice "salta". Hacer eso por casualidad, con la frecuencia suficiente para notar que vale la pena, prácticamente nunca pasa.

Este es el **problema de la exploración** del aprendizaje por refuerzo: solo aprende de lo que prueba, y algunas soluciones son demasiado largas o demasiado improbables para encontrarlas probando al azar. Más exploración (intento 4) no lo arregló, e incluso dejó al agente peor en el juego original.

---

## 4. Un planificador lo resuelve, pero no aprende

[`planner.py`](planner.py) no aprende nada. En cada momento mira los obstáculos en pantalla, imagina los siguientes 80 cuadros con la propia física del juego y busca una secuencia de teclas que no choque.

| | Juego original | Juego difícil |
|---|---|---|
| El planificador | **5000** | **5000** |

Es perfecto, pero:

- es **lento**: unos 10 a 20 segundos por partida en esta Mac, mientras que una red entrenada juega una partida en menos de un segundo;
- necesita **conocer las reglas del juego** (gravedad, fuerza del salto, velocidad) para imaginar lo que va a pasar;
- nunca mejora: cada decisión se calcula otra vez desde cero.

---

## 5. Un alumno que imita a un entrenador

Entonces: quien aprende no puede encontrar la solución por sí solo, y el planificador la encuentra pero es lento. La solución es dejar que el planificador **enseñe**: la imitación de la lección 5, más el DAgger de la lección 6.

**La primera idea no funcionó.** Con el planificador solo como profesor, el alumno casi no aprendió nada: 463 puntos en el juego difícil después de la ronda 0, y solo de 88 a 312 en el juego *original*. Dos rondas más de DAgger no lo llevaron más allá de 1161. Puedes repetirlo:

```bash
python lessons/08-hard-mode/train_student.py --teacher planner --rounds 2
```

El planificador presionó "saltar" en solo el **2%** de sus decisiones: prefiere no hacer nada, así que salta una vez, en el **último momento posible**, y después espera. Eso es perfecto para algo que calcula, pero un pésimo ejemplo para un alumno. Los saltos son tan raros que el alumno apenas los aprende (la trampa de la exactitud de la lección 5), y un salto en el último momento no deja margen, así que un alumno que se retrasa una decisión choca (como los profesores con retraso de la lección 5). **Un buen planificador no es necesariamente un buen profesor.**

**Lo que funcionó: un entrenador.** [`agents.py`](agents.py) arma uno con dos piezas:

- **el agente de la lección 4** juega como siempre, con sus saltos cómodos y anticipados;
- cuando una de las formaciones nuevas está cerca, **el planificador toma el control**, y prefiere agacharse temprano y quedarse agachado.

[`train_student.py`](train_student.py) entrena entonces a un alumno (una red pequeña que ve los mismos 6 sensores) para presionar lo que presionaría el entrenador:

- **ronda 0:** el entrenador juega 6 partidas difíciles, y se graba cada decisión;
- **rondas 1 a 6 (DAgger):** juega el *alumno*; en cada situación en la que se mete, el entrenador dice qué presionaría; esos ejemplos se agregan, y se entrena un alumno nuevo con todo lo acumulado hasta ahí.

![rondas del alumno](results/04_student_rounds.png)

| Ronda | Ejemplos | Juego difícil (6 partidas) | Juego original (3 partidas) |
|---|---|---|---|
| 0 | 15 000 | 3171 | 3231, 3331, 2624 |
| 1 | 20 676 | 1053 | 5000, 5000, 5000 |
| 2 | 25 764 | 1053 | 5000, 5000, 5000 |
| 3 | 32 773 | 3021 | 5000, 5000, 5000 |
| 4 | 42 773 | 3677 | 5000, 5000, 5000 |
| 5 | 52 773 | **5000** | 5000, 5000, 5000 |
| 6 | 62 773 | **5000** | 5000, 5000, 5000 |

Y en la prueba final, con 10 partidas nuevas de cada juego: **5000 en todas, en los dos juegos**, tan bueno como el planificador y rápido como una red.

Dos cosas para notar:

- **Aprender lo nuevo costó lo viejo, por un tiempo.** Después de la ronda 0, el alumno también era peor en el juego original (alrededor de 3 000 en lugar de 5000). Las rondas de DAgger lo recuperaron a medida que se acumulaban ejemplos. Los ejemplos siempre incluían también los obstáculos viejos (el juego difícil todavía tiene cactus y pájaros solos), así que el alumno los siguió practicando. Empeorar en habilidades viejas mientras se aprenden nuevas se llama *olvido catastrófico* (*catastrophic forgetting*), y mezclar ejemplos viejos y nuevos es uno de los remedios habituales.
- **El progreso no fue parejo.** Las rondas 1 y 2 se quedaron atascadas en 1053, chocando contra la misma escalera cada vez, hasta que se acumularon suficientes ejemplos de esa situación.

---

## 6. Dos formas de pasar: ¿qué elige cada uno?

El **par** se puede pasar de las dos formas. [`pair_choices.py`](pair_choices.py) observa cada par que logra pasar cada agente y comprueba si el dino estaba en el aire (saltó) o agachado (se agachó):

```bash
python lessons/08-hard-mode/pair_choices.py
```

En 12 partidas difíciles:

| Agente | Pares que pasó | Saltó | Se agachó |
|---|---|---|---|
| Agente de la lección 4 | 9 | **9** | 0 |
| El entrenador | 85 | 0 | **85** |
| Alumno de la lección 8 | 80 | 0 | **80** |

- **El agente de la lección 4 siempre salta.** Es su viejo hábito con los pájaros, y en un par resulta que funciona. (Pasa tan pocos porque choca en el primer túnel o en la primera escalera.)
- **El entrenador siempre se agacha**, porque a su planificador se le indicó que prefiriera agacharse cerca de las formaciones nuevas.
- **El alumno también se agacha siempre.** Nunca tuvo que elegir: copió el *estilo* del entrenador, no solo sus resultados.

Las dos respuestas son correctas. Cuál elige un agente depende de dónde lo aprendió: de su propia experiencia (lección 4) o de un profesor con una preferencia (el entrenador). Un alumno adopta las elecciones de su profesor incluso cuando eran solo una cuestión de gusto.

---

## 7. ¿Qué tiene que ver esto con los LLM?

Los mismos problemas, a una escala mucho mayor:

| En el dino | En un LLM |
|---|---|
| Los agentes viejos no conocen las formaciones | el modelo no sabe qué pasó después de que se recolectaron sus datos de entrenamiento (su *fecha de corte*) |
| Los ojos de la lección 6 fallan en una escena que nunca vieron | al modelo le va peor en temas, formatos o idiomas que eran raros en su texto de entrenamiento |
| El alumno empeora en el juego original mientras aprende el difícil | un modelo ajustado con datos nuevos puede empeorar en cosas que antes hacía bien (*olvido catastrófico*) |
| Los agentes conservan un hábito viejo ("saltar ante los pájaros") que ahora es fatal | el modelo aprende atajos de sus datos, como "todo número de la pregunta se usa en la respuesta" — y agregar una frase irrelevante con un número lo puede despistar |
| El aprendizaje por refuerzo no puede encontrar una solución larga probando al azar | el aprendizaje por refuerzo en los LLM también necesita un punto de partida que ya haga la tarea razonablemente bien |

Y las soluciones también se parecen:

- **Consultar en lugar de recordar.** El planificador mira la pantalla *ahora*. A los LLM se les dan herramientas de búsqueda y documentos en el momento de la pregunta, para que no dependan solo de lo que aprendieron.
- **Mostrar ejemplos antes de recompensar.** El alumno necesitó un profesor que le mostrara cómo pasar; el refuerzo solo no pudo encontrarlo. Los LLM se ajustan con buenos ejemplos (lección 5) antes del aprendizaje por refuerzo (lección 4).
- **Mezclar lo viejo con lo nuevo.** El alumno recuperó el juego original porque los ejemplos siempre mezclaban los dos. Cuando a los LLM se les hace un ajuste fino, se mezclan datos viejos por la misma razón, y los cambios se mantienen pequeños para que el modelo no se aleje demasiado de lo que sabía.
- **Entrenar en los casos difíciles.** DAgger agregó justo las situaciones en las que el alumno se equivocaba. Quienes construyen LLM recolectan ejemplos de los casos en los que los modelos fallan, incluidas preguntas con detalles irrelevantes, y entrenan con ellos.
- **Medir todo, antes y después.** Cada agente de esta lección se prueba en los dos juegos. Mejorar una cosa puede empeorar otra, así que los LLM se revisan con grandes conjuntos de pruebas antes de lanzar cada versión nueva.

**Lección anterior:** [un mini modelo de lenguaje](../07-language-model/README.es.md) · **Inicio del estudio:** [todas las lecciones](../../README.es.md)
