[English](README.md) · [Português](README.pt-BR.md) · **Español**

# Explicación a fondo: los LLM de hoy y el harness que los rodea

> El modelo de la lección 7 escribe un carácter a la vez. ¿Qué relación tiene eso con los LLM que la gente usa hoy,
> que leen archivos, ejecutan código y corrigen sus propios errores? En su mayor parte, es el mismo ciclo, mucho más grande,
> envuelto en otro programa: el **harness** (en inglés, el "arnés" que se le pone a un caballo para guiarlo y aprovechar su fuerza).
> Aquí construimos uno de juguete, y le enseñamos a usarlo al modelo de la lección 7.

Las gráficas salen de dos scripts. Los dos usan el modelo de la lección 7, así que entrena primero la lección 7 si todavía no lo hiciste:

```bash
python deep-dives/llm-and-harness/toy_agent.py      # a few seconds
```

```bash
python deep-dives/llm-and-harness/teach_tools.py    # about 5 minutes on a Mac mini M4
```

Esta página parte de la [lección 7](../../lessons/07-language-model/README.es.md), y retoma ideas de las lecciones [4](../../lessons/04-reinforcement/README.es.md) (refuerzo) y [5](../../lessons/05-imitation/README.es.md) (imitación).

---

## 1. El ciclo que ya escribiste

Este es el corazón de la lección 7 (`tiny_lm.py`, `generate`):

```python
for _ in range(n):
    ids.append(draw(self.next_probabilities(ids, temperature)))
```

El modelo solo sabe hacer una cosa: dado el texto hasta ahora, dar la probabilidad de cada siguiente token posible. Los LLM de hoy hacen exactamente lo mismo, en el mismo ciclo. No tienen un modo aparte que "piensa la respuesta completa". La respuesta completa es este ciclo, ejecutado miles de veces.

Lo que cambia es el tamaño:

| | Lección 7 | Un LLM grande de hoy |
|---|---|---|
| Token | 1 carácter (53 distintos) | pedazos de palabras (de 50 000 a 200 000 distintos) |
| Contexto | 64 caracteres | de cientos de miles a cerca de un millón de tokens |
| Tamaño | 2 capas, vectores de 64 números, ~110 000 pesos | decenas de capas, vectores de miles de números, cientos de miles de millones de pesos (a menudo una "mezcla de expertos": solo una parte de la red trabaja en cada token) |
| Texto de entrenamiento | los README del estudio (~47 000 caracteres) | billones de tokens |
| Entrenamiento | predecir el siguiente carácter | tres etapas, abajo |

---

## 2. Cómo se entrena un LLM grande: tres etapas que ya viste

1. **Preentrenamiento.** Lo que hizo la lección 7: predecir el siguiente token, con una cantidad enorme de texto. El resultado es un autocompletado muy leído. Continúa textos; todavía no sigue instrucciones.
2. **Ajuste supervisado** (*supervised fine-tuning*, SFT). Esta es la [lección 5](../../lessons/05-imitation/README.es.md), imitación. El modelo se entrena con ejemplos de buenas conversaciones, y los copia. Aquí también aprende formatos, como la manera de pedir una herramienta. La sección 5 le hace exactamente esto al modelo de la lección 7.
3. **Aprendizaje por refuerzo.** Esta es la [lección 4](../../lessons/04-reinforcement/README.es.md), PPO. El modelo escribe, algo le pone una nota (personas, otro modelo que hace de juez, o una prueba que pasa o falla), y el modelo se empuja hacia lo que sacó buena nota. Los modelos que "piensan antes de responder" salieron en buena parte de aquí: con refuerzo, aprendieron que escribir el razonamiento antes de la respuesta da más recompensa.

¿Recuerdas la meseta de "saltar siempre" de la lección 4? Los LLM también encuentran atajos que dan recompensa sin hacer el trabajo (*reward hacking*). Diseñar recompensas que no se puedan burlar es una parte grande del trabajo.

---

## 3. El harness: el ciclo alrededor del modelo

Incluso después de todo ese entrenamiento, el modelo sigue siendo una función de texto a texto. No puede leer un archivo, ejecutar un comando ni recordar nada entre una llamada y otra. El programa que hace esas cosas es el **harness**. Claude Code es un harness; las apps de chat también.

![Tres ciclos](charts/00_three_loops.png)

1. **El ciclo de tokens**, dentro del modelo: el `generate` de la lección 7.
2. **El ciclo del agente**, en el harness. El modelo escribe un poco de texto. Si pide una herramienta, el harness la ejecuta, pega el resultado en la conversación y vuelve a preguntarle al modelo:

   ```python
   transcript = system_prompt + tool_descriptions + question
   while True:
       reply = model(transcript)              # el ciclo de tokens, adentro
       transcript += reply
       if reply asks for a tool:
           transcript += run_tool(reply)      # la ejecuta el HARNESS, no el modelo
       else:
           break                              # la respuesta final
   ```

3. **El ciclo de la sesión**, también en el harness. El contexto tiene un límite (64 caracteres para el modelo de la lección 7), así que cuando la conversación se hace demasiado larga, el harness resume la parte vieja. La memoria entre sesiones es solo un archivo de texto que el harness pega al principio.

Esta misma explicación la escribió un modelo de IA trabajando dentro de un harness. Para leer `tiny_lm.py`, el modelo escribió una petición como "ejecuta `grep` en este archivo"; el harness la ejecutó y pegó la salida de vuelta. El modelo nunca tocó el disco. Solo escribió texto en un formato acordado.

---

## 4. Un agente de juguete

`harness.py` tiene todo en unas pocas decenas de líneas. La conversación es un solo texto, la **transcripción**, y cada línea empieza con quién la escribió:

```
q: what is 37+48?          <- the user
call calc: 37+48           <- the model asks for a tool
result: 85                 <- the harness ran the tool and pasted the result
answer: 85                 <- the model answers, and the loop stops
```

- **Las herramientas** son funciones normales de Python: `calc` suma o resta dos números, `letters` cuenta las letras de una palabra.
- **El modelo** es cualquier cosa que reciba la transcripción y devuelva una línea. El modelo falso es un script de unos cuantos `if`. Como un LLM de verdad, no mira nada más que la transcripción.
- **El harness** es el ciclo: preguntarle al modelo, buscar `call <tool>: <input>`, ejecutar la herramienta, pegar `result: ...` y volver a preguntar, hasta que una línea empiece con `answer:`.

![El ciclo del agente, paso a paso](charts/01_agent_loop.gif)

Ahora la parte interesante: **el mismo harness**, con el modelo de la lección 7 en lugar del falso:

```
q: what is 37+48?
the step and the score the student the s
result: error: write call or answer
the student the script the same and the
result: error: write call or answer
the student the script the same and the
...                                          (gives up after 4 turns)
```

Tres cosas para notar:

- **El harness es el mismo, y el resultado es completamente distinto.** Para que el ciclo funcione, el modelo tiene que conocer el formato `call` / `answer`. El modelo de la lección 7 solo leyó README, así que escribe "como un README".
- **Ni siquiera puede ver la pregunta por mucho tiempo.** Con un contexto de 64 caracteres, solo ve los últimos 64. Un harness de verdad pone las descripciones de las herramientas al principio de la transcripción, y un LLM de verdad puede leerlas ahí. El nuestro nunca las vería.
- **Los turnos 2, 3 y 4 son idénticos.** El modelo no tiene memoria: el mismo texto de entrada da el mismo texto de salida (aquí el harness elige siempre el carácter más probable). Todo lo que "sabe" de la conversación es el texto que arma el harness.

---

## 5. Enseñarle el formato al modelo

Entonces, enseñémosle, como la [lección 5](../../lessons/05-imitation/README.es.md) le enseñó al dino: con **imitación**. El maestro es el modelo falso; sus transcripciones son los ejemplos. `teach_tools.py` arranca tres alumnos desde los pesos de la lección 7, y entrena a cada uno con el mismo ciclo de la lección 7:

| Alumno | Ejemplos de los que aprende | Pasos |
|---|---|---|
| 1. responder directo | `q: what is 37+48?` → `answer: 85`, sin herramientas | 3 000 |
| 2. llamar herramientas | las transcripciones del modelo falso: números de dos cifras, palabras del estudio | 3 000 |
| 3. llamar herramientas, datos variados | el mismo formato, pero con números de 1 a 3 cifras y palabras inventadas como `qzwkt` | 25 000 |

Después, los cuatro modelos (la lección 7 tal cual y los tres alumnos) se ejecutan en el mismo harness, con preguntas que **ninguno vio en el entrenamiento**: sumas y restas de dos cifras, palabras reales que se dejaron fuera del entrenamiento, y números de cuatro cifras, más largos que cualquiera del entrenamiento.

![Respuestas correctas por modelo](charts/02_teach_tools.png)

Lo que pasó, con transcripciones reales (todas en `charts/samples.md`):

**El alumno 1 no sabe hacer cuentas: 0 %.** Responde `46-30` con `-10`. Una red así de pequeña no aprende a sumar con unos pocos miles de ejemplos. Los LLM grandes son mucho mejores haciendo cuentas de cabeza, pero todavía se equivocan con números largos, y esa es una de las razones por las que se les da una calculadora o una herramienta para ejecutar código. (Sacó 41 % contando letras. Adivinar siempre la longitud más común daría 19 %, así que aprendió algo sobre la longitud de las palabras, pero no a contar).

**El alumno 2 llama a la calculadora a la perfección: 100 %.** El modelo no suma; **copia** `46-30` en `call calc:`, y Python suma. Pero mira las palabras y los números grandes:

```
q: how many letters in blows?        q: what is 6950+1180?
call letters: block                  call calc: 69+18
result: 5                            result: 87
answer: 5                            answer: 87
```

Aprendió la **forma** de sus ejemplos, no la regla "copia la entrada". Todos los números que vio tenían dos cifras, así que escribe dos cifras. Había visto las palabras del estudio miles de veces, así que escribe una que conoce (`block`) en lugar de la nueva (`blows`). Su 11 % en letras es suerte: `block` y `blows` tienen 5 letras las dos.

**El alumno 3 aprendió a copiar: 90 % con palabras reales que nunca vio.** Las palabras inventadas no se pueden memorizar, así que la única manera de acertarlas es copiar de verdad, letra por letra. Pero tardó mucho:

![El alumno 3 aprendiendo a copiar](charts/03_copy_emerges.png)

Copiar números de dos cifras llegó antes de 2 500 pasos. Copiar palabras se quedó en 0 % durante 20 000 pasos, y luego saltó a más de 80 % casi de golpe. Los investigadores han visto el mismo tipo de salto repentino en transformers reales cuando aprenden a copiar de partes anteriores del texto (las "induction heads", o cabezas de inducción, que Anthropic estudió en 2022).

**Nadie maneja números de cuatro cifras: 0 %.** El alumno 3 escribe `695+118` para `6950+1180`. Vio números de hasta 3 cifras, así que escribe hasta 3 cifras. La herramienta calcula `813` correctamente, para la pregunta equivocada, y el modelo lo informa con seguridad. Una herramienta solo es tan buena como la petición que escribe el modelo.

Lo que esto dice de los LLM de verdad:

- **Llamar herramientas es una habilidad aprendida**, que se enseña con ajuste fino y refuerzo, con conjuntos de ejemplos enormes y muy variados, para que el modelo aprenda la regla y no la forma de unos pocos ejemplos.
- **Las herramientas le ganan a memorizar**: 0 % contra 100 % en las mismas cuentas, con la misma red diminuta.
- **Las fallas se parecen a las reales**: un argumento equivocado enviado a una herramienta correcta, y luego informado con seguridad. Los buenos harness ayudan comprobando (ejecutando las pruebas, mostrándole los errores al modelo, preguntándole a una persona antes de acciones riesgosas).
- Contar letras es famosamente difícil para los LLM grandes, por otra razón: leen pedazos de palabras, no letras, así que no "ven" las letras. El nuestro lee un carácter a la vez, y por eso el alumno 1 al menos pudo acercarse.

---

## 6. Por qué el harness importa tanto como el modelo

El estudio ya vio esto antes, con el dino:

| En el estudio | En un agente LLM |
|---|---|
| La red de política de la [lección 4](../../lessons/04-reinforcement/README.es.md): entran 6 números, sale una acción | el modelo: entra texto, sale texto |
| `game.py` y el ciclo principal: arman la observación, aplican la acción, devuelven el resultado | el harness |
| [Lección 6](../../lessons/06-vision/README.es.md): los "ojos" de la CNN convierten la pantalla en los 6 números que necesita el cerebro | el harness convirtiendo archivos, páginas web y la salida de comandos en texto para el modelo |
| [Lección 8](../../lessons/08-hard-mode/README.es.md): la red sola llega a unos 900 puntos; con un planificador al lado, a 5000 | un modelo solo contra un modelo que puede ejecutar código, leer errores y volver a intentar |

Con el **mismo modelo**, un buen harness (herramientas útiles, una transcripción bien armada, maneras de comprobar su propio trabajo) cambia mucho lo que se logra. Uno malo desperdicia un buen modelo. El modelo es el motor; el harness es el resto del auto.

---

## Lo que simplificamos

- Las llamadas a herramientas de verdad son estructuradas (normalmente JSON con el nombre de la herramienta y argumentos con nombre), y el harness las revisa antes de ejecutar nada. La nuestra es una línea de texto.
- El ajuste fino de verdad normalmente aprende solo de las líneas del propio modelo. El nuestro también aprendió a predecir las preguntas y los resultados, lo que aquí no cuesta nada.
- Un harness de verdad envía la lista de herramientas y sus descripciones en cada petición, así que el modelo puede usar herramientas que nunca vio en el entrenamiento. Nuestro modelo tiene un contexto de 64 caracteres, así que el formato tiene que quedar grabado por el entrenamiento.

---

## Resumen

| Pregunta | Respuesta corta |
|---|---|
| ¿Un LLM grande es distinto del modelo de la lección 7? | El mismo ciclo y el mismo tipo de red, millones de veces más grande, entrenado en tres etapas: preentrenamiento (lección 7), imitación (lección 5) y refuerzo (lección 4). |
| ¿Qué es un harness? | El programa alrededor del modelo: arma la transcripción, ejecuta las herramientas que pide el modelo, pega los resultados de vuelta y maneja el contexto y la memoria. |
| ¿El modelo ejecuta las herramientas? | No. Solo escribe texto pidiéndolas. Las ejecuta el harness. |
| ¿El modelo de la lección 7 puede usar herramientas? | Tal cual, no. Después de un ajuste fino con ejemplos, sí, pero solo de forma tan general como variados fueron los ejemplos. |

**Volver a:** [la página principal del estudio](../../README.es.md)
