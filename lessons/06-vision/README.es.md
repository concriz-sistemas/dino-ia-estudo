[English](README.md) · [Português](README.pt-BR.md) · **Español**

# Lección 6 — Aprender a ver: el dino juega desde la pantalla

> **Objetivo:** hasta ahora el dino recibía 6 números listos del juego. Aquí recibe solo **la pantalla**,
> y una red convolucional (CNN) aprende a leerla.
> Es el mismo tipo de entrenamiento que "gato o perro" con fotos, con etiquetas que el juego da gratis.

![ojos + cerebro jugando](results/eyes.gif)

*Jugando solo desde la pantalla. La línea naranja es donde los ojos creen que empieza el siguiente obstáculo. Abajo: la imagen pequeña que reciben los ojos y los 6 números, los verdaderos (del juego) y los estimados (por los ojos).*

Ejecuta uno a la vez, en este orden.

1. Mira a ojos + cerebro jugando. Los ojos entrenados ya están en el repositorio. R empieza una partida nueva:

```bash
python lessons/06-vision/watch.py
```

2. Entrena los ojos tú mismo. Tarda unos 20-25 minutos en una Mac reciente y más en una computadora sin una tarjeta gráfica que PyTorch pueda usar:

```bash
python lessons/06-vision/train_eyes.py
```

3. Prueba el otro enfoque, de los píxeles directo a las teclas (unos minutos):

```bash
python lessons/06-vision/pixels_to_keys.py
```

Esta lección usa PyTorch, que ya se instaló en la lección 4.

---

## 1. De 6 números a píxeles

La pantalla mide 800 × 300 píxeles, pero la mayor parte no importa. [`vision.py`](vision.py) recorta la zona que va desde lo alto de un salto hasta un poco debajo del suelo, y hasta 600 píxeles delante del dino, junta cada bloque de 3 × 5 píxeles en uno solo y lo pasa a escala de grises. La red recibe una **imagen de 200 × 44**, con las cosas oscuras (dino, cactus, pájaros) convertidas en claras.

Una imagen no alcanza. En una foto no se puede saber qué tan rápido se mueve algo, y la velocidad decide cuándo saltar. Por eso la red recibe **dos imágenes: la actual y la de 3 cuadros atrás**. La diferencia entre ellas es el movimiento.

![lo que ve cada capa](results/04_what_layers_see.png)

La fila de arriba muestra lo que recibe la red: la escena ahora y "ahora menos 3 cuadros atrás", donde el cactus aparece dos veces (rojo donde está ahora, azul donde estaba).

---

## 2. Una red convolucional

Una **convolución** desliza un filtro pequeño (aquí de 5 × 5 píxeles) por toda la imagen y, en cada posición, mide cuánto se parece ese pedazo de la imagen al filtro. El mismo filtro se usa en todas partes, así que un cactus a la izquierda y un cactus a la derecha se detectan de la misma forma. Eso es lo que hace que las CNN sean buenas con imágenes.

Los ojos tienen 3 capas convolucionales y después una capa densa que convierte lo que encontraron en los 6 números de los sensores: 630 406 pesos en total.

- La **capa 1** (la segunda fila de la imagen de arriba) reacciona a detalles pequeños: el contorno del dino y del cactus.
- La **capa 2** los combina en formas pequeñas.
- La **capa 3** ve mucho menos detalle pero cosas más grandes: una mancha para el dino, una para el cactus. Dice "aquí hay algo".

Estos son los 16 filtros de la primera capa, tal como se aprendieron:

![filtros](results/03_filters.png)

En los libros, los filtros de la primera capa parecen detectores de bordes bien ordenados. Estos se ven más ruidosos: las imágenes son diminutas y el entrenamiento es corto. Algunos todavía muestran franjas horizontales o verticales de claro y oscuro.

Un truco más: las convoluciones son buenas para decir **qué** hay en la imagen, pero no **dónde**. Los ojos reciben un canal de entrada extra que solo dice "qué tan a la derecha está este píxel" (una técnica llamada *CoordConv*), lo que les ayuda a medir la distancia.

---

## 3. Dos formas de jugar desde los píxeles

**De extremo a extremo** ([`pixels_to_keys.py`](pixels_to_keys.py)): una sola red, de las dos imágenes directo a "nada / saltar / agacharse", copiando al agente de la lección 4, como el alumno de la lección 5.

**Ojos + cerebro** ([`train_eyes.py`](train_eyes.py)): la CNN solo estima los 6 números de los sensores (los **ojos**). El agente de la lección 4, sin cambios, decide a partir de esos números (el **cerebro**). El cerebro no sabe si los números vinieron del juego o de los ojos.

| Enfoque | Coincidencia con el profesor | Puntaje promedio en 10 partidas nuevas | Partidas con 5000 |
|---|---|---|---|
| De extremo a extremo, píxeles → teclas | 94.5% de los cuadros | 3013 | 4 |
| Ojos + cerebro, ronda 0 (solo las escenas del profesor) | — | 316 | 0 |
| Ojos + cerebro, después de 6 rondas de DAgger | — | **5000** | **10** |

Sin ningún truco extra, la red de extremo a extremo lo hizo mejor que ojos + cerebro. Pero cuando la red de extremo a extremo choca, no hay forma de saber por qué: es un solo bloque de los píxeles a las teclas. Con ojos + cerebro se puede comparar, escena por escena, lo que estimaron los ojos con la verdad. Eso fue lo que mostró el problema, y lo que llevó a la solución.

---

## 4. Por qué fallaron los primeros ojos

En la ronda 0 los ojos aprenden solo de escenas grabadas mientras el cerebro juega con los sensores **verdaderos**. Su error promedio cerca de los obstáculos es de 55 píxeles, y ojos + cerebro chocan pronto en todas las partidas (promedio 316).

Seguir un choque cuadro por cuadro, en una versión anterior de estos ojos, mostró la causa. Mientras el cactus está lejos, al agente de la lección 4 le gusta **agacharse**. Cuando el cactus se acerca, normalmente ya está saltando, así que las escenas grabadas casi nunca muestran **un dino agachado justo al lado de un cactus**. Con los ojos al mando, el salto llega un poquito tarde, el dino todavía está agachado en esa posición, y los ojos, frente a una escena que nunca vieron, se equivocan: la distancia estimada *crece* mientras el cactus se acerca. El cerebro cree que el cactus está lejos y no salta.

Es otra vez la trampa de la lección 5, ahora en la percepción: una red entrenada solo con las escenas del profesor no sabe qué hacer en las escenas a las que la llevan sus propios errores.

---

## 5. La solución: DAgger

**DAgger** (*Dataset Aggregation*, agregación de datos) cierra esa brecha:

1. Ojos + cerebro juegan **juntos**, con errores y todo.
2. Cada escena a la que llegan se graba con los números **verdaderos** de los sensores como clave de respuestas (el juego siempre los conoce).
3. Los ojos siguen entrenando con todo lo grabado hasta ahora.
4. Repite.

Cada ronda agrega justo las escenas en las que el sistema se estaba equivocando.

![rondas](results/01_rounds.png)

| Ronda | Escenas grabadas | Puntaje promedio | Partidas con 5000 | Error de distancia cerca de los obstáculos |
|---|---|---|---|---|
| 0 | 20 000 | 316 | 0 | 55 píxeles |
| 1 | 20 962 | 1063 | 0 | 36 píxeles |
| 2 | 26 969 | 2359 | 4 | 16 píxeles |
| 3 | 37 442 | 3322 | 6 | 12 píxeles |
| 4 | 50 349 | 4576 | 9 | 9 píxeles |
| 5 | 65 349 | 4576 | 9 | 7 píxeles |
| 6 | 80 349 | **5000** | **10** | 6 píxeles |

Las primeras rondas agregan pocas escenas, porque ojos + cerebro todavía chocan pronto. A medida que los ojos mejoran, las partidas duran más, cada ronda trae más escenas y los errores que quedan desaparecen.

![distancia verdadera vs estimada](results/02_estimates.png)

Después de la ronda 6, la distancia que estiman los ojos (naranja) sigue a la verdadera (negro) en todos los obstáculos de una partida.

---

## 6. Las etiquetas salieron gratis

Para entrenar "gato o perro" con fotos, alguien tiene que etiquetar miles de imágenes a mano. Aquí, cada imagen viene con sus etiquetas gratis: el juego conoce la distancia, el tamaño y la altura exactos de cada obstáculo. Por eso también DAgger fue fácil: cualquier escena nueva se podía etiquetar al instante.

Los autos autónomos y los robots usan la misma idea. Gran parte de su entrenamiento ocurre en simuladores, donde cada píxel viene con su respuesta. La parte difícil es la misma que aquí: el mundo real les muestra escenas que el simulador nunca les mostró.

---

## Ejercicios

1. En `vision.py`, pon `FRAME_GAP = 1` y vuelve a entrenar. ¿Los ojos todavía pueden estimar la velocidad?
2. Quita a los ojos el canal de CoordConv (el buffer `columns`). ¿Cómo cambia el error de distancia?
3. Ejecuta `train_eyes.py --rounds 0` (sin DAgger) y después `--rounds 2`. Compara con la tabla de arriba.
4. Aplica la idea de DAgger a `pixels_to_keys.py`: déjalo jugar, etiqueta las escenas que visita con las teclas del profesor y vuelve a entrenarlo. ¿Llega a 5000 en todas las partidas?
5. Usa al alumno de la lección 5 como cerebro en lugar del agente de la lección 4. Lee los mismos 6 números.

---

## ¿Qué relación tiene con los LLM?

- **Los modelos que ven imágenes** funcionan de forma parecida: una red de visión convierte la imagen en listas de números, y el modelo de lenguaje lee esos números como si fueran palabras. Es otra vez ojos + cerebro, con un cerebro mucho más grande.
- **Las escenas nunca vistas en el entrenamiento** también son un problema para los LLM. Cuando una pregunta es muy distinta de todo lo que había en sus datos de entrenamiento, un modelo puede responder con seguridad y aun así equivocarse, igual que los ojos que "vieron" un cactus lejos cuando estaba justo delante del dino.
- **Las etiquetas son la parte cara.** Aquí el juego las dio gratis. Para los LLM, los buenos ejemplos escritos por personas cuestan caro, y esa es una de las razones por las que primero se preentrenan con texto en bruto: la siguiente palabra es una etiqueta que sale gratis.

**Lección anterior:** [imitación](../05-imitation/README.es.md) · **Inicio del estudio:** [todas las lecciones](../../README.es.md)
