[English](README.md) · [Português](README.pt-BR.md) · **Español**

# Lección 2 — El entorno: el juego que la IA va a jugar

> **Objetivo:** entender el "mundo" de la IA — qué ve, qué puede hacer
> y cómo sabemos si le fue bien.

```bash
python lessons/02-the-environment/play.py
```

Juega algunas partidas. En la esquina de la pantalla, bajo "What the AI sees:" (lo que ve la IA), aparecen los **6 números que la IA va a recibir**. Fíjate cómo cambian cuando se acerca un cactus.

---

## ¿Qué es pygame?

Es una biblioteca de **Python** para hacer juegos 2D. Abre una ventana, dibuja rectángulos e imágenes, lee el teclado y controla el tiempo (60 cuadros por segundo). Se instala con `pip install pygame`.

Todo nuestro juego está en [`dino/game.py`](../../dino/game.py) — poco más de 200 líneas, todas comentadas.

---

## El ciclo del juego

Todo juego (y todo entorno de IA) es un ciclo que se repite:

```
┌──────────────────────────────────────────────┐
│ 1. OBSERVAR   el estado actual → sensors()   │
│ 2. DECIDIR    una acción      → humano o IA  │
│ 3. ACTUAR     avanzar un cuadro → step()     │
│ 4. EVALUAR    ¿chocó? ¿+1 punto?             │
└──────────────── se repite ───────────────────┘
```

Este mismo ciclo — **observación → acción → recompensa** — es la base del aprendizaje por refuerzo, la técnica que también se usa en el post-entrenamiento de los LLMs.

---

## Los tres ingredientes que necesita la IA

### 1. Observación: lo que "ve"

La IA no ve la pantalla. Recibe 6 números, calculados en `Game.sensors()`. En la pantalla, los nombres aparecen en inglés:

| # | Sensor | Significado | Rango |
|---|---|---|---|
| 1 | distance | Qué tan lejos está el próximo obstáculo (distancia) | 0 (pegado) a 1 (borde de la pantalla) |
| 2 | width | Ancho del obstáculo | ~0.3 a ~0.9 |
| 3 | height | Altura del obstáculo | ~0.4 a ~0.9 |
| 4 | elevation | Distancia del obstáculo al suelo (elevación) | 0 = cactus; > 0 = pájaro |
| 5 | dino height | Altura del dino (¿está en el aire?) | 0 = en el suelo |
| 6 | speed | Velocidad actual del juego | 0.4 al inicio, 1 como máximo |

Todos los números están **normalizados** (cerca de 0 a 1). Las redes neuronales aprenden mal cuando un número vale 800 y otro vale 0.02 — fue el mismo motivo por el que usamos `normalize()` en la Lección 1.

> 💡 **Elegir buenas entradas es la mitad del trabajo.** Si nos olvidáramos de la velocidad, la IA no tendría forma de saber que necesita saltar antes cuando el juego acelera.

### 2. Acción: lo que puede hacer

Tres opciones: `NOTHING` (nada), `JUMP` (saltar) o `DUCK` (agacharse). La red tendrá **3 salidas**, y gana la mayor.

### 3. Recompensa (o *fitness*): cómo sabemos si le fue bien

**+1 punto por cuadro sobrevivido.** Simple, y no dice *cómo* jugar — solo qué es un buen resultado. La IA tendrá que descubrir el resto.

> ⚠️ La recompensa define lo que aprende la IA. Si diéramos puntos por salto, aprendería a saltar sin parar. Este problema — que la IA optimice la nota en lugar del objetivo real — tiene nombre: *reward hacking*. También pasa con los LLMs.

---

## ¿Por qué varios dinos al mismo tiempo?

`Game` acepta `n_dinos`. Todos enfrentan **los mismos obstáculos** en la misma partida, lo que hace justa la comparación entre ellos. Esto será esencial en la próxima lección, en la que 50 dinos compiten para pasar sus "genes" a la siguiente generación.

## Ejercicios

1. ¿Cuántos puntos logras? (Spoiler: la IA va a llegar a 5000.)
2. Cuando aparece el primer pájaro, ¿cuál es el valor del sensor "elevation"? ¿Y para cada uno de los 3 tipos de pájaro (`bird_low`, `bird_mid`, `bird_high`)?
3. En `dino/game.py`, cambia `JUMP_FORCE` a `-12`. ¿El juego sigue siendo posible?

**Próxima lección:** [neuroevolución — 50 dinos evolucionando](../03-neuroevolution/README.es.md)

---

## Cambiando los dibujos

El dino, los cactus y el pájaro están dibujados en [`dino/sprites.py`](../../dino/sprites.py) como **texto**: `#` es un píxel pintado, `w` es blanco y `.` es transparente. Cambia los dibujos y crea el personaje que quieras.

Esto es **solo visual**. Para el juego, cada cosa sigue siendo un rectángulo (`rect` en `game.py`), y la colisión y los sensores usan esos rectángulos. La IA nunca ve los dibujos, solo los 6 números. Por eso cambiar el dibujo no cambia nada en el aprendizaje.
