[English](README.md) · [Português](README.pt-BR.md) · **Español**

# Explicación a fondo: chips, memoria y cuantización

> ¿Qué tipos de chips ejecutan redes neuronales, desde una tarjeta gráfica para videojuegos hasta un centro de datos? ¿Por qué la
> *memoria* importa tanto como el chip? ¿Qué puede ejecutar o entrenar una computadora normal? ¿Y qué es
> la cuantización, el truco que permite ejecutar modelos grandes en máquinas pequeñas?

Las gráficas salen de dos scripts. El segundo usa el modelo de la lección 7, así que entrena primero la lección 7 si todavía no lo hiciste:

```bash
python deep-dives/hardware/memory.py
```

```bash
python deep-dives/hardware/quantization.py
```

Esta página parte de [¿Por qué GPU?](../gpu/README.es.md). Las especificaciones cambian rápido; los números de aquí son especificaciones publicadas en el momento de escribirla (fuentes al final).

---

## 1. Los tipos de chips

| Tipo | Ejemplos | Memoria que puede usar el modelo | Velocidad de la memoria | Bueno para |
|---|---|---|---|---|
| **PC normal, solo CPU** | Intel Core, AMD Ryzen, con RAM DDR5 | la RAM que tenga (32 GB es común) | ~80-100 GB/s | modelos pequeños, y todo lo que no sea una red grande |
| **Tarjeta gráfica para videojuegos** | NVIDIA GeForce RTX 4060 / 4090 / 5090, AMD Radeon | 8 / 24 / 32 GB en la tarjeta | 272 / 1 008 / 1 792 GB/s | rápida, siempre que el modelo quepa en la memoria de la tarjeta |
| **Memoria unificada** (CPU y GPU comparten una sola reserva) | chips Apple de la serie M, AMD Ryzen AI Max | hasta ~75 % del total en una Mac, por defecto; hasta 96 GB en el Ryzen AI Max+ 395 | 120 GB/s (M4) a 546 GB/s (M4 Max); 256 GB/s (Ryzen AI Max+ 395) | modelos grandes en una sola máquina, más despacio que una tarjeta gráfica grande |
| **GPU de centro de datos** | NVIDIA H100 / B200, AMD Instinct MI300X | 80 / 192 / 192 GB | 3 350 / 8 000 / 5 300 GB/s | entrenar y servir modelos grandes, en grupos de miles |

También hay chips hechos solo para redes neuronales, como las TPU de Google, que se usan en los propios centros de datos de Google.

**Analogía.** Piensa en la cocina de un restaurante. El chip son los cocineros; la memoria es la **despensa**. El *tamaño de la memoria* es qué tan grande es la despensa: si los ingredientes no caben, no se puede preparar el platillo. La *velocidad de la memoria* (ancho de banda) es qué tan ancha es la puerta de la despensa: cuántos ingredientes llegan a los cocineros por segundo. Miles de cocineros (una GPU) no sirven de nada si los ingredientes pasan por una puerta angosta.

---

## 2. Por qué la memoria importa tanto

### El modelo tiene que caber

El tamaño de un modelo es una cuenta sencilla: **número de pesos × bytes por peso**. Normalmente cada peso se guarda en 16 bits (2 bytes), así que un modelo con 7 mil millones de pesos ocupa 14 GB, y uno con 70 mil millones ocupa 140 GB.

![¿cabe?](charts/04_does_it_fit.png)

- Un **modelo de 7B en 4 bits** (3.5 GB) cabe casi en cualquier lugar, incluso en la Mac mini de este estudio.
- Un **modelo de 70B en 16 bits** (140 GB) solo cabe en las GPU de centro de datos más grandes. En cualquier otra cosa hay que repartirlo entre varias GPU, o encogerlo (sección 4).
- **Entrenar** necesita mucho más que ejecutar: además de los pesos, guarda el gradiente de cada peso y la memoria del optimizador (Adam guarda dos promedios móviles por peso). Una estimación común es **unos 16 bytes por peso**, así que entrenar incluso un modelo de 7B necesita unos 112 GB: fuera del alcance de una tarjeta para videojuegos.

### La velocidad de la memoria marca qué tan rápido escribe

Para escribir **un token**, un LLM tiene que leer **todos sus pesos** de la memoria una vez (el `generate.py` de la lección 7 ejecuta la red completa para cada carácter). Así que, por más rápido que sea el chip, hay un techo:

```
tokens por segundo  ≤  velocidad de la memoria  ÷  tamaño del modelo
```

![tokens por segundo](charts/05_tokens_per_second.png)

| Dispositivo | Modelo de 7B, 4 bits (3.5 GB) | Modelo de 70B, 4 bits (35 GB) |
|---|---|---|
| PC normal, solo CPU | 26 tokens/s como máximo | no cabe en 32 GB |
| GeForce RTX 4090 | 288 | no cabe |
| Mac mini M4 (este estudio) | 34 | no cabe |
| Mac con M4 Max, 128 GB | 156 | 16 |
| AMD Ryzen AI Max+ 395, 128 GB | 73 | 7 |
| NVIDIA H100 | 957 | 96 |
| NVIDIA B200 | 2 286 | 229 |

Estos son techos; las velocidades reales son menores. Las pruebas publicadas de modelos de 70B en 4 bits logran unos 4 a 6 tokens por segundo en el Ryzen AI Max+ 395, por debajo de su techo de 7, como se esperaba.

Por eso importa el **tipo de memoria**:

- **DDR5 / LPDDR5** (RAM normal): grande y barata, pero una puerta angosta.
- **GDDR6 / GDDR7** (tarjetas para videojuegos): soldada justo al lado del chip gráfico, con una puerta mucho más ancha, pero solo de 8 a 32 GB.
- **HBM** (GPU de centro de datos): pilas de chips de memoria colocadas justo al lado de la GPU, conectadas por miles de cables. La puerta más ancha de todas, y muy cara.
- **Memoria unificada** (Apple, AMD Ryzen AI Max): una sola reserva grande compartida por la CPU y la GPU, así que un modelo grande cabe, con una puerta de anchura intermedia entre las otras dos.

---

## 3. Lo que una computadora normal puede y no puede hacer

| | Una computadora normal | Un centro de datos |
|---|---|---|
| **Ejecutar** un modelo pequeño (hasta ~8B pesos, en 4 bits) | ✅ sin problema | ✅ |
| **Ejecutar** un modelo de 70B | ⚠️ solo con mucha memoria (una máquina grande con memoria unificada), en 4 bits, despacio | ✅ rápido |
| **Entrenar** algo como los modelos de este estudio | ✅ de segundos a minutos | ✅ |
| **Entrenar** un LLM grande desde cero | ❌ la memoria y el tiempo están fuera de alcance | ✅ miles de GPU durante semanas o meses |
| **Ajustar** (fine-tune) un poco un modelo pequeño | ⚠️ posible con trucos que entrenan solo unos pocos pesos extra | ✅ |

Y más allá de la memoria y la velocidad:

- **Energía y calor.** Una GeForce RTX 5090 está especificada para 575 W, una H100 para 700 W y una B200 para unos 1 000 W, cada una, y los centros de datos tienen miles de ellas funcionando.
- **Conexiones entre GPU.** Entrenar en miles de GPU solo funciona porque están conectadas por enlaces muy rápidos (900 GB/s por H100), para poder compartir sus resultados todo el tiempo. Las computadoras normales no tienen eso.

---

## 4. Cuantización: guardar los pesos con menos bits

**Analogía.** Piensa en los precios de una tienda. $3.4719 es exacto, pero $3.47 es suficiente, y $3 empezaría a causar problemas. La **cuantización** hace lo mismo con los pesos de una red: en lugar de guardar cada uno con 32 o 16 bits, los redondea a unos pocos valores permitidos, guardados con 8, 4 o incluso 2 bits.

Es lo mismo que guardar una imagen con menos colores:

![colores](charts/03_colors.png)

Con 16 colores, la gráfica se ve casi igual. Con 4, el degradado suave se convierte en franjas. Con 2, solo queda la forma general.

### Qué les pasa a los pesos

![redondeo](charts/01_rounding.png)

Estos son los 16 384 pesos de una tabla del modelo de la lección 7. Con 4 bits, cada peso se mueve al más cercano de 15 valores permitidos (en naranja). Con 2 bits solo hay 3 valores permitidos, y casi todos los pesos se vuelven 0.

### El experimento: el modelo de la lección 7, redondeado

[`quantization.py`](quantization.py) redondea cada tabla de pesos del diminuto modelo de lenguaje de la lección 7 y lo vuelve a medir con el texto que nunca vio:

![calidad](charts/02_quality.png)

| Bits por peso | Tamaño | Pérdida en texto nunca visto, una regla por tabla | Pérdida, una regla por fila | Lo que escribe después de "the dino " (una regla por fila) |
|---|---|---|---|---|
| 32 (original) | 434 KB | 1.642 | | `only matches teacher's when example find of the 292020000 15` |
| 8 | 114 KB | 1.643 | 1.642 | `only matches teacher's when example find of the 292020000 15` |
| 6 | 87 KB | 1.647 | 1.643 | `only matches teacher's when example find of the 292020000 15` |
| 4 | 61 KB | 1.806 | 1.699 | `only model step for from -1.99. the mird of the 200020000 15` |
| 3 | 47 KB | 2.996 | 1.982 | `only mateat the age appeare enviou + into siticndary; alll 5eps` |
| 2 | 34 KB | 4.552 | 4.122 | `o lh maeeap, led olenas es uess sluteinananho` |

- **8 y 6 bits casi no cambian nada.** El modelo escribe el mismo texto, con una cuarta parte del tamaño o menos.
- **4 bits cuestan un poco**, sobre todo con una regla por fila.
- **2 bits lo destruyen**: queda peor que un modelo que adivina cada carácter al azar (pérdida 3.97).

**Una regla por tabla vs una por fila.** Redondear necesita una *escala*, el tamaño de los escalones entre los valores permitidos. Con una sola escala para toda la tabla, las filas con pesos pequeños desperdician la mayoría de los escalones: todos sus pesos se redondean a los mismos pocos valores. Darle a cada fila su propia escala (o a cada grupo de unas cuantas decenas de pesos, como hacen las herramientas reales) conserva mucho más detalle. Con 3 bits, es la diferencia entre un modelo que todavía escribe casi-inglés (1.98) y uno que no (3.00).

Así es como la gente ejecuta modelos grandes en casa: los modelos suelen compartirse ya cuantizados a unos 4 bits. Un modelo de 70B pasa de 140 GB a unos 35 GB y puede caber en una máquina grande con memoria unificada. El precio es una pequeña pérdida de calidad que, como muestra el experimento, crece rápido por debajo de 4 bits.

---

## Resumen

| Pregunta | Respuesta corta |
|---|---|
| ¿Qué chip? | Para redes neuronales: una GPU, o un chip parecido. Las tarjetas para videojuegos son rápidas pero tienen poca memoria; las máquinas con memoria unificada tienen mucha memoria pero una puerta más angosta; las GPU de centro de datos tienen ambas cosas, a un precio enorme. |
| ¿Por qué importa la memoria? | El modelo tiene que caber (pesos × bytes por peso), y escribir cada token lee todos los pesos, así que la velocidad de la memoria limita la velocidad. |
| ¿Puedo entrenar un LLM en casa? | Uno diminuto, como el de la lección 7, sí. Uno grande, no: la memoria y el tiempo están fuera de alcance. |
| ¿Qué es la cuantización? | Redondear los pesos a menos bits: 8 bits sale casi gratis, 4 bits cuestan un poco, 2 bits rompen el modelo. |

---

## Fuentes

Las especificaciones de los dispositivos salen de estas páginas (consultadas en 2026):

- NVIDIA H100: [página del chip en InferenceX](https://inferencex.semianalysis.com/chips/h100), [ficha técnica de la ML cloud de Tubinga](https://portal.mlcloud.uni-tuebingen.de/user-guide/_specs/gpu/h100/)
- NVIDIA B200: [Jarvislabs](https://jarvislabs.ai/ai-faqs/nvidia-b200-specs), [Spheron](https://www.spheron.network/blog/nvidia-b200-complete-guide/)
- AMD Instinct MI300X: [hoja de datos de AMD](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf), [Lenovo Press](https://lenovopress.lenovo.com/LP1943)
- GeForce RTX 5090: [TechPowerUp](https://www.techpowerup.com/gpu-specs/geforce-rtx-5090.c4216)
- GeForce RTX 4060 y 4090: [comparación en GPUprix](https://gpuprix.com/es/compare/geforce-rtx-4060-vs-geforce-rtx-4090)
- Apple M4 (Mac mini): [especificaciones de la Mac mini de Apple](https://images.apple.com/ie/mac-mini/specs/)
- Apple M4 Max: [MacRumors](https://macrumors.com/2024/10/30/apple-announces-m4-max-chip); límite de memoria de la GPU en macOS: [APXML](https://apxml.com/hardware/apple-m4-max-128gb)
- AMD Ryzen AI Max+ 395: [página del producto de AMD](https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo/ryzen-ai-max-plus-395.html), [pruebas de ar-go.co](https://ar-go.co/blog/amd-ryzen-ai-max-395-local-llm/)
- Ancho de banda de DDR5: [HotHardware](https://hothardware.com/news/ddr5-memory-spec-finalized-2tb-64gbps)

**Volver a:** [la página principal del estudio](../../README.es.md)
