# Nivel 01 — El sensor de imagen: del fotón al ndarray

**Objetivo**: entender qué pasa entre la luz y la matriz de números del
nivel 00 — y por qué las imágenes reales traen ruido, saturación, colores
interpolados y distorsiones de movimiento. Todo con un SIMULADOR que
escribes y mides tú (no hace falta dataset ni webcam).

## Teoría mínima

**El fotosito.** Cada píxel del sensor es un pozo que cuenta fotones
convertidos en electrones durante el tiempo de exposición. Ese conteo es un
proceso de Poisson: si esperas N electrones, la desviación estándar es
**√N** (el *shot noise* — no es un defecto del sensor, es física de contar
cosas discretas). A eso se suma el ruido de LECTURA de la electrónica
(gaussiano, constante). Consecuencia medible: las zonas oscuras son
proporcionalmente más ruidosas — lo verás en una gráfica log-log con
pendiente 1/2.

**La saturación.** El pozo tiene capacidad finita (*full well*). Cuando se
llena, el conteo se trunca: información PERDIDA para siempre (nivel 00:
el pico en 255 del histograma; nivel 05: sin gradiente no hay esquinas).

**El mosaico de Bayer.** El sensor no ve color: cada fotosito lleva un
filtro R, G o B en un patrón RGGB (el doble de verdes porque el ojo pesa más
el verde — la misma razón del 0.587 del nivel 00). Dos de cada tres valores
de color de tu imagen son INVENTADOS por interpolación (demosaicado). En
este nivel demosaicas a mano y mides el error.

**Rolling shutter.** La mayoría de sensores CMOS no capturan todas las filas
a la vez: las leen de arriba abajo con un retardo. Si algo se mueve mientras
tanto, la imagen se INCLINA (las verticales se vuelven diagonales). Lo vas a
simular y medir la inclinación exacta. Guárdalo en la memoria: el repo padre
midió que el rolling shutter + motion blur de las secuencias handheld es el
techo fotométrico de su mapa denso (su lección 41).

## Cómo correr

```bash
pip install -r requirements.txt
python 01_sensor.py          # los tres experimentos, con graficas en salida/
python verificacion.py       # el examen del nivel
```

Resultados en `salida/`: `ruido_vs_senal.png`, `bayer_demosaico.png`,
`rolling_shutter.png`.

## Qué debes poder explicar al terminar

- Por qué el ruido crece como la raíz de la señal, y qué implica para las
  zonas oscuras.
- Qué se pierde exactamente al saturar.
- Por qué 2/3 del color de una foto es interpolado, y qué artefactos deja.
- Qué le hace un rolling shutter a una línea vertical en movimiento, y
  cuánto (fórmula y medición).
