# Ejercicios — Nivel 01

## 1. El trío de exposiciones (fácil)

Genera la misma escena con exposición 0.05, 0.5 y 1.5, conviértelas a uint8
y compara sus histogramas (nivel 00).

**Objetivo**: identifica en cada histograma el problema dominante (ruido /
correcto / saturación) y cuenta cuántos píxeles quedaron clavados en 255 con
1.5 — esa es exactamente la información que ninguna calibración recupera.

## 2. Caracteriza el sensor como un ingeniero (medio)

Sin mirar las constantes del script, estima FULL_WELL y READ_NOISE solo con
mediciones: el ruido de lectura es la sigma a exposición ~0, y el full well
es donde la varianza se desploma al saturar. (Es la "photon transfer curve"
completa: sigma² vs media en todo el rango.)

**Objetivo**: tus estimaciones dentro del 15% de los valores reales del
simulador (compruébalas al final).

## 3. Artefactos de demosaico (medio)

Añade a la escena de color una rejilla fina de líneas negras de 1 px cada
3 px y demosaica.

**Objetivo**: captura el moiré / falso color en un recorte ampliado y
explica por qué la interpolación bilineal inventa colores justo ahí
(pista: la frecuencia de la rejilla vs la frecuencia de muestreo de cada
canal — el canal R muestrea 1 de cada 2 píxeles en cada eje).

## 4. Rolling shutter con tu teléfono (medio, sin código)

Graba con el móvil las aspas de un ventilador o pasa la cámara rápido
frente a postes verticales.

**Objetivo**: una foto tuya con el artefacto + el cálculo inverso: a partir
de la inclinación medida en píxeles y el tiempo de lectura típico (~30 ms),
estima la velocidad del objeto. Compara con la fórmula pendiente = v/H del
script.

## 5. El gris del nivel 00, revisitado (fácil)

El sensor entrega Bayer crudo. ¿Qué pasa si calculas el gris ponderado del
nivel 00 DIRECTAMENTE sobre el crudo (sin demosaicar), tratándolo como si
fuera gris de verdad?

**Objetivo**: muestra el patrón de tablero de ajedrez residual que aparece
y su periodo (2 px), y explica de dónde sale.
