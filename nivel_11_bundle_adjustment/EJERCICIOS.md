# Ejercicios — Nivel 11

## 1. El Schur, cronometrado (fácil, y muy convincente)

Implementa la versión INGENUA: construye la matriz H completa
`(6K + 3P) × (6K + 3P)` y resuélvela con `np.linalg.solve`, sin Schur.
Compara resultados (deben coincidir a 1e-9) y CRONOMETRA ambas con
60, 200 y 600 puntos.

**Objetivo**: la curva de tiempos. El Schur resuelve un sistema de 18×18
(3 cámaras libres) sin importar cuántos puntos haya; el ingenuo crece como
el cubo del número de puntos. Ese es el truco que hace posible el SLAM.

## 2. La estructura de S ES la covisibilidad (medio)

Dibuja el patrón de dispersión (`plt.spy`) de la matriz `S` reducida. Luego
construye una escena donde las cámaras 0-1 vean unos puntos y las 3-4 vean
otros DISTINTOS (sin solapamiento).

**Objetivo**: verás que el bloque (0,1)-(3,4) de `S` es CERO — dos cámaras
sólo se acoplan si comparten puntos. El grafo de covisibilidad no es un
truco de implementación del nivel 13: es literalmente la estructura de las
ecuaciones normales.

## 3. La escala se ve en el espacio nulo (medio — teórico y bonito)

Con UNA sola cámara anclada, construye la matriz `S` y calcula sus valores
singulares (`np.linalg.svd`).

**Objetivo**: encontrarás un valor singular ~0 (el resto, grandes). Ese es
el séptimo grado de libertad, visible con los ojos. Recupera su vector
singular y comprueba que corresponde a una expansión/contracción de la
escena. Con dos anclas, ese valor singular deja de ser ~0.

## 4. Anclas envenenadas (medio — el error que cometí construyendo el nivel)

Perturba las cámaras ANCLADAS (1 cm y 1° de error) y corre el BA.

**Objetivo**: mide el error final de los puntos. Verás que explota (medí
~1 m con este baseline), aunque la reproyección quede baja. El BA reconstruye
una escena perfectamente consistente... con un marco de referencia
equivocado. Explica la amplificación con la ley del nivel 09:
`dZ = ε·Z²/(f·B)`, con B = 36 cm el baseline entre las anclas. Moraleja: el
BA no puede arreglar lo que le clavas mal — y por eso en un SLAM real las
anclas son poses YA optimizadas.

## 5. Huber vs rechazo explícito (difícil)

Añade un paso de rechazo: tras 5 iteraciones, elimina las observaciones con
residuo > 3σ (test chi²) y sigue optimizando.

**Objetivo**: compara los tres regímenes (cuadrático puro / Huber / Huber +
rechazo) con 10% de outliers. Deberías recuperar casi el error sin outliers.
Y prueba qué pasa si rechazas DEMASIADO pronto (en la iteración 1, cuando la
semilla aún es mala): rechazarás inliers buenos que simplemente aún no
convergían. El orden importa.
