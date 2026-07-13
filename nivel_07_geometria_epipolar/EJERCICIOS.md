# Ejercicios — Nivel 07

## 1. La restricción, a mano (fácil)

Para 10 inliers, calcula tú mismo x̂_bᵀ·E·x̂_a (normaliza los píxeles con
K⁻¹) y verifica que sale ~0. Hazlo también con 10 matches DESCARTADOS por
RANSAC.

**Objetivo**: histograma de |x̂ᵀEx̂| para inliers vs outliers — la
separación que RANSAC explota. ¿Dónde pondrías tú el umbral?

## 2. El barrido del baseline (medio)

Corre `--par 0 K` con K en {1, 2, 4, 6, 10, 20} y tabula: error de R, error
de dirección de t, e inliers.

**Objetivo**: la curva en U del baseline — con K pequeño la dirección de t
es ruidosa (poco paralaje: el numerador de la señal es minúsculo frente al
ruido de píxel), con K grande el matching se degrada (la vista cambió
demasiado). Es la misma tensión del ejercicio 3 del nivel 08, ahora medida
en grados contra GT exacto.

## 3. Rotación pura, el caso degenerado (medio)

Genera un par sintético SIN traslación: rota la imagen A unos grados con
`cv2.warpAffine` alrededor del centro óptico (equivale a una cámara que solo
rota si la escena está lejos... ¿por qué? — piénsalo con la homografía del
nivel del generador) y corre la estimación.

**Objetivo**: verifica que la dirección de t recuperada es basura
(compárala entre varias corridas de RANSAC: debe cambiar erráticamente)
mientras que R sale bien. Documenta los números — es el caso degenerado
anunciado en el nivel 08.

## 4. Datos reales con mocap (difícil)

Repite el experimento 1 con dos frames de TUM fr1_xyz separados 30 frames
(calibración fr1: `fx=517.3 fy=516.5 cx=318.6 cy=255.3`, distorsión
`k1=0.2624 k2=-0.9531 p1=-0.0054 p2=0.0026 k3=1.1633` — pre-rectifica con
`cv2.undistort`, como hará el nivel 14). El GT de TUM viene de captura de
movimiento en el marco del CUERPO del sensor, no de la cámara óptica: el
error de R contra ese GT incluye un extrínseco fijo desconocido.

**Objetivo**: compara la MAGNITUD del ángulo de rotación (invariante al
marco: tr(R) = 1 + 2cos θ) estimada vs GT en varios pares. ¿A cuántos
grados de acuerdo llegas? ¿Y si NO pre-rectificas la distorsión?

## 5. La degeneración planar (difícil — pasó de verdad construyendo este nivel)

En `genera_datos.py`, sustituye los tres planos por UNO solo gigante (p. ej.
`TexturedPlane(2.0, 0.0, 8.0, 12.0, 7.0, ...)`), regenera y corre la
estimación del par 0→6 varias veces.

**Objetivo**: observa que (a) la distancia de los puntos a las epipolares
sigue siendo diminuta (~0.3 px: ¡la E "ajusta"!), pero (b) el error de
dirección de t es enorme y/o errático. Explícalo: los puntos de UN plano se
relacionan por una homografía H, y hay una FAMILIA entera de pares (R, t)
compatibles con ella — la restricción epipolar no basta para elegir. Es la
lección 2 del repo padre (su init valida con una TERCERA vista por esto
mismo, nivel 10) y la razón de que ORB-SLAM inicialice eligiendo por consenso
entre modelo H y modelo E. Cuando construimos este nivel, la escena original
(dominada 91% por un plano) nos dio 80.9° de error de t con la E ajustando a
0.4 px — el número está en el docstring del generador.
