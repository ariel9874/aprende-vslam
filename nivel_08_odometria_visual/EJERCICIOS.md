# Ejercicios — Nivel 08

Cada ejercicio tiene un número objetivo. Trabaja sobre una copia de
`08_odometria_visual.py`.

## 1. El barrido del ratio test (fácil)

Corre la VO con `RATIO` en {0.6, 0.7, 0.75, 0.8, 0.9} y tabula
matches / inliers / ATE.

**Objetivo**: encuentra el punto donde relajar el ratio deja de ayudar y
explica por qué (más matches ≠ mejores matches: RANSAC paga los outliers en
iteraciones y en riesgo).

## 2. La trampa de recoverPose, en vivo (medio)

Cambia `CHEIRALITY_DIST_THRESH` de 2000 a 50 (el default efectivo de la
sobrecarga básica de OpenCV) y corre.

**Objetivo**: reproduce la lección 1 del repo padre — los inliers deben
desplomarse en los tramos donde la escena está lejos respecto al paso entre
frames. Reporta en qué frames colapsa y a cuánto.

## 3. Saltarse frames = más baseline (medio)

Procesa solo uno de cada K frames (K = 2, 4, 8) reindexando la secuencia.

**Objetivo**: dos efectos en tensión — más baseline mejora el
condicionamiento de E (mejor dirección de t), pero el matching se degrada al
cambiar más la vista. Grafica ATE vs K y encuentra el óptimo de ESTA escena.

## 4. Datos reales, sin red de seguridad (difícil)

Corre tu VO sobre los primeros 300 frames de TUM `freiburg1_xyz` (el dataset
del nivel 00; calibración fr1: `fx=517.3 fy=516.5 cx=318.6 cy=255.3`).
Compara contra su `groundtruth.txt`... con cuidado: el GT va a 100 Hz y las
imágenes a 30 (tendrás que asociar por timestamp, o comparar solo la forma).

**Objetivo**: observa y documenta AL MENOS dos maneras en que lo real rompe
lo que en sintético funcionaba (pistas: velocidad no constante rompe la
suposición de escala; motion blur mata keypoints; la distorsión de lente
sesga la geometría — nivel 04). Este ejercicio es el trailer del nivel 14.
