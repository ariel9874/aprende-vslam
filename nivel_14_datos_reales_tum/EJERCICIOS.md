# Ejercicios — Nivel 14

El primero es el ejercicio de honestidad del curso: medir dónde tu sistema
NO funciona, y entender por qué, vale más que otro decimal de ATE.

## 1. fr1_desk: el límite, documentado (el ejercicio estrella)

Baja la secuencia handheld (`python descarga_datos.py --fr1-desk`) y corre:

```bash
python 14_datos_reales.py --root data/rgbd_dataset_freiburg1_desk
```

**Objetivo**: documenta el fallo como lo haría un ingeniero: cuántos frames
perdidos, en qué frame muere por primera vez, y qué pasa con el mapa después.
Mira 4-5 imágenes alrededor de esa primera muerte (están en `rgb/`) y ponle
nombre a lo que ves.

Lo que vas a encontrar (lección 28 del repo padre): fr1_desk es una cámara EN
LA MANO con rotación rápida y motion blur. El matching ORB — incluso guiado —
no engancha en los tramos borrosos, y este sistema no tiene relocalización:
perderse es definitivo. Medido con este código: **562 de 613 frames
perdidos** (muere en la primera ráfaga de blur, antes del frame ~60, y ya no
vuelve; ATE online 85 cm; sólo 5 keyframes). El padre pierde **560** ahí
mismo — CON relocalización: el envelope es casi idéntico porque el límite no
es la arquitectura, es el frontend ORB contra el blur. Las features
aprendidas lo bajan a ~140 (nivel 17) y el residuo de profundidad RGB-D lo
cruza entero (nivel 15). No es un umbral mal puesto: es el borde del
envelope. Saber dónde está es el resultado.

## 2. El barrido de la lección 27 (fácil, lento)

El BA global del examen corre 50 iteraciones. Repite la secuencia del examen
midiendo el ATE final-KF con GBA de 0, 10, 25 y 50 iteraciones
(`s.global_bundle_adjustment(iterations=N)` sobre el MISMO tracker recién
corrido — ojo: re-corre el tracking para cada N, el GBA muta el mapa).

**Objetivo**: la curva ATE vs iteraciones. En el tramo del examen la deriva
es pequeña y la curva es plana; en la secuencia ENTERA el padre midió
13.0 → 12.0 → 3.5 → 0.4 cm (0/10/25/50). La moraleja vale para toda tu
carrera: antes de creer que un residual es "de fondo", verifica que el
optimizador CONVERGIÓ.

## 3. La ventana del guiado (fácil)

`GUIDED_RADIUS_PX` vale 15 (el radio de ORB-SLAM). Corre el examen con 5 y
con 40, midiendo inliers (mediana) y frames perdidos.

**Objetivo**: explica los dos extremos. Con 5 px la ventana no perdona ni el
error del prior de velocidad constante (estrangula los candidatos); con 40 px
readmite la ambigüedad que el guiado venía a eliminar. El punto dulce existe
porque el prior es bueno pero no perfecto.

## 4. La perilla que dejó de importar (medio)

`KF_HEALTH_INLIERS` vale 45. En el repo padre, antes del matching guiado, esta
perilla era un trade-off POR SECUENCIA (su lección 21): 45 mataba de hambre a
fr2_desk (1347 perdidos), 25 envenenaba fr1_xyz (6.9 → 18.4 cm). Corre las 4
combinaciones sobre el tramo del examen: health ∈ {45, 25} × guiado ∈ {ON, OFF}.

**Objetivo**: con guiado ON, 45 y 25 deben dar prácticamente lo mismo (los
inliers viven muy por encima del piso). La lección 24 del padre en tus manos:
la cura de una perilla sensible no siempre es calibrarla mejor — a veces es
arreglar lo que la hacía sensible.

## 5. ¿Dónde se va el tiempo? (medio — el puente al nivel 18)

Envuelve las tres etapas caras (`detectAndCompute`, `_guided_match`,
`_ba_local`) con `time.perf_counter()` acumulado y corre 600 frames.

**Objetivo**: tu propia tabla de perfilado, en % del tiempo total. El padre
midió BA 57% + matching guiado 37% + resto 6% — y su intuición previa
(apostaba por cv2) estaba equivocada. Perfila antes de optimizar: ese es el
método completo del nivel 18, donde estas dos etapas se sustituyen por
gemelas rápidas (GTSAM, C++) con test de equivalencia.
