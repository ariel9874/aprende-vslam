# Nivel 16 — Estéreo (EuRoC) · electiva

**Objetivo**: la cámara derecha VIRTUAL del nivel 15 se vuelve REAL. Un rig
estéreo calibrado en un dron, rectificación epipolar, profundidad por
disparidad — y el mismo tracker métrico corriendo sin tocarlo.

## La idea en una línea

En el nivel 15 sintetizabas `u_R = u − bf/z` desde la profundidad del Kinect,
con `bf = 40` por convención. Aquí `u_R` se MIDE (es el match en la imagen
derecha) y `bf` lo firma la calibración del rig (`bf = −P2[0,3]` ≈ 47.9).
Mismo residuo, misma init instantánea, mismo bucle SE(3). **Un backend bien
factorizado no distingue el sensor** (lección 37 del padre).

## Las tres piezas nuevas de datos (todas en `dataset.py`)

1. **El rig** (`RigEstereo`): `cv2.stereoRectify` toma los dos `sensor.yaml`
   (intrínsecos + distorsión + extrínsecos T_BS) y la pose relativa
   cam0←cam1, y devuelve los mapas que reproyectan ambas imágenes a un par
   virtual con planos coplanares. Tras rectificar, las rectas epipolares del
   nivel 07 se vuelven **FILAS**: la correspondencia colapsa de una búsqueda
   2D a una resta de columnas. La izquierda queda como pinhole SIN distorsión.

2. **La disparidad** (`CargadorEstereo`): SGBM matchea denso a lo largo de
   cada fila; `z = bf/d`. El loader entrega `(ts, gris, profundidad)` — la
   MISMA firma que la SecuenciaTUM del nivel 15, y por eso el tracker no
   cambia. `0 = sin dato`, como siempre.

3. **El ground truth con brazo de palanca** (`leer_gt_euroc`): el GT de EuRoC
   vive en el frame del CUERPO (IMU). La cámara está a ~7 cm; ese brazo ROTA
   con la pose y NO lo absorbe la alineación del ATE. Se corrige con
   `p_cam = R_wb·t_BS + p_body` — la trampa perfecta a la escala del error
   que quieres medir (ejercicio 1).

## Las dos piezas que el dron exigió (medidas, una por una)

El plan era no tocar el tracker. El dron tenía otro plan, y este nivel se
construyó midiéndolo — la progresión completa: **54 → 37 → 9 cm**.

**1. La RELOCALIZACIÓN** (54 → 37 cm). El primer intento se perdió en el
frame ~1770 y NUNCA volvió: el coast por velocidad constante EXTRAPOLA un
vuelo 6-DoF y diverge en metros en segundos (1141 frames perdidos). En las
secuencias de mano de los niveles 14/15 perderse era raro; en un dron, un
solo tropiezo sin salida cuesta el resto de la secuencia. La salida (aplazada
desde el nivel 13, por fin en el código enseñado): tras 5 frames perdidos,
reconocer lugar contra TODA la base de keyframes (sin filtro temporal — a
diferencia del bucle) y saltar a la pose que diga la geometría. La vara es la
del bucle: **PnP con ≥ 40 inliers, o nada** — re-engancharse a una pose
equivocada es peor que seguir perdido. Y el detalle de la lección 24 del
padre: tras el salto, el mapa local se RE-ANCLA en el keyframe reconocido y
sus covisibles (la recencia apunta a la zona de ANTES de perderse).

**2. El FILTRO ANTI-DUPLICADOS** (37 → 9 cm). Con la reloc, la secuencia se
cruzaba entera... y aún daba 37 cm. La gráfica señaló al ARRANQUE: el dron
pasa ~9 s quieto antes de despegar, y en ese tramo nuestra estimación vagaba
~1 cm/frame (¡sin moverse!) hasta desplazar el ancla de la sesión ~1.5 m. La
causa: cada keyframe estático re-creaba cientos de COPIAS de los mismos
puntos (ORB no re-detecta idéntico, así que quedaban "sin punto asignado") y
el PnP saltaba entre copias. El filtro del padre (v0.4): si un candidato cae
a < 1.5% de su profundidad de un punto local existente, es la misma
característica física y NO se crea. El mapa bajó de 66k a 26k puntos y el
ancla dejó de vagar.

## La simetría bonita (lección 37)

El ruido de la profundidad estéreo crece con z²:  ∂z/∂d = −bf/d² = −z²/bf.
El peso del residuo `u_R` en el BA **decae** con z²:  ∂u_R/∂z = bf/z².
La geometría compensa el ruido exactamente en la dirección correcta — la
misma cancelación que hacía funcionar al Kinect, ahora con la z triangulada.

## El examen NO necesita el dataset

EuRoC pesa ~1.1 GB, así que `verificacion.py` fabrica su propio rig: dos
cámaras sintéticas con baseline de 10 cm (bf = 40) y un plano fronto-paralelo
a 2.5 m. Verifica que `stereoRectify` recupera la geometría, que SGBM
recupera el plano, y la identidad que cierra el bloque:

```
u_R medida = u_L − d = u_L − bf/z = u_R sintetizada (nivel 15)
```

Real y virtual son la MISMA ecuación. El dataset es para el driver.

## Los números (V1_01_easy: un dron real, 6-DoF, 2912 frames)

Medidos con este código, secuencia completa:

| | valor |
|---|---|
| keyframes / bucles SE(3) / relocs | 181 / 8 / 3 |
| frames perdidos | 228 (ráfagas rápidas del vuelo) |
| mapa | 25 842 puntos |
| ATE final-KF RÍGIDO (pre-GBA) | 83.7 cm |
| **ATE final-KF RÍGIDO + BA global** | **9.0 cm** |
| **escala de similitud** | **1.004** — metros del rig, sin gauge |

La referencia del padre: **6.9 cm, escala 1.002** (34 perdidos, con compuerta
de movimiento, BoW y culling — las piezas que a este nivel le faltan; la
compuerta es el ejercicio 0). El ATE online es 170 cm aquí y 63.8 cm en el
padre: el vuelo agresivo mete excursiones per-frame y tramos de coast que se
congelan al emitirse — la métrica del sistema es la final de keyframes
(nivel 13). Fíjate en el salto pre→post GBA (83.7 → 9.0): con un mapa LIMPIO
(sin duplicados), el BA global sí puede hacer su trabajo.

## Cómo correr

```bash
pip install -r requirements.txt
python verificacion.py            # el examen: SIN dataset, <1 min
python descarga_datos.py          # V1_01_easy, ~1.1 GB (solo para el driver)
python 16_estereo.py              # el dron completo (~30 min con SGBM)
```

Nota de datos: el host oficial de EuRoC lleva temporadas caído; la descarga
usa el mirror de HuggingFace `pepijn223/euroc-mirror` (mismo formato).

## Qué debes poder explicar al terminar

- Qué hace exactamente la rectificación con las rectas epipolares del
  nivel 07, y por qué convierte el matching denso en una búsqueda 1D.
- De dónde sale `bf` en un rig real, y qué era en el nivel 15.
- Por qué el GT de EuRoC no se puede comparar directo contra tu trayectoria
  (el brazo de palanca que rota).
- La cancelación z² del ruido y el peso — y por qué aún así existe DEPTH_MAX.
- Por qué la ruta métrica del tracker no necesitó ni una línea nueva (y qué
  dice eso del diseño del residuo [u, v, u_R]).
- Por qué el dron sí exigió la relocalización, en qué se diferencia del
  cierre de bucle (¿qué filtro pierde?), y por qué el mapa local debe
  re-anclarse tras el salto.
