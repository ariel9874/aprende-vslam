# Nivel 00 — Entorno y primeros píxeles

**Objetivo**: montar tu entorno de Python y perder el miedo al array — una
imagen digital ES una matriz de números, y todo el curso consiste en hacerle
matemática a esa matriz.

## Teoría mínima (5 minutos)

Una imagen a color es un arreglo de `alto x ancho x 3` números enteros de 8
bits (0–255): tres "capas" (canales) con la intensidad de azul, verde y rojo
en cada píxel. Ojo: OpenCV las carga en orden **BGR**, no RGB — la fuente de
un clásico "¿por qué mi imagen sale azulada?".

Convertir a **escala de grises** no es promediar los canales: el ojo humano
es mucho más sensible al verde que al azul, así que se usa una combinación
ponderada (estándar ITU-R BT.601, el de la televisión):

```
gris = 0.299·R + 0.587·G + 0.114·B
```

En este nivel la implementas A MANO y compruebas que coincide con la de
OpenCV. Ese patrón — implementar tú, verificar contra la referencia — se
repite en todo el curso.

El **histograma** cuenta cuántos píxeles hay de cada intensidad: es la
radiografía de la exposición de una imagen (¿está oscura? ¿saturada?) y
reaparecerá cuando hablemos del sensor (nivel 01) y de por qué el matching
falla con cambios de iluminación (nivel 06).

## Cómo correr

```bash
pip install -r requirements.txt
python descarga_datos.py            # baja TUM fr1_xyz (~450 MB, una vez)
python 00_hola_pixeles.py           # usa el primer frame del dataset
python verificacion.py              # el examen del nivel
```

¿Ya tienes el dataset en otra ruta? Todos los scripts aceptan
`--root <carpeta_de_la_secuencia>` o `--imagen <archivo>`.

Los resultados quedan en `salida/`: el gris hecho a mano, el negativo, un
recorte y el histograma.

## El dataset

TUM RGB-D `freiburg1_xyz`: una cámara Kinect movida a mano frente a un
escritorio, con la trayectoria real registrada por captura de movimiento
(ground truth). Es EL dataset del curso: lo usarás desde aquí hasta el SLAM
completo. Referencia: https://cvg.cit.tum.de/data/datasets/rgbd-dataset

## Qué debes poder explicar al terminar

- Qué hay dentro de un `ndarray` de imagen: shape, dtype, y por qué uint8.
- Por qué el gris es una combinación ponderada y no un promedio.
- Qué te dice un histograma sobre la exposición.
- BGR vs RGB (y cómo te muerde si lo ignoras).
