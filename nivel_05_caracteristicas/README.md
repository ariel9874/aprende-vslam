# Nivel 05 — Características: qué mirar de una imagen

**Objetivo**: decidir QUÉ puntos de una imagen valen la pena — esquinas — y
cómo describirlos para re-encontrarlos desde otro punto de vista. Implementas
el detector de Harris A MANO en numpy, lo verificas contra OpenCV y mides
qué significa "invariante".

## Teoría mínima

**Por qué esquinas (el problema de apertura).** Mirando por una ventanita,
un tramo de borde recto se ve IGUAL si se desliza a lo largo de sí mismo: es
imposible saber cuánto se movió en esa dirección. Una esquina — dos bordes
que se cruzan — es reconocible sin ambigüedad en las dos direcciones. Harris
formaliza esto con el tensor de estructura (la matemática está en el script):
sus dos autovalores miden el contraste en las dos direcciones principales;
esquina = ambos grandes.

**Del punto al descriptor.** Detectar no basta: hay que RECONOCER el mismo
punto en otra imagen. Un descriptor resume el parche alrededor del punto en
un vector comparable:

- **ORB** (binario, 256 bits): comparaciones de brillo entre pares de
  píxeles. Rapidísimo de extraer y comparar (distancia de Hamming = XOR +
  contar bits). El caballo de batalla de los SLAM en tiempo real.
- **SIFT** (flotante, 128-D): histogramas de orientación de gradiente.
  Más discriminativo y robusto, más caro. El patrón oro clásico.

**Invarianza.** Una detección sirve si SOBREVIVE a los cambios de punto de
vista: rotación (el descriptor se orienta con el gradiente dominante) y
escala (se detecta en una pirámide de resoluciones). En este nivel no te lo
crees: lo MIDES (repetibilidad = % de puntos re-detectados en el mismo sitio
tras rotar/escalar la imagen).

## Cómo correr

```bash
pip install -r requirements.txt
python descarga_datos.py            # TUM fr1_xyz (~450 MB; el mismo del nivel 00)
python 05_caracteristicas.py       # Harris a mano + comparador + invarianza
python verificacion.py             # el examen del nivel
```

¿Ya tienes fr1_xyz de otro nivel? Pasa `--root <carpeta_de_la_secuencia>`.

Resultados en `salida/`: `harris_propio_vs_cv2.png`, `detectores.png`
(GFTT/ORB/SIFT lado a lado) y la tabla de repetibilidad impresa en consola.

## Qué debes poder explicar al terminar

- El problema de apertura y por qué las esquinas lo resuelven.
- Qué miden los autovalores del tensor de estructura.
- Binario vs flotante: qué se paga y qué se gana.
- Qué número mide la "invarianza" y cómo lo obtuviste.
