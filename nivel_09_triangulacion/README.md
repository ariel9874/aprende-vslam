# Nivel 09 — Triangulación: los matches se vuelven puntos 3D

**Objetivo**: recuperar la profundidad que la cámara destruyó (nivel 02) —
convertir correspondencias en PUNTOS 3D — y aprender a no fiarte de ellos:
un mapa sucio es peor que un mapa pequeño.

Este es tu primer MAPA. Lo vas a exportar a un `.ply` y abrirlo en un visor.

## Teoría mínima

**Dos rayos se cortan (casi).** El nivel 02 te enseñó que un píxel define un
RAYO, no un punto. Con dos vistas de pose conocida (nivel 07) tienes dos
rayos del mismo punto físico: donde se cruzan está el punto 3D. En la
práctica no se cruzan exactamente (ruido de píxel), así que se resuelve por
mínimos cuadrados: el método **DLT** (Direct Linear Transform) apila las
ecuaciones de ambas vistas y toma el vector singular más pequeño de la SVD.
La derivación completa está en el script.

**La DLT no sabe decir que no.** El sistema lineal SIEMPRE devuelve un
punto, aunque las correspondencias sean basura. Hay que filtrar con tres
criterios geométricos (los tres implementados y medidos aquí):

1. **Quiralidad**: el punto debe estar DELANTE de ambas cámaras. Un punto
   "detrás" es geometría imposible.
2. **Reproyección**: proyectar el punto 3D de vuelta debe caer cerca de las
   dos observaciones que lo generaron (< 2 px). El error algebraico de la
   DLT no es el geométrico.
3. **Paralaje**: el ángulo entre los dos rayos. Con rayos casi paralelos
   (poco baseline frente a la profundidad) la intersección está mal
   condicionada: un error de ε píxeles mueve el punto ~ `profundidad² /
   (baseline · f)`. Son los puntos "en el infinito", y envenenan cualquier
   PnP posterior (nivel 10).

**Por qué importa el filtro.** El repo padre midió esto de la peor forma: un
keyframe insertado desde una pose dudosa creó 584 puntos basura, y el
tracking empezó a teletransportarse metros. La regla que salió de ahí:
*nunca crear mapa desde una pose incierta*. Aquí lo vas a ver en pequeño.

## Cómo correr

```bash
pip install -r requirements.txt
python genera_datos.py         # la secuencia sintetica (GT exacto)
python 09_triangulacion.py     # triangula, filtra, mide y exporta el PLY
python verificacion.py         # el examen del nivel
```

Resultados en `salida/`: `mapa.ply` (ábrelo en MeshLab, CloudCompare o el
visor de Windows), `mapa_planta.png` y `reproyeccion.png`.

## Qué debes poder explicar al terminar

- Por qué dos vistas bastan y una no.
- Qué resuelve la DLT y por qué su solución hay que filtrarla.
- Los tres filtros y qué patología ataja cada uno.
- Por qué el paralaje pequeño hace explotar la incertidumbre en profundidad
  (y por qué eso te perseguirá en todo el curso).
