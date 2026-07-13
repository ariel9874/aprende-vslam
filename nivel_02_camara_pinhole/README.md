# Nivel 02 — La cámara pinhole: de 3D a píxeles

**Objetivo**: entender la cámara como una FUNCIÓN matemática — proyectar
puntos 3D a píxeles con la matriz K — y construir con eso tu primer
renderizador: un cubo alámbrico en numpy puro, sin OpenGL.

## Teoría mínima

**El modelo pinhole.** Una cámara ideal es un agujero: cada punto 3D
`(X, Y, Z)` (en el marco de la cámara: +Z hacia delante, +Y hacia abajo —
la convención de OpenCV que usará TODO el curso) se proyecta por triángulos
semejantes:

```
u = fx · X/Z + cx
v = fy · Y/Z + cy
```

- `fx, fy`: la distancia focal EN PÍXELES (focal física ÷ tamaño del
  fotosito — por eso no se mide en mm aquí).
- `cx, cy`: el punto principal, donde el eje óptico pincha el sensor.

En forma matricial, con coordenadas homogéneas:

```
        [fx  0  cx]
K  =    [ 0 fy  cy]        [u, v, 1]ᵀ ~ K · [X, Y, Z]ᵀ / Z
        [ 0  0   1]
```

**Lo que se pierde.** Dividir por Z destruye la profundidad: TODOS los
puntos del rayo que pasa por el agujero y por el píxel `(u, v)` se proyectan
al mismo sitio. Una imagen es un haz de rayos, no un mapa 3D — recuperar la
Z perdida es literalmente el resto del curso (dos vistas: nivel 07;
triangulación: nivel 09; sensores de profundidad: nivel 15).

En este nivel usamos la K real de la cámara de TUM `freiburg1`
(fx=517.3, fy=516.5, cx=318.6, cy=255.3) — los mismos números que
calibrarás tú en el nivel 04 y usarás en el SLAM de los niveles 8+.

## Cómo correr

```bash
pip install -r requirements.txt
python 02_pinhole.py         # proyecciones + el cubo alambrico animado
python verificacion.py      # el examen del nivel
```

Resultados en `salida/`: `cubo_giro.png` (mosaico de la animación),
`cubo.avi` (el video) y `fov_vs_fx.png` (el zoom es solo multiplicar).

## Qué debes poder explicar al terminar

- Qué significa cada número de K y sus unidades.
- Por qué la proyección es una división por Z y qué información destruye.
- Qué es el rayo de un píxel (backprojection) y por qué "ida y vuelta"
  cierra exacto si conoces Z.
- Qué le hace la focal al campo de visión (y por qué el zoom no es
  perspectiva: es recorte).
