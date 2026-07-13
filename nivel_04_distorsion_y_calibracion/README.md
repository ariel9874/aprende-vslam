# Nivel 04 — Distorsión y calibración: de la cámara ideal a la real

**Objetivo**: estimar K y la distorsión de una cámara con el flujo estándar
de tablero de ajedrez — y verificar el resultado contra la verdad, porque
aquí la cámara es SIMULADA con la distorsión REAL de TUM freiburg1. Con
webcam propia es el mismo código (ejercicio 4).

## Teoría mínima

**La lente miente.** El modelo pinhole (nivel 02) supone rayos rectos por un
punto. Una lente real curva los rayos, más cuanto más lejos del centro: las
rectas del mundo salen CURVAS en la imagen (efecto barril/cojín). El modelo
clásico (Brown-Conrady) corrige en coordenadas normalizadas `(x, y) = 
((u−cx)/fx, (v−cy)/fy)` con `r² = x² + y²`:

```
x' = x·(1 + k1·r² + k2·r⁴ + k3·r⁶) + 2·p1·x·y + p2·(r² + 2·x²)
y' = y·(1 + k1·r² + k2·r⁴ + k3·r⁶) + p1·(r² + 2·y²) + 2·p2·x·y
```

- `k1, k2, k3`: distorsión RADIAL (polinomio en r² — simétrica respecto al
  centro óptico).
- `p1, p2`: TANGENCIAL (la lente no está perfectamente paralela al sensor).

En la fr1 real, la esquina de la imagen se desplaza ~17 px. Ignorarlo
sesga TODA la geometría de los niveles 07+ (el repo padre pre-rectifica
cada frame antes de tocar nada).

**Calibrar = optimizar reproyección.** Un tablero de ajedrez es un objeto
de geometría PERFECTAMENTE conocida (esquinas en una rejilla plana). Con
~10-15 vistas desde ángulos distintos, `cv2.calibrateCamera` resuelve

```
min  Σ ‖ esquina_detectada − proyectar(esquina_3D; K, dist, pose_vista) ‖²
```

sobre K, los 5 coeficientes y la pose de cada vista. El **error de
reproyección** residual (en px) es LA métrica de calidad: < 0.5 px es una
calibración sana. (Es el mismo residuo del bundle adjustment del nivel 11 —
calibrar ES un BA con la estructura conocida.)

**Ojo al comparar coeficientes.** k1/k2/k3 están fuertemente correlacionados
(son un polinomio): dos juegos de números distintos pueden describir casi la
MISMA curva. Por eso la verificación honesta compara el CAMPO de distorsión
(px de desplazamiento en toda la imagen), no coeficiente a coeficiente.

## Cómo correr

```bash
pip install -r requirements.txt
python genera_tablero.py       # 14 vistas sinteticas con la distorsion de fr1
python 04_calibracion.py       # detecta esquinas + calibra + compara con GT
python verificacion.py        # el examen del nivel
```

Resultados en `salida/`: `deteccion.png` (las esquinas encontradas),
`undistort_antes_despues.png` y la comparación con la verdad en consola.

## Qué debes poder explicar al terminar

- Qué corrige cada coeficiente y en qué coordenadas actúa el modelo.
- Por qué un tablero, por qué muchas vistas y por qué inclinadas.
- Qué mide el error de reproyección y qué valor es aceptable.
- Por qué comparar k's sueltos engaña y comparar el campo no.
