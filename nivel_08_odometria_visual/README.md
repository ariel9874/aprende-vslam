# Nivel 08 — Odometría visual monocular

**Objetivo**: construir tu primer SISTEMA completo — una odometría visual
que encadena la pose de la cámara frame a frame — y MEDIR su deriva contra
la trayectoria real. Este nivel junta todo lo anterior: características
(05), matching (06) y geometría epipolar (07), más una pieza nueva: la
evaluación honesta (ATE).

## El pipeline

```
imagenes ─▶ ORB ─▶ matching (ratio test) ─▶ matriz esencial (RANSAC)
         ─▶ pose relativa (R, t) ─▶ composicion de trayectoria ─▶ ATE vs GT
```

## Teoría mínima

**Componer la trayectoria.** La geometría epipolar (nivel 07) te da el
movimiento entre DOS frames: `T_curr<-prev`. Para saber dónde está la cámara
en el mundo se encadena:

```
T_w<-curr = T_w<-prev · (T_curr<-prev)^-1
```

**Las tres limitaciones deliberadas** (son la lección, no un descuido):

1. **ESCALA**: una cámara sola no mide magnitudes — `t` sale con ||t|| = 1
   por convención. La trayectoria tiene la FORMA correcta solo si la
   velocidad real es ~constante. (Se resuelve triangulando un mapa — nivel
   09/10 — o con estéreo/RGB-D — niveles 15/16.)
2. **DERIVA**: cada pose se apoya en la anterior; el error se ACUMULA sin
   límite porque nada lo corrige (eso llega con el BA, nivel 11, y los
   bucles, nivel 12). En este nivel la vas a VER crecer.
3. **2D-2D siempre**: re-estimamos la geometría desde cero en cada par de
   frames, tirando todo lo aprendido. El nivel 10 arregla esto con un mapa.

**Evaluar con ATE.** La estimación vive en OTRO origen, rotación y escala
que el ground truth (todo eso es inobservable para una cámara sola). Antes
de comparar hay que alinear con la similitud óptima — el método de Umeyama
(la matemática está en `evaluacion.py`) — y el ATE es el RMSE de las
distancias punto a punto tras alinear: mide la consistencia GLOBAL (la
deriva), no el error local por paso.

## Cómo correr

```bash
pip install -r requirements.txt
python genera_datos.py         # secuencia sintetica con GT exacto (~10 s)
python 08_odometria_visual.py  # corre la VO y reporta el ATE
python verificacion.py         # el examen del nivel
```

Resultados en `salida/`: `trayectoria.txt` (formato TUM) y
`trayectoria.png` (vista cenital estimada-vs-real).

## El número esperado

**ATE ~= 13 cm** sobre la secuencia sintética de 80 frames. No es un número
inventado: es la medición real de la v0.1 del repo padre con este mismo
pipeline sobre esta misma escena. Si te da entre 8 y 20 cm vas bien (RANSAC
es aleatorio). En el nivel 10, el MISMO dataset con tracking PnP contra un
mapa baja a ~7 cm — y esa mejora la vas a medir tú.

## Por qué datos sintéticos

La secuencia se GENERA con geometría exacta (planos texturizados renderizados
con la homografía real de cada pose), así que el ground truth es perfecto y
gratis. Cuando algo falle, sabrás que es tu código y no los datos. Los datos
reales — con blur, exposición y calibración imperfecta — llegan en el nivel
14, cuando el sistema ya esté completo.

## Qué debes poder explicar al terminar

- Por qué ||t|| = 1 y qué significa "hasta escala".
- Cómo se compone `T_w_c` y por qué el error se acumula.
- Qué alinea Umeyama y por qué el ATE sin alinear no tiene sentido en
  monocular.
- Qué hace el sistema cuando el tracking falla (coasting por velocidad
  constante).
