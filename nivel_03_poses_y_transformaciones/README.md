# Nivel 03 — Poses y transformaciones: dominar T_w_c

**Objetivo**: mover la cámara por el mundo sin perderse — componer e
invertir poses SE(3), adoptar la notación de subíndices que te salvará de
la mitad de los bugs del curso, y exportar tu primera trayectoria en
formato TUM.

## Teoría mínima

**Una pose es una matriz 4×4.**

```
T = [ R  t ]      R ∈ SO(3): rotación (RᵀR = I, det = +1)
    [ 0  1 ]      t ∈ R³:    traslación
```

Actúa sobre puntos como `X' = R·X + t`, y COMPONER dos poses es multiplicar
sus matrices — en ese orden, porque SE(3) no conmuta (gira-y-avanza ≠
avanza-y-gira; compruébalo con la mano).

**La notación que lo ordena todo.** `T_a_b` lleva puntos del marco `b` al
marco `a`. Los subíndices se encadenan "cancelándose", como unidades:

```
T_w_c2 = T_w_c1 · T_c1_c2        (w←c1 por c1←c2 da w←c2)
X_w    = T_w_c · X_c             (la pose de la cámara lleva puntos
                                  de la cámara al mundo)
```

Si una expresión no cancela los subíndices, ESTÁ MAL — antes de correrla.
Es la convención del repo padre y de este curso: `T_w_c` = pose de la
cámara (cámara→mundo), ejes OpenCV.

**La inversa, en forma cerrada.** Despejando X de X' = R·X + t:

```
T⁻¹ = [ Rᵀ  −Rᵀ·t ]
      [ 0     1   ]
```

Más barata que `np.linalg.inv` y EXACTAMENTE rígida (Rᵀ es una rotación
perfecta; la inversa numérica genérica solo aproximadamente).

**El formato TUM.** Una trayectoria es una lista `timestamp tx ty tz qx qy
qz qw` — la traslación y la rotación como CUATERNIÓN unitario (4 números,
una restricción, sin gimbal lock, interpolable). En este nivel implementas
las dos conversiones (R→q, método de Shepperd; q→R, Rodrigues) y verificas
que cierran exacto.

**Lo que NO va aquí**: Exp/Log y el espacio tangente de SE(3). Llegan en el
nivel 12, cuando el grafo de poses los necesite — en este curso la
abstracción se introduce cuando se usa.

## Cómo correr

```bash
pip install -r requirements.txt
python 03_poses.py           # la camara vuela en circulo alrededor del cubo
python verificacion.py      # el examen del nivel
```

Resultados en `salida/`: `orbita.png` (mosaico de vistas), `orbita.avi`,
`trayectoria.txt` (formato TUM) y `trayectoria_planta.png`.

## Qué debes poder explicar al terminar

- Por qué componer es multiplicar y el orden importa.
- La gimnasia de subíndices: probar que una cadena de T's es correcta sin
  ejecutar nada.
- Por qué la inversa cerrada es mejor que la numérica.
- Qué guarda exactamente una línea del formato TUM.
