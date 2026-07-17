# Nivel 21 — Grafos de factores · bonus

**Objetivo**: entender que casi todo lo que optimizaste en el curso era un
**grafo de factores** — y que las grandes familias de backends (BA completo,
grafo de poses, ventana deslizante, filtro) difieren en UNA sola decisión:
**qué variables conservas y qué haces con las que sueltas**. Cuatro
backends, cada uno construido desde cero, sobre las MISMAS medidas.

## El marco: estimación MAP

Variables Θ (poses, landmarks); cada medida es un FACTOR gaussiano
p(z|Θ) ∝ exp(−½‖e(Θ)‖²_Λ). Maximizar el producto = minimizar la suma de
Mahalanobis: Gauss-Newton sobre `H·δ = −g` con **H = ΣJᵀΛJ, la matriz de
información** — dispersa porque cada factor toca 1-2 variables. El BA del
nivel 11 y el grafo de poses del 12 eran exactamente esto; ahora la forma
del grafo es la protagonista.

El mundo (común a los cuatro, `mundo.py`): un robot SE(2) da dos vueltas a
un circuito con 14 landmarks; odometría ruidosa, observaciones con alcance,
todo sembrado. 2D a propósito: los jacobianos caben en tres líneas y las
matrices de información se DIBUJAN.

## Los cuatro, y lo que midió cada uno

| backend | conserva | RMSE | costo |
|---|---|---|---|
| odometría pura | nada | 56.0 cm | — |
| **grafo COMPLETO** | todo (poses+landmarks) | **8.2 cm** | 0.16 s |
| grafo de POSES | solo poses; landmarks → 8 factores de bucle | 29.0 cm | 0.03 s |
| VENTANA marginalizada | la mitad reciente + un prior de Schur | 25.9 cm | 0.19 s |
| ventana CORTADA | la mitad reciente, sin prior | 59.1 cm | 0.07 s |
| FILTRO EKF (online) | solo el presente + landmarks | 9.4 cm | 0.02 s |

Las lecciones, una por fila:

1. **El completo gana siempre** — conserva TODA la información. Es también
   el que crece sin límite: por eso existe el resto de la tabla.
2. **El grafo de poses paga la compresión** (29 vs 8.2): al convertir los
   landmarks en unos POCOS factores relativos, la retícula de covisibilidad
   (dos poses cualesquiera acopladas por un landmark común) se tira. Barato
   y suficiente para redistribuir un bucle en caliente (nivel 13) — no para
   el refinado final (por eso el GBA del nivel 14 existe).
3. **Marginalizar ≠ cortar** (25.9 vs 59.1 — 2.3×): el complemento de Schur
   comprime el pasado en un prior gaussiano EXACTO sobre la frontera (el
   examen lo verifica a 10⁻¹¹ contra resolver todo). Sus dos precios,
   medidos: el **fill-in** (el prior es denso: 1.33× más entradas — y crece
   con el lag) y el **punto de linealización congelado** (el prior quedó
   evaluado en la odometría inicial: es la brecha 25.9 vs 8.2, y es el
   problema que FEJ gestiona en OKVIS/MSCKF).
4. **El filtro es marginalización al límite**: cada pose se marginaliza al
   llegar la siguiente. Su trayectoria es online POR CONSTRUCCIÓN (cada
   pose se emite y se sella — la lección 25 del curso, ahora estructural).
   Y la sorpresa honesta: en un mundo amable casi empata al smoother
   (9.4 vs 8.2) — su talón es la CONSISTENCIA cuando la no-linealidad
   crece, y torturarlo es el ejercicio 4.

## El dibujo que lo explica todo

`salida/grafos.png` (del driver): la H del grafo completo es una FLECHA
dispersa (poses encadenadas | landmarks diagonales, acoplados solo vía
poses); el prior marginalizado es un bloque DENSO. Ese par de dibujos es la
razón de existir de GTSAM/g2o (eliminación explotando el patrón) y de iSAM2
(re-eliminar solo lo que el factor nuevo toca — nivel 18, ejercicio 4).

## Cómo correr

```bash
pip install -r requirements.txt      # numpy + matplotlib
python 21_grafos.py                  # la tabla + las graficas (segundos)
python verificacion.py               # el examen (10 checks, segundos)
```

Sin dataset, sin GPU: numpy puro.

## Qué debes poder explicar al terminar

- Por qué H es dispersa y qué estructura tiene (¿qué acopla un landmark?).
- Qué conserva y qué tira cada forma del grafo — con los números de la tabla.
- Por qué marginalizar es exacto en el sistema lineal y aun así la ventana
  no alcanza al completo (los dos precios).
- En qué sentido un filtro es un caso extremo de marginalización, y por qué
  su trayectoria es online por construcción.
- Dónde encaja cada forma en un SLAM real (bucle en caliente → grafo de
  poses; refinado → completo offline; VIO embebido → ventana; recursos
  mínimos → filtro).
