# Nivel 12 — Grafo de poses y cierre de bucle

**Objetivo**: deshacer la deriva. Hasta ahora el error sólo crecía (nivel 08)
o se contenía (niveles 10-11). Aquí, cuando el sistema **reconoce un sitio
por el que ya pasó**, esa única restricción se propaga hacia atrás y corrige
la trayectoria entera.

Y aquí llega por fin el álgebra de Lie: **Exp/Log** de SE(3) y Sim(3). No
antes — en este curso la abstracción entra cuando hace falta.

## Teoría mínima

**El grafo de poses.** El bundle adjustment (nivel 11) optimiza poses Y
puntos. El grafo de poses optimiza **sólo poses**, usando como medidas las
transformaciones relativas que el tracking ya estimó:

```
argmin_{T_i}   Σ_(i,j)   ‖ Log( T̂_ij⁻¹ · T_i⁻¹ · T_j ) ‖²_Λ
```

Cada arista dice: *"medí que del nodo i al j hay T̂_ij"*. El error es lo
MEDIDO contra lo que las poses actuales IMPLICAN — y se expresa en el espacio
tangente con `Log`: si coinciden, el argumento es la identidad y `Log(I) = 0`.

Es mucho más barato que el BA (no hay puntos), y es lo que se corre al cerrar
un bucle.

**El bucle.** Una arista más — pero entre dos nodos LEJANOS en el tiempo. Esa
sola restricción reparte el error acumulado por toda la cadena: el nodo 29 ya
no puede estar a 1 metro de donde debería, porque el bucle lo ata al nodo 0.

## El experimento del nivel: Strasdat (RSS 2010)

La odometría **monocular** no sólo deriva en posición: deriva en **escala**
(el gauge del nivel 10 se va corrompiendo). ¿Qué pasa si intentas cerrar un
bucle con un grafo SE(3), cuyos nodos son rígidos?

| | ATE |
|---|---|
| odometría (1% de deriva de escala por paso) | 0.70 m |
| grafo **SE(3)** | **0.80 m** ← ¡EMPEORA! |
| grafo **Sim(3)** | **0.00 m** |

El SE(3) **empeora la trayectoria**. No es un bug: le estás pidiendo lo
imposible. Sus nodos no pueden re-escalarse, así que la única manera que tiene
de cerrar el bucle es **deformar la geometría** — mover traslaciones que
estaban bien para absorber un error que es de escala. Reparte la mentira en
vez de corregirla.

Sim(3) tiene el séptimo grado de libertad exactamente para esto. Por eso el
SLAM monocular usa grafos Sim(3).

> **Adelanto del nivel 15**: en RGB-D la escala es una MEDICIÓN, no un gauge,
> y el bucle vuelve a ser SE(3). Usar Sim(3) allí es el error simétrico, y el
> repo padre lo pagó caro (cada bucle re-escalaba el mapa métrico: 22 cm de
> ATE, escala 2.09). **El grupo correcto depende de quién fija la escala.**

## La lección incómoda sobre Huber

El nivel también mide qué pasa con un **falso positivo** de bucle (el sistema
cree reconocer un sitio y se equivoca — dos pasillos idénticos):

| | ATE |
|---|---|
| sin robustez (cuadrático) | 3.54 m |
| Huber δ=1.0 | 3.27 m ← apenas ayuda |
| Huber δ=0.01 | 0.22 m |
| **rechazado antes de entrar al grafo** | **0.15 m** |

Huber con un umbral razonable **no salva al grafo**: degrada el outlier a un
empuje *lineal constante* (`w·‖r‖ = δ`), que sigue compitiendo con los 29
factores buenos. Bajar mucho δ funciona... pero también amansa el bucle
LEGÍTIMO. La defensa real es **verificar geométricamente el bucle antes de
creérselo** (matching + PnP + contar inliers). Huber es la segunda línea, no
la primera. Lo verás hecho en el nivel 13.

## Cómo correr

```bash
pip install -r requirements.txt
python 12_grafo_de_poses.py     # los 4 experimentos
python verificacion.py          # el examen del nivel
```

Sin dataset: todo es geometría simulada con verdad exacta.

## Qué debes poder explicar al terminar

- Por qué se optimiza en el tangente y qué es `T ← T·Exp(δ)`.
- Qué mide el residuo de una arista y por qué lleva `Log`.
- Por qué SE(3) no puede repartir deriva de escala y Sim(3) sí.
- Por qué el grupo correcto depende de quién fija la escala.
- Por qué Huber no basta contra un falso positivo de bucle.
