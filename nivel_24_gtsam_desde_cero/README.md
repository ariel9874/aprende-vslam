# Nivel 24 — GTSAM desde cero · bonus

**Objetivo**: el nivel 21 dejó lista la matemática del grafo de factores
(residuos, JᵀΛJ, Schur, marginalización) — lo que GTSAM *calcula*. Este
nivel construye **cómo lo calcula**, que es su verdadera aportación: la
**eliminación de variables**, el **orden** (de donde sale el fill-in que en
el 21 solo contamos), el **árbol de Bayes**, y sobre todo **iSAM**:
re-resolver solo lo que cambió. Cierra la trilogía de estimación del curso:
el batch del 21 re-resuelve TODO; el filtro del 23 tira el pasado; iSAM es
el camino de en medio.

El mundo es el del 21, duplicado (`mundo.py`, misma semilla): por eso cada
número de aquí se compara directo con la tabla de allá.

## Acto 1 — Eliminar UNA variable es resolver (`eliminacion.py`)

Junta los factores que tocan `v`, factoriza (QR) y quedan dos piezas: el
**condicional** p(v | separador) y un **factor nuevo sobre el separador**.
Repetir hasta vaciar el grafo convierte el grafo de factores en una **red de
Bayes**; sustituir hacia atrás da la solución. Verificado:

- un solve lineal por eliminación == el solve denso, a **1.5e-14**;
- el Gauss-Newton entero == el batch del nivel 21, a **7.1e-15**
  (RMSE **8.24 cm** — el 8.2 de la tabla del 21);
- el factor que la eliminación deja sobre el separador == el **complemento
  de Schur** del 21, entrada por entrada (**3.8e-16**). La marginalización
  del 21 era eliminación con otro nombre.

## Acto 2 — El orden importa: el fill-in, medido

El factor nuevo acopla TODO el separador: eliminar temprano una variable
muy conectada llena la matriz. Mismo grafo, cuatro órdenes, misma solución
(8.5e-15) — y estos no-ceros en el factor R:

| orden | nnz(R) | vs el mejor |
|---|---|---|
| **min-degree** (el COLAMD de juguete, desde cero) | **5,091** | 1× |
| landmarks primero (el Schur de BA, nivel 11) | 44,225 | 8.7× |
| temporal (el de un SLAM que avanza) | 46,803 | 9.2× |
| max-degree (el peor, a propósito) | 56,345 | 11.1× |

Dos lecciones con historia: (1) el orden "landmarks primero" que es ÓPTIMO
en BA (miles de puntos, pocas cámaras) aquí es de los malos — la heurística
no viaja entre problemas, min-degree lo descubre solo; (2) el orden temporal
paga **los bucles**: los landmarks de la vuelta 1 re-observados en la
vuelta 2 acoplan las dos vueltas enteras. Ese es el fill-in que el nivel 21
midió (1.33×) sin explicarlo — ahora explicado y controlado.

## Acto 3 — El árbol de Bayes (`arbol_de_bayes.py`)

Los condicionales, agrupados en cliques (construcción de Kaess et al.),
forman un **árbol**: resolver = bajar de la raíz (idéntico al acto 1,
verificado a 0.0). Y la revelación estructural: **un factor nuevo solo
invalida el camino de su clique a la raíz** — el resto del árbol ni se
entera. Medido en los tres árboles del mundo:

| árbol | cliques | prof | odo nueva toca | bucle x5–x104 toca |
|---|---|---|---|---|
| cadena de odometría | 104 | 103 | **1** | **99 (95%)** |
| completo, orden temporal | 20 (gordos: hasta 78 vars) | 19 | 1 | 15 (75%) |
| completo, min-degree | 103 | 9 | 7 | **9 (9%)** |

El costo del loop closure ES la forma del árbol; y un buen orden (acto 2)
acorta todos los caminos. Por eso GTSAM ordena con COLAMD y iSAM2 re-ordena
lo que actualiza.

## Acto 4 — iSAM de juguete (`isam_de_juguete.py`) — el acto estrella

El circuito del 21 EN LÍNEA, pose a pose: batch re-resolviendo todo en cada
paso (el estándar de oro, caro) contra nuestro incremental. Tres mecanismos,
cada uno con su contraparte real de iSAM2: **factores cacheados** (re-eliminar
un sufijo sin tocar el prefijo), **re-linealización por umbral** (el
`relinearizeThreshold`/wildfire) y **reorden al vuelo** de landmarks
re-observados (iSAM1 re-ordenaba cada 100 pasos; iSAM2, el sub-árbol
afectado). Medido:

- misma respuesta: final == batch a **0.12 cm** (online a 0.35 cm);
- el paso típico re-elimina **8 de 119** variables; el **cierre de bucle**
  dispara un pico de **98** (12× la mediana) — y el costo vuelve a caer:
  cada vuelta paga su cierre UNA vez;
- speedup **12.9×** con 2 vueltas, **18×** con 3 — crece con el viaje,
  porque el batch paga el grafo entero y iSAM solo el camino afectado.

La tabla que cierra la trilogía (mismas medidas; online = cada pose al
emitirse, final = la trayectoria refinada):

| estimador | online | final | tiempo |
|---|---|---|---|
| FILTRO EKF (nivel 23) | 9.4 cm | no hay | 0.02 s |
| **iSAM de juguete (este nivel)** | 9.9 cm | **8.2 cm** | 0.25 s |
| batch por paso (nivel 21 en línea) | 10.0 cm | 8.2 cm | 3.22 s |

La sorpresa honesta: **online los tres casi empatan** — nadie reescribe la
pose que ya emitió. La diferencia es lo que ADEMÁS te llevas: el filtro
nada, el batch todo (re-pagándolo entero cada paso), iSAM todo a precio
incremental. Eso es iSAM2, y es por qué existe.

## Acto 5 — GTSAM de verdad (`con_gtsam_real.py`, Docker)

El mismo mundo con la API real, dentro del contenedor (no hay wheel de
gtsam para Windows/Python 3.13 — verificado 2026-07; el contenedor es la
adaptación honesta, patrón del nivel 20, con `numpy<2` por el ABI de la
wheel). El diccionario de traducción:

| nuestro concepto | la clase de GTSAM |
|---|---|
| lista de factores no lineales | `NonlinearFactorGraph` |
| dict clave → valor | `Values` |
| clave `('x', i)` / `('l', j)` | `symbol_shorthand.X(i)` / `L(j)` |
| factor de odometría (between) | `BetweenFactorPose2` |
| factor de observación | `BearingRangeFactor2D` (†) |
| Λ = diag(1/σ²) | `noiseModel.Diagonal.Sigmas` |
| prior del gauge (1e8) | `PriorFactorPose2` (σ = 1e-4) |
| `gauss_newton` (batch) | `LevenbergMarquardtOptimizer` |
| `ISAMJuguete.paso()` | `ISAM2.update()` |
| umbral de re-linealización | `ISAM2Params.relinearizeThreshold` |
| marginalizar / covarianzas | `Marginals` (ejercicio 4) |

(†) La adaptación honesta de la medida: nuestro factor es el landmark en el
marco del robot (dx, dy); el estándar 2D de GTSAM es bearing+range. Misma
información en polares (σ_bearing = σ_obs/rango), por eso el acuerdo
esperado — y verificado — es de **milímetros de ATE**, no de épsilon.

Medido en el contenedor: GTSAM LM **8.21 cm** vs nuestro batch **8.24 cm**
(**0.4 mm** de diferencia); y el ISAM2 real online da **10.01 cm** — casi
calca a nuestro juguete (9.94). Bonus de humildad: GTSAM resuelve en 0.02 s
lo que nuestro numpy tarda ~1 s — la eliminación es la misma; el C++, el
ordering y veinte años de ingeniería hacen el resto.

## Cómo correr

```bash
pip install -r requirements.txt      # numpy + matplotlib (gtsam NO: acto 5)
python 24_gtsam.py                   # actos 1-4: tablas + graficas (segundos)
python verificacion.py               # el examen (13 checks, sin gtsam)

# acto 5 (opcional, requiere Docker):
docker compose up --build            # GTSAM real vs nuestro numero
python verificacion.py --docker      # el examen lo valida solo
```

Sin dataset, sin GPU: numpy puro (y un contenedor opcional).

## Qué debes poder explicar al terminar

- Por qué eliminar una variable = un condicional + un factor sobre el
  separador, y por qué ese factor ES el complemento de Schur.
- De dónde sale el fill-in, por qué el orden no cambia la solución, y por
  qué "landmarks primero" es óptimo en BA y malo aquí.
- Qué es el árbol de Bayes y por qué un factor nuevo solo invalida el
  camino de su clique a la raíz — con los números de la tabla.
- Los tres mecanismos de iSAM (caches, umbral, reorden) y qué compra cada
  uno; por qué el cierre de bucle es un pico y no un costo permanente.
- La trilogía filtro/iSAM/batch: qué conserva, qué cuesta y qué te llevas
  con cada uno — y por qué VIO/SLAM modernos (GTSAM en el corazón de
  muchos) viven en el punto medio.

## La lectura del nivel

- Kaess, Johannsson, Roberts, Ila, Leonard, Dellaert: *iSAM2: Incremental
  Smoothing and Mapping Using the Bayes Tree* (IJRR 2012).
- Dellaert & Kaess: *Factor Graphs for Robot Perception* (2017) — el libro
  corto que cuenta este nivel entero, de los autores de GTSAM.

Lo que iSAM2 real hace más fino que nuestro juguete (y dónde leerlo): la
re-linealización fluida dentro del árbol y los deltas parciales (nuestro
sufijo temporal es la versión de una dimensión del sub-árbol afectado).
