# Ejercicios — Nivel 24

## 1. La perilla real de iSAM2 (fácil)

`UMBRAL_RELIN` en `isam_de_juguete.py` es 0.05. Bárrelo (0.01, 0.05, 0.2,
1.0, ∞) y grafica dos curvas contra el umbral: la dif máxima vs el batch por
paso, y el total de variables re-eliminadas.

**Objetivo**: la curva precisión-vs-costo de iSAM2 con tus propias manos.
Con umbral ∞ (nunca re-linealizar) deberías ver al juguete degradarse hacia
un "filtro con memoria" (linealizaciones selladas — ¿te suena del 23?); con
umbral 0 deberías recuperar el batch, pagándolo casi entero. ¿Dónde está el
codo? El default de GTSAM es 0.1 — ¿le darías la razón en este mundo?

## 2. COLAMD de verdad-de-juguete (medio)

Nuestro `orden_min_degree` opera sobre el grafo de VARIABLES (bloques).
COLAMD opera sobre COLUMNAS de la matriz A escalar. Implementa la variante
por columnas (cada variable aporta `dim` columnas; el grado se cuenta en
filas compartidas) y mide nnz(R) contra los cuatro órdenes del acto 2.

**Objetivo**: comprobar si el refinamiento paga en este mundo (spoiler: la
ganancia grande ya la dio min-degree; la de COLAMD es marginal aquí — y esa
TAMBIÉN es una lección sobre heurísticas).

## 3. Marginalizar en el árbol (medio)

La ventana del 21 marginalizaba con Schur denso sobre H entera. Hazlo bien:
en el árbol de Bayes del acto 3, elimina las poses más viejas que un lag
SIN tocar la raíz — solo los cliques donde viven (sus condicionales ya son
p(viejo | resto): tirarlos con cuidado ES marginalizar).

**Objetivo**: la ventana deslizante del 21, ahora a costo de camino y no de
matriz entera. Verifica contra `demo_schur` que la información sobre la
frontera es la misma.

## 4. Los `Marginals` (difícil)

GTSAM te da la covarianza de UNA pose con `Marginals(graph, result)`. Hazlo
desde nuestro árbol: la covarianza del clique raíz es (RᵀR)⁻¹ de sus
frontales; bajando, cada clique combina la de su padre con su condicional.
Implementa la recursión y compara la covarianza de x50 contra invertir la H
entera (a 1e-9).

**Objetivo**: covarianzas sin invertir la información completa — la razón
por la que los sistemas reales pueden reportar incertidumbre en vivo.

## 5. El factor de IMU, a GTSAM real (difícil)

El nivel 22 construyó la preintegración desde cero. Dentro del contenedor
del acto 5, reproduce su experimento del apagón visual con la API real:
`PreintegratedImuMeasurements` + `ImuFactor` + `ISAM2` (necesitarás pasar a
Pose3/NavState: el mundo del 22 ya es 3D).

**Objetivo**: el diccionario de traducción del README extendido al VIO — y
comprobar que el `ImuFactor` de GTSAM reproduce el rescate del apagón que
mediste en el 22 (62.2 → 4.8 cm). Es el ejercicio más largo del curso
bonus; también es, casi línea por línea, el esqueleto de un VIO comercial.
