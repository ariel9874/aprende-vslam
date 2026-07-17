# Ejercicios — Nivel 21

## 1. El lag de la ventana (fácil, la curva del nivel)

`optimizar_ventana` corta en N/2. Barre el corte (20%, 50%, 80% del
recorrido) y grafica RMSE de la ventana vs lag, con y sin marginalizar.

**Objetivo**: dos curvas que se separan. La cortada empeora rápido al
acortar (cada pose olvidada es información tirada); la marginalizada aguanta
mucho más… pero no llega al completo ni con lag enorme — ¿por qué? (El punto
de linealización congelado: no mejora por recordar más si lo recordado quedó
linealizado lejos del óptimo.)

## 2. FEJ casero (medio — el fix de los VIO reales)

La brecha ventana-vs-completo (25.9 vs 8.2 cm) viene de marginalizar
linealizado en la odometría INICIAL. Mejora el prior: antes de marginalizar,
corre unas iteraciones de Gauss-Newton SOLO sobre el pasado (o sobre todo el
grafo) y marginaliza en ese punto mejor.

**Objetivo**: mide cuánto se cierra la brecha. Y la reflexión de diseño: un
sistema en línea no puede optimizar el pasado antes de marginalizarlo (por
eso existe FEJ: mantener jacobianos consistentes aunque el punto no sea el
óptimo). ¿Qué pudiste hacer tú que un VIO real no puede?

## 3. La retícula perdida del grafo de poses (medio)

El grafo de poses dio 29 cm con 8 bucles. Sube `CADENCIA` a 1 (un factor de
bucle por cada pose que re-observa) y baja `GAP_BUCLE`.

**Objetivo**: la curva RMSE vs número de factores relativos. Con suficientes
factores te acercas al completo — estás re-densificando a mano la retícula
que la compresión tiró. El costo también crece: mide ambos. ¿En qué punto el
"grafo de poses denso" deja de ser más barato que el completo?

## 4. Torturar al filtro (medio — su talón de verdad)

El EKF empató casi al smoother (9.4 vs 8.2 cm). Multiplica `SIGMA_ODO_TH`
por 5 (rotación ruidosa: la no-linealidad muerde) y regenera el mundo.

**Objetivo**: la brecha filtro-vs-completo se abre (el smoother
re-linealiza al converger; el filtro selló cada linealización al llegar).
Grafica ambas trayectorias: verás al EKF describir el circuito torcido sin
poder arrepentirse. Bonus: mide la CONSISTENCIA (el error respecto a la
covarianza P que el filtro cree tener — el NEES de la literatura).

## 5. Un factor nuevo: el GPS de juguete (fácil, estructural)

Añade al grafo completo un factor UNARIO de posición absoluta (x, y medidos
con σ = 20 cm) cada 15 poses.

**Objetivo**: tres líneas de código (residuo p[:2] − z, jacobiano [I₂ 0]) y
una pregunta estructural: ¿sigue haciendo falta el prior de gauge en la
pose 0? ¿Y los bucles? Mide el RMSE con GPS y sin bucles. La gracia de los
grafos de factores es exactamente esta: un sensor nuevo = un factor nuevo,
y NADA más cambia.
