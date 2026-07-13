# Ejercicios — Nivel 10

## 1. ¿Cuánto vale cada keyframe? (fácil)

Barre `KF_MAX_GAP` en {5, 10, 15, 30, 999} y tabula: número de keyframes,
tamaño del mapa y ATE.

**Objetivo**: la curva en U. Con muchos keyframes el mapa crece pero se
llena de puntos con poco paralaje (nivel 09); con muy pocos, el tracking se
queda sin puntos que ver y entra en COAST. Encuentra el óptimo de ESTA
secuencia y explica qué patología aparece en cada extremo.

## 2. El gauge es arbitrario (fácil, y conceptualmente clave)

Cambia el gauge de la inicialización: en vez de profundidad mediana = 1.0,
usa 10.0 (o 0.1).

**Objetivo**: verifica que el ATE **no cambia** (el error se mide tras
alineación de similitud — Umeyama, que absorbe la escala) pero el mapa PLY
sale 10× más grande. La escala monocular no es una propiedad del mundo: es
una convención tuya. Recuérdalo en el nivel 15, cuando un sensor de
profundidad la vuelva una MEDICIÓN y ya no se pueda negociar.

## 3. La lección de los 584 puntos basura (medio — reprodúcela)

Baja `KF_MIN_INLIERS` de 45 a 5 (es decir, deja que se creen keyframes desde
poses malas) y añade ruido a la secuencia: perturba cada imagen con
`cv2.GaussianBlur` fuerte cada 10 frames (simula motion blur).

**Objetivo**: mide el tamaño del mapa y el ATE en ambos casos. Con el piso
de salud bajo, el mapa crece MÁS (más keyframes) y el ATE EMPEORA — puntos
nacidos de poses malas, que luego el PnP usa como si fueran verdad. Es la
lección exacta del repo padre: *nunca crear mapa desde una pose incierta*.
Un mapa grande no es un mapa bueno.

## 4. El mapa nunca olvida (medio — el problema que resuelve el nivel 13)

Imprime, en cada frame, cuántos de los matches contra el mapa son con puntos
creados en los últimos 2 keyframes vs puntos viejos.

**Objetivo**: verás que casi todos los inliers vienen de puntos RECIENTES:
el mapa entero se matchea por fuerza bruta contra cada frame, pero los
puntos lejanos ya no son visibles. Estás pagando un coste O(tamaño del mapa)
por frame para nada. La solución (mapa LOCAL por covisibilidad) es el nivel
13 — y en el repo padre esa misma optimización resultó ser no solo velocidad
sino CORRECCIÓN (con un mapa de 10k puntos, el matching global daba 0
inliers geométricos: su lección 22).

## 5. PnP con un solo punto de más (difícil, teórico + experimental)

El PnP necesita mínimo 3 puntos (P3P, con hasta 4 soluciones) o 4 para ser
único. Verifícalo: coge 3, 4, 6 y 20 correspondencias correctas (sin ruido)
y resuelve con `cv2.solvePnP` usando `SOLVEPNP_P3P` / `SOLVEPNP_ITERATIVE`.

**Objetivo**: documenta cuántas soluciones válidas hay en cada caso y qué
pasa al añadir 0.5 px de ruido a las 3 correspondencias del P3P (pista: sin
redundancia, no hay forma de detectar el error — el mínimo cuadrático de la
redundancia no es un lujo, es la única defensa).
