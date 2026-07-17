# Ejercicios — Nivel 16

## 0. La compuerta de movimiento (medio — la pareja de la reloc)

Este nivel metió la RELOCALIZACIÓN al código enseñado. El padre la empareja
con una COMPUERTA: si el paso de traslación de un frame supera 6× el p95 de
los pasos recientes aceptados (con ≥ 20 muestras), la pose se RECHAZA y se
cae a coast — cuyo contador dispara la reloc. Impleméntala en `_track`.

**Objetivo**: mide el ATE con y sin compuerta en V1_01. Y la lección de
diseño (la 19 del padre, aprendida por las malas): la compuerta SOLA es
dañina — bloquea recuperaciones legítimas además de teletransportes (el
padre midió 8.4 → 37.7 cm) y puede auto-congelarse tras una pausa (→202 cm).
La cura no fue ajustar el umbral: fue darle una SALIDA (la reloc decide con
geometría global a qué pose volver). Rechazar sin salida es quedarse ciego;
rechazar con reloc es un filtro. Ojo extra en este dataset: el arranque
estático hace p95 ≈ 0 — ¿qué le pasa a tu compuerta en el despegue?

## 1. El brazo de palanca, medido (fácil, muy instructivo)

`leer_gt_euroc` corrige el GT del CUERPO (IMU) a la CÁMARA con el extrínseco
T_BS (~7 cm de brazo). Evalúa la misma trayectoria dos veces: con la
corrección y sin ella (usa `p_cuerpo` directamente).

**Objetivo**: mide cuánto ATE cuesta ignorar el brazo. Pista de por qué no lo
absorbe la alineación: el error del brazo ROTA con la pose del dron — no es
un offset constante que el Umeyama pueda tragarse. En un dataset de mano
(TUM) el efecto es menor; en un dron 6-DoF que gira sobre sí mismo, no.

## 2. Las perillas del SGBM (medio)

`CargadorEstereo` usa `numDisparities=96, blockSize=7`. Corre un tramo de
V1_01 con (48, 7), (96, 3) y (96, 15), midiendo: % de píxeles con z válida,
ATE final y fps.

**Objetivo**: entiende el trade-off triple. `numDisparities` fija la z MÍNIMA
visible (z_min = bf/d_max: con 48, todo lo más cercano que bf/48 desaparece);
`blockSize` cambia densidad vs detalle (bloques grandes alisan discontinuidades
de profundidad — el mismo motivo por el que la profundidad se remapea con
NEAREST). Y todo cuesta tiempo de CPU por frame.

## 3. La z máxima de confianza (medio)

El tracker corta en `DEPTH_MAX = 8.0` m. Derívalo en vez de creerlo: el error
de profundidad por un error de ±0.5 px de disparidad es Δz ≈ z²·Δd/bf
(∂z/∂d = −bf/d²). Con bf ≈ 48, tabula Δz para z = 2, 5, 8, 15 m.

**Objetivo**: la tabla te dirá en qué z el error supera lo que el BA puede
tratar como "ruido de píxel". Sube DEPTH_MAX a 20 y mide el ATE: ¿confirma tu
tabla? Conecta con la simetría de la lección 37: el residuo u_R ya pesa menos
lo lejano — ¿entonces por qué ayuda ADEMÁS cortar?

## 4. Estéreo sin rectificar (difícil, conceptual)

Desactiva la rectificación: matchea ORB entre izquierda y derecha CRUDAS,
estima la disparidad por diferencia de u (ignorando que las epipolares no
son filas) y alimenta esas z al tracker.

**Objetivo**: mide cuánto se degrada todo (densidad de z, ATE, perdidos).
La rectificación no es un detalle de implementación: es lo que convierte una
búsqueda 2D con geometría por-par en una resta de columnas. Explica dónde se
va la z falsa cuando la v de ambas cámaras no coincide.

## 5. El mismo tracker, tres sensores (medio — el cierre del bloque)

Tienes el MISMO tracker corriendo con: gauge monocular (nivel 14), Kinect
(nivel 15) y estéreo (nivel 16). Haz la tabla final: ATE rígido (o de
similitud en monocular), escala de similitud, frames perdidos y de dónde
sale z en cada caso.

**Objetivo**: explica con la tabla qué compró cada sensor. La escala de
similitud es la columna reveladora: ~libre (monocular), ~1.0 (Kinect, sensor
activo con rango corto), ~1.0 (estéreo, geometría pura del rig). Y el bf que
en el nivel 15 era una convención (40.0) aquí lo firmó la calibración
(~47.9): busca en `dataset.py` la línea exacta donde nace.
