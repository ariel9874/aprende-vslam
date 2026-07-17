# Ejercicios — Nivel 15

El primero es el ejercicio estrella: deshace a propósito la decisión central
del nivel y mide el desastre.

## 1. Strasdat al revés: el bucle Sim(3) en un mapa métrico (estrella)

En `slam.py`, `_cerrar_bucle` usa `GrafoDePoses("se3")` con información 6×6.
Cámbialo a `"sim3"` (información 7×7, `np.eye(7)`) y corre sobre **fr2_xyz
con profundidad** (el dataset del nivel 14 ya trae `depth/`; pásalo con
`--root`). Esa secuencia cierra ~20 bucles: el escenario perfecto.

**Objetivo**: mide ATE rígido y escala de similitud con cada grupo, y espía
la escala relativa que mide cada bucle (imprime `s_rel =
det(R_del_bucle)^(1/3)` o el factor de Umeyama). Lo que el padre midió: los
bucles Sim(3) empiezan en s_rel ≈ 1 y DEGENERAN hasta 0.03 — cada corrección
re-escala el mapa viejo, los puntos nuevos siguen naciendo métricos, y el
siguiente bucle "corrige" la discrepancia que el anterior creó (ATE 22.1 cm,
escala 2.09; en SE(3): 4.7 cm, 1.036). En el nivel 12 mediste que Sim(3) era
LA cura; aquí es EL veneno. Mismo grafo, distinto dueño de la escala.

## 2. Los agujeros del sensor (fácil)

Toma 10 mapas de profundidad de fr1_desk y visualízalos (`plt.imshow` con
`0 = sin dato` en negro). Calcula el % de píxeles sin dato por frame.

**Objetivo**: nombra los tres tipos de agujero que ves (sombras del proyector
IR en los bordes de objetos, superficies especulares/negras, fuera de rango)
y explica por qué `DEPTH_MIN < z < DEPTH_MAX` no basta como único filtro de
calidad. Conecta: ¿por qué el residuo u_R = NaN degrada con elegancia a
residuo 2D en vez de descartar la observación?

## 3. ¿Cuánto te regala el alineador? (fácil, muy instructivo)

Evalúa la MISMA trayectoria final de keyframes dos veces: con alineación
rígida (`with_scale=False`, la honesta en métrico) y de similitud
(`with_scale=True`, la única posible en monocular).

**Objetivo**: la diferencia entre ambos ATE es lo que el alineador te estaba
"regalando". Medido aquí: la escala de similitud sale 1.012, así que el
regalo es pequeño (~2%) — PORQUE el mapa ya es métrico. Repite sobre la
trayectoria del nivel 14 (monocular): allí la escala de similitud es lo único
que hace comparable la trayectoria, y fijarla a 1 daría un ATE absurdo. La
métrica que eliges es parte del experimento.

## 4. Relocalización RGB-D (difícil — el hueco que queda)

Los 203 frames perdidos son episodios de blur SIN salida: `_coast` aplica
velocidad constante para siempre. Implementa la reloc del ejercicio 2 del
nivel 13: tras N frames en LOST, matchea contra TODOS los keyframes (sin
filtro temporal) y haz PnP; si hay ≥ 40 inliers, salta a esa pose y vuelve a
TRACK.

**Objetivo**: mide los perdidos antes/después (el padre, con reloc: **0** en
esta secuencia). Ojo a la trampa del nivel 14: sin verificación geométrica
seria, re-engancharse a una pose equivocada es peor que seguir perdido.

## 5. fr2_xyz métrico, y un matiz honesto (medio)

Corre este nivel sobre fr2_xyz con profundidad (`--root` al dataset del
nivel 14; tardará ~30-40 min) y compara con el 1.4 cm del nivel 14 monocular.

**Objetivo**: dos cosas. (a) La escala de similitud debe salir ≈ 1 SIN gauge
— metros de verdad donde el nivel 14 necesitaba la convención mediana=1. (b)
Es probable que el ATE métrico salga PEOR que el 1.4 cm monocular+GBA (el
padre midió 4.7 cm métrico ahí, y 1.1 cm apagando los bucles): con deriva tan
pequeña, cada bucle corrige menos de lo que ensucia. Un número mejor no
siempre significa un sistema mejor — pregúntate qué compraste: el monocular
te da forma; el métrico te da METROS.
