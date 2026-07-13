# Ejercicios — Nivel 13

Los dos primeros son **los ejercicios estrella del curso**: son las piezas
que este SLAM deliberadamente NO tiene, y construirlas te enseña por qué
existen.

## 1. COVISIBILIDAD (el ejercicio estrella)

El mapa local es por RECENCIA: los últimos 5 keyframes. Sustitúyelo por
recencia ∪ **covisibilidad**: además de los recientes, incluye los keyframes
que comparten ≥ 15 puntos con el keyframe actual (usa `mapa.obs`).

**Objetivo**: mide el ATE antes y después. Y entiende POR QUÉ mejora: con
recencia sola, al re-visitar el corredor de vuelta, el mapa local **excluye
los puntos originales de la ida** — así que el sistema triangula puntos
DUPLICADOS, desplazados por la deriva. Acaban existiendo dos versiones
coherentes del mismo sitio, y el PnP salta entre ellas (biestabilidad).

La covisibilidad lo hace imposible: si estás mirando lo que miraste antes,
esos keyframes entran al mapa local. El repo padre midió 8.4 → 2.2 cm con
este único cambio, y descubrió que convierte cada re-visita en un cierre de
bucle implícito (su lección 14).

## 2. RELOCALIZACIÓN

Ahora mismo, perderse es definitivo: `_coast()` aplica velocidad constante
para siempre. Implementa la relocalización: tras N frames en LOST, matchea
el frame actual contra TODOS los keyframes de la base (sin filtro temporal) y
haz PnP. Si hay ≥ 40 inliers, salta a esa pose y vuelve a TRACK.

**Objetivo**: fabrica un "secuestro" (alimenta los frames 0..79 y luego salta
al 110) y mide en cuántos frames se recupera. El repo padre lo consigue en 2.

**Aviso medido** (su lección 20): si secuestras a una zona que el mapa local
ya cubre, la covisibilidad **absorbe el salto** y la reloc no dispara — el
tracking simplemente continúa. Para ejercitar la reloc hay que teletransportar
a una zona mapeada pero DISJUNTA. Buena noticia de robustez; ojo al diseñar
el test.

## 3. La escena importa (fácil, muy instructivo)

Modifica `genera_datos.py` para que en vez de carteles disjuntos haya UNA
pared de fondo enorme visible desde todo el corredor. Corre el SLAM.

**Objetivo**: observa los cierres de bucle que dispara. Verás que "cierra
bucles" a mitad de camino, contra keyframes con los que sólo comparte la
pared. Es la lección 15 del repo padre: la escena de prueba es parte del
experimento, no decorado. Un test que no puede fallar no prueba nada.

## 4. El bucle falso, en un SLAM de verdad (medio)

Baja `LOOP_MIN_INLIERS` de 40 a 5 (es decir: acepta bucles sin verificación
geométrica seria). Corre.

**Objetivo**: cuenta cuántos bucles se disparan y mide el ATE. Conecta con lo
que mediste en el nivel 12: un falso positivo mete una arista mentirosa en el
grafo, y Huber no basta para contenerla. Aquí ves la defensa real funcionando
—y lo que pasa cuando la quitas.

## 5. Puntos que sobran (medio)

El mapa acumula 5000+ puntos y nunca borra ninguno. Implementa el *culling*:
desactiva los puntos con < 3 observaciones que ya tienen ≥ 3 keyframes de
antigüedad.

**Objetivo**: mide la reducción del mapa y el ATE. El repo padre midió −33.9%
del mapa sin degradar el ATE. **Cuidado con el umbral**: `min_obs=2` es un
NO-OP, porque todo punto nace con exactamente 2 observaciones (los dos
extremos de su triangulación — la lección 7 del nivel 11). El umbral honesto
es 3: "un punto que ningún keyframe volvió a ver tras su par fundacional".
