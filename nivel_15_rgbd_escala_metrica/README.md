# Nivel 15 — RGB-D y escala métrica

**Objetivo**: eliminar la ambigüedad de escala con un sensor de profundidad —
SLAM en **METROS de verdad**. Y de paso, cruzar la secuencia que derrotó al
nivel 14.

## La idea en una línea

Desde el nivel 07 arrastras una deuda: la traslación monocular sale UNITARIA,
la escala es un *gauge* que se fija por convención (mediana = 1, nivel 10) y
el ATE se mide tras regalarle una similitud al alineador. La profundidad lo
cambia todo: **z es una MEDICIÓN**, y la retro-proyección

```
X_c = z · K⁻¹ · [u, v, 1]ᵀ
```

da puntos en la unidad del sensor. Tres decisiones se caen en cascada de ese
hecho — cada una una lección medida del repo padre:

## 1. La init es instantánea

Nada de esperar paralaje ni matriz esencial: **el primer frame con
profundidad YA es un mapa métrico** (~50+ puntos retro-proyectados). Por eso
RGB-D no sufre el fallo handheld de fr1: crear mapa no requiere baseline. Los
puntos nacen con UNA observación y el BA los excluye hasta la segunda
(`min_obs=2` — la lección del nivel 11 sigue mandando).

**El bug del mapa mixto** (lección 36 del padre): el stream de profundidad
puede arrancar tarde (fr1_desk: los primeros frames no tienen pareja de
depth). Si inicializas monocular "mientras tanto", nace un mapa a escala
gauge que luego se mezcla con puntos en metros — dos escalas en tensión
permanente, y con la escala ≈ 1 de PURA casualidad (la mediana del escritorio
es ~1 m ≈ el gauge mediana=1: el bug queda invisible). La cura es doble: el
driver ESPERA al primer frame con depth, y el tracker no inicializa sin ella.

## 2. El residuo de profundidad: el ancla métrica del BA

Meter z directo al costo mezclaría unidades (metros vs píxeles). El truco de
ORB-SLAM2: convertir la profundidad en la coordenada que mediría una cámara
derecha VIRTUAL a baseline b:

```
u_R = u − fx·b/z        (fx·b ≡ bf = 40)
```

y extender el residuo de `[u, v]` a `[u, v, u_R]`. Todo queda en píxeles
(misma Huber, mismo Schur) y el peso de la profundidad decae con z² —
exactamente el inverso del ruido del Kinect: la física y la geometría se
cancelan en la dirección correcta. Cada observación (también las
RE-observaciones de puntos viejos) lleva su medición métrica fresca.

Medido aquí (la ablación del examen, `--sin-residuo`): el mapa pre-GBA pasa
de 2.6 a **5.8 cm** (2.3×) y la escala de 1.012 a 0.982. En el padre, que sí
tiene relocalización, la ablación además cruzaba peor el blur: 12.8 vs 2.8 cm.

## 3. El bucle métrico va en SE(3), no en Sim(3)

La moraleja conceptual más bonita del curso (lección 35 del padre). En el
nivel 12 aprendiste que el bucle monocular DEBE ser Sim(3): la deriva incluye
escala y hay que redistribuirla (Strasdat). Aquí es al revés: la escala es
una medición del sensor y **no se negocia**. Un bucle Sim(3) re-escalaría el
mapa viejo mientras los puntos nuevos siguen naciendo métricos — el siguiente
bucle "corrige" la discrepancia que el anterior CREÓ y el error se COMPONE.
El padre lo midió en fr2_xyz RGB-D: los bucles empezaban midiendo s_rel ≈ 1 y
degeneraban hasta 0.03 (ATE 22.1 cm, escala 2.09); en SE(3), 4.7 cm y escala
1.036. **El grupo del bucle depende de QUIÉN fija la escala**: convención →
Sim(3); sensor → SE(3).

## El chequeo de honestidad

El ATE de este nivel se evalúa con alineación **RÍGIDA** (`with_scale=False`:
si el mapa está en metros, no hay escala que regalar) — y la escala de
similitud se calcula APARTE, como diagnóstico: **≈ 1.000 es la prueba de que
el mapa está de verdad en metros**.

## Los números (fr1_desk, 613 frames, medidos con este código)

| | nivel 14 (monocular) | nivel 15 (RGB-D) |
|---|---|---|
| frames perdidos | 562 de 613 (muere ~frame 60) | **203** (los episodios de blur) |
| keyframes | 5 | 30 |
| ATE final-KF | — (sin mapa útil) | **2.3 cm RÍGIDO** tras GBA |
| escala similitud | — | **1.012** |

La referencia del padre: 2.8 cm, escala 1.005, 0 perdidos — con
relocalización. Los 203 perdidos que quedan son episodios de motion blur
donde ORB no engancha: la reloc (ejercicio 4) los convierte en recuperables,
y las features aprendidas (nivel 17) los reducen de raíz.

## Cómo correr

```bash
pip install -r requirements.txt
python descarga_datos.py        # fr1_desk, ~330 MB (una vez)
python 15_rgbd.py               # la secuencia completa (~2 min + GBA)
python verificacion.py          # el examen (~5 min)
```

Si ya tienes fr1_desk (p. ej. del ejercicio 1 del nivel 14), pasa `--root`.
`--sin-residuo` es la ablación del examen.

## Qué debes poder explicar al terminar

- Por qué la retro-proyección mata la danza de la inicialización monocular —
  y qué problema de fr1 desaparece con ella.
- Cómo el truco del estéreo virtual mete metros al BA sin mezclar unidades,
  y por qué su peso decae con z² es una VIRTUD.
- Por qué el bucle métrico en Sim(3) no es un error inocente sino un
  compositor de error — y quién fija la escala en cada configuración.
- Qué prueba la escala de similitud ≈ 1.000 (y qué NO prueba un ATE con
  alineación de similitud).
- El bug del mapa mixto: cómo nace, por qué es invisible, y las dos curas.
