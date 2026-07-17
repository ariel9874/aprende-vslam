# Nivel 17 — Features aprendidas · electiva

**Objetivo**: saber cuándo el deep learning gana a lo clásico — **midiéndolo**,
no creyéndolo. SuperPoint + LightGlue contra el ORB de todo el curso, sobre
fr1_desk: la secuencia handheld con motion blur que ya conoces (el nivel 14
la perdió; el 15 la cruzó con profundidad).

## La victoria, medida donde existe: a través del blur

El experimento controlado del examen: un frame nítido (el 40) contra frames
de la ráfaga de blur, mismo par para ambos frontends, con el RANSAC epipolar
de árbitro:

| par | ORB (E-inliers) | SuperPoint+LightGlue | factor |
|---|---|---|---|
| (40, 52) | 28 | **183** | 6.5× |
| (40, 56) | 17 | **118** | 6.9× |
| (40, 60) | 8 | **109** | **13.6×** |

Cuanto peor el blur, mayor la ventaja. La razón está en qué ES cada
descriptor: ORB compara intensidades en pares de puntos — vive de gradientes
locales que el blur borra. SuperPoint produce 256 floats desde un receptive
field grande: **contexto**, que sobrevive donde el detalle murió. Y LightGlue
además razona sobre las POSICIONES de ambos conjuntos (atención espacial).

## La lección incómoda (la 29 del padre, en su versión más cruda)

Con esa ventaja de 13×, ¿el sistema completo se salva? **No.** En este
tracker, la ráfaga dura de fr1_desk no la cruza NINGÚN frontend:

| frontend | perdidos (de 613) | fps |
|---|---|---|
| ORB | ~546 | ~26 (CPU) |
| SuperPoint+LightGlue | ~557 | ~9 (GPU) |

Empatados en la muerte. El episodio es **estructural**, no de umbrales ni de
descriptores: durante la ráfaga no hay pose fiable desde la que crear mapa, y
tras ella el barrido sigue por territorio nunca mapeado. El padre, con su
tracker completo (compuerta, gestión más fina), rescató 560 → ~140 perdidos —
pero su lección 29 termina igual que la nuestra: *el episodio que ni el deep
cruza*. La cura real no fue el frontend: fue el SENSOR. Ya la mediste — el
residuo de profundidad del nivel 15 cruza esta misma secuencia entera a
**2.3 cm**. Cuando un problema es de información faltante (baseline durante
el blur), cambiar de detector no lo inventa.

## Lo que construir este nivel destapó (todo quedó en el código)

1. **La vara de la INIT era floja** (`MIN_INIT_POINTS`): el nivel 14 aceptaba
   la init con ≥ 15 puntos triangulados. Con ORB nunca mordió; SuperPoint
   matchea TAN bien que un par de rotación casi pura produjo una "init
   válida" de 21 puntos — y el sistema nació muerto. Un frontend mejor
   destapa las varas flojas del resto del sistema.
2. **El ratio de Lowe NO es una constante universal** (`MatcherRatio`): en
   256-D float las distancias al 1er y 2do vecino se comprimen (concentración
   de la medida) y con 0.75 casi nada sobrevive — la relocalización no
   disparaba NUNCA (votos máx 20 con umbral 40). Para float se usa 0.90.
3. **El PUENTE frame-a-frame** (`_puente` — el "track with motion model" de
   ORB-SLAM): matchear el frame borroso contra el mapa nítido es lo difícil;
   borroso-contra-borroso es fácil (LightGlue: 949 matches en plena ráfaga).
   Las correspondencias al mapa viajan por la cadena y se re-abastecen
   re-proyectando el mapa con cada pose fresca. Alarga la agonía unos frames;
   no cruza la ráfaga dura (ver arriba: estructural).
4. **La asimetría de LightGlue**: necesita keypoints de AMBOS lados — sirve
   para pares de imágenes (init, puntos nuevos, bucle, reloc, puente), NO
   para el matching 3D→2D contra el mapa. Ahí sigue el ratio por descriptor
   y el matching guiado (agnóstico al descriptor).
5. **La RELOCALIZACIÓN** del nivel 16, portada al tracker monocular.

## Cómo correr

```bash
pip install -r requirements.txt          # lee dentro: torch + lightglue
python descarga_datos.py                 # fr1_desk, ~330 MB (una vez)
python 17_aprendidas.py                  # la comparativa completa
python verificacion.py                   # el examen (~4 min con GPU)
```

Con GPU NVIDIA, SuperPoint+LightGlue corre a ~9 fps. En CPU funciona pero
tarda MUCHO más (el examen puede pasar de 30 min): paciencia o GPU.

## Qué debes poder explicar al terminar

- Por qué el contexto (SuperPoint) sobrevive al blur y los gradientes
  locales (ORB) no — y cómo lo demuestra la tabla de pares.
- Por qué 13× más inliers par-a-par NO salvaron al sistema completo (¿qué
  información falta durante la ráfaga que ningún detector puede inventar?).
- La asimetría 2D-2D vs 3D-2D de los matchers aprendidos.
- Por qué el ratio de Lowe depende del espacio de descriptores.
- Qué es el puente frame-a-frame, por qué su cadena se muere de hambre sin
  re-abastecimiento, y qué límite tiene aun así.
