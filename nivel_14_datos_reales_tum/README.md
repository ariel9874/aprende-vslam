# Nivel 14 — Datos reales: TUM RGB-D

**Objetivo**: cruzar el salto sim→real. El SLAM del nivel 13, contra imágenes
de una cámara de verdad — y las palancas medidas que hacen posible el cruce:
el **matching guiado por reproyección**, un **mapa que se re-observa** en vez
de duplicarse, y un **BA global offline** midiendo donde hay que medir.

## Lo que el mundo real añade (y quién lo resuelve)

| lo nuevo | la cura | dónde |
|---|---|---|
| distorsión de lente | pre-rectificar con la calibración publicada (nivel 04) | driver |
| timestamps reales | asociar RGB (~30 Hz) ↔ mocap (~100 Hz) por cercanía | `dataset.py` |
| matching ambiguo | matching GUIADO por reproyección | `slam.py` |
| re-visitas → duplicados | covisibilidad + "observar antes que crear" | `slam.py` |
| deriva acumulada | UN bundle adjustment global, offline | `slam.py` |

## La palanca: el matching guiado (lección 24 del padre)

Hasta ahora el tracking matcheaba por descriptor contra el mapa local. En
datos reales eso se degrada: contra un mapa grande, ORB tiene vecinos
casuales y el matching por descriptor produce asociaciones ambiguas (el repo
padre midió matches con CERO inliers geométricos — su lección 22). La cura no
es un umbral: es usar la GEOMETRÍA.

1. PREDICE la pose (velocidad constante: `T_pred = T_w_c · T_rel`),
2. PROYECTA cada punto del mapa local a la imagen,
3. busca su descriptor sólo entre los keypoints a **< 15 px** de la proyección,
4. asigna GREEDY por distancia: cada punto y cada keypoint se usan una vez.

Dentro de esa ventana el vecino correcto casi no tiene rival: suben los
inliers verdaderos Y baja el ruido. Es el "track local map" de ORB-SLAM. El
padre lo midió como un solo cambio: **fr2_desk pasó de 104.9 cm / 1347
frames perdidos (colapso) a 21.9 cm / 0 perdidos**. Aquí, en fr2_xyz:
mediana de inliers **914 con guiado vs 574 sin él** (+59%, la ablación del
examen).

## El mapa-espejismo (la lección 14, sufrida en carne propia)

Construyendo este nivel reprodujimos — sin querer — la lección 14 del padre.
La primera versión heredaba del nivel 13 el mapa local por RECENCIA (los
últimos 5 keyframes) y triangulaba en cada keyframe todo lo no visto por el
keyframe ANTERIOR. En el corredor de 200 frames eso aguantaba. En una sesión
real de 3669 frames que re-visita la misma escena una y otra vez, NO:

- al re-visitar, los puntos originales no están en el mapa local → se
  re-triangulan como puntos NUEVOS, desplazados por la deriva;
- el mapa acabó en **96 000 puntos** (la escena cabe en ~35 000) y el ATE en
  **35 cm**;
- y el BA global NO pudo arreglarlo: su costo se estancaba en un plateau —
  dos copias coherentes del mismo mundo no se reconcilian optimizando. El
  problema no era el optimizador: era el mapa.

La cura (la arquitectura del padre, dos piezas que cooperan con el bucle):

1. **Covisibilidad en el mapa local**: además de los recientes, entran los
   keyframes que comparten ≥ 15 puntos con el último. El cierre de bucle
   registra observaciones puente → el keyframe actual se vuelve covisible
   con el segmento viejo → sus puntos ORIGINALES vuelven al mapa local.
2. **Observar antes que crear**: el keyframe registra como observaciones las
   correspondencias que el PnP ya verificó, y sólo triangula keypoints que
   quedaron sin punto. Cada re-visita se convierte en un cierre de bucle
   implícito.

Resultado medido (misma secuencia, mismo frontend): mapa 96 000 → **35 714
puntos**, con **6.3 observaciones por punto** (antes ~2.4), y el ATE de la
sesión entera **35.2 → 1.4 cm**.

## El remate: el BA global offline (lecciones 26-27)

El grafo Sim(3) del bucle corrige POSES; los PUNTOS quedan donde estaban y
la escala intermedia queda mal. Un BA global (todos los keyframes y puntos,
2 anclas de gauge) sí la reparte — y las observaciones puente atan los
extremos de la cadena. Dos lecciones medidas del padre, incorporadas:

- **Offline, no en caliente**: probado tras cada bucle, el BA global sacude
  el mapa y descarrila el tracking (fr2_xyz del padre: 5 → 346 frames
  perdidos). Como la métrica es la trayectoria FINAL de keyframes (nivel
  13), UN BA al terminar da el beneficio sin tocar nada.
- **50 iteraciones, no 10**: su "límite de deriva" no era estructural — era
  un optimizador a medio converger (su barrido: 0→13.0, 10→12.0, 25→3.5,
  **50→0.4 cm**). Y este nivel añade el matiz que nos tocó descubrir: si el
  costo del BA se ESTANCA lejos del suelo, más iteraciones no salvan un mapa
  roto. Convergencia se verifica, y estructura también.

## Los números de este nivel (fr2_xyz, medidos con este código)

| | tramo del examen (600 frames) | sesión entera (3669) |
|---|---|---|
| frames perdidos | **0** | **0** |
| keyframes / bucles | 41 / 2 | 245 / 22 |
| mapa | 8 882 pts (6.3 obs/pt) | 35 714 pts |
| ATE online | 0.9 cm | 3.2 cm |
| ATE final-KF | 1.4 cm | 4.9 cm |
| **ATE final-KF + BA global** | **0.8 cm** | **1.4 cm** |

La referencia del padre en esta secuencia: 0.4–1.5 cm según configuración —
con covisibilidad completa, culling, relocalización y BoW. Este nivel, con
~450 líneas de tracker, queda dentro de esa banda.

## Cómo correr

```bash
pip install -r requirements.txt
python descarga_datos.py        # fr2_xyz, ~2.1 GB (una vez)
python 14_datos_reales.py       # la secuencia entera (~30 min + GBA)
python verificacion.py          # el examen: 600 frames, ~12 min
```

Si ya tienes fr2_xyz en otra ruta, todos los scripts aceptan `--root`.
`--max-frames N` acorta cualquier corrida; `--sin-guiado` es la ablación.

## Lo que este nivel NO arregla (y dónde sigue)

- **fr1_desk se pierde** (handheld, rotación rápida, motion blur): el
  matching ORB — aun guiado — no engancha en el blur, y sin relocalización
  perderse es definitivo. Medido: **562 de 613 frames perdidos** (el padre,
  con reloc: 560 — el límite es el frontend, no la arquitectura). Es el
  ejercicio 1: documentar el límite. Las features aprendidas lo rescatan a
  medias (nivel 17); el residuo de profundidad RGB-D lo cruza entero
  (nivel 15).
- **La escala sigue siendo un gauge**: el ATE se mide tras alinear una
  similitud. El SLAM en METROS llega con el sensor de profundidad (nivel 15).
- **La fuerza bruta del bucle no escala**: aquí se vota con los 300
  descriptores más fuertes por keyframe; la solución real es BoW (nivel 18).

## Qué debes poder explicar al terminar

- Por qué el matching por descriptor se degrada contra un mapa real — y qué
  información nueva aporta la ventana de 15 px.
- Qué es el mapa-espejismo, por qué NINGÚN optimizador lo arregla, y cómo lo
  evitan covisibilidad + "observar antes que crear".
- Por qué las observaciones puente importan más que la corrección Sim(3).
- Por qué el BA global va offline, y las dos cosas que hay que verificar
  antes de culpar al método: que convergió, y que el mapa no está roto.
- Dónde está el borde del envelope de este sistema, con números.
