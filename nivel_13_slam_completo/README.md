# Nivel 13 — El SLAM completo

**Objetivo**: ensamblar TODO lo anterior en un sistema que funciona. Una
máquina de estados, un mapa que crece por keyframes, un BA que lo sostiene y
un cierre de bucle que se **verifica antes de creérselo**.

Al terminar este nivel tienes un SLAM. Pequeño, pero un SLAM de verdad.

## La arquitectura (la de ORB-SLAM, en pequeño)

```
INIT    dos vistas con paralaje -> matriz esencial (nivel 07)
        -> triangular (nivel 09) -> gauge mediana = 1 (nivel 10)

TRACK   matchear contra el MAPA LOCAL -> PnP (nivel 10)
        -> ¿toca keyframe? -> triangular puntos nuevos
                           -> BA de ventana (nivel 11)
                           -> ¿bucle? -> VERIFICAR -> grafo Sim(3) (nivel 12)

LOST    sin pose fiable: velocidad constante ("coasting")
```

## La ablación: cada pieza carga peso

Misma secuencia, mismo frontend. Sólo se apagan piezas del backend:

| configuración | ATE (keyframes) |
|---|---|
| sin BA (ni bucle) | **148.5 cm** ← colapso |
| sin cierre de bucle | 6.1 cm |
| sistema completo | **5.8 cm** |

**El BA es la pieza que sostiene el mapa.** Sin él, la triangulación y el PnP
se retroalimentan con su propio error: cada keyframe crea puntos desde una
pose ya sesgada, y esos puntos sesgan la siguiente pose. El sistema no se
"pierde" (0 frames perdidos), simplemente construye un mundo coherente y
equivocado. El cierre de bucle refina lo que el BA ya hizo bien.

## Las dos métricas (y por qué la que usabas mentía)

| | ATE |
|---|---|
| trayectoria **online** (poses emitidas) | 8.8 cm |
| trayectoria **final de keyframes** | **5.8 cm** |

Las poses que el sistema EMITE frame a frame **se congelan al salir**. Cuando
el bucle corrige el mapa en el frame 190, nadie reescribe los frames 0..189.
Evaluar sobre ellas es **no ver nada de lo que hace el backend** — de hecho,
el ATE online puede EMPEORAR al activar el bucle (la corrección mete un
escalón en una trayectoria que ya no se puede arreglar; aquí: 8.0 → 8.8 cm).

Los keyframes SÍ se reescriben (el BA y el grafo los tocan). Su trayectoria
final es la que refleja el estado real del sistema, y es la que reporta
ORB-SLAM. **El repo padre descubrió esto tarde y le costó caro** (su lección
25): en fr2_desk medía 21.9 cm online contra 4.8 cm en keyframes. Su
"problema de deriva de escala" era, en buena parte, un artefacto de medición.

## El bucle se verifica, no se cree

La lección del nivel 12 fue que Huber **no** salva un falso positivo. Así que
antes de meter la arista al grafo, tres filtros:

1. **temporal**: sólo keyframes lejanos en el tiempo (si no, "cierras bucles"
   con tu propio vecino);
2. **descriptores**: matching contra ese keyframe (≥ 60 matches);
3. **VERIFICACIÓN GEOMÉTRICA**: PnP contra sus puntos 3D. Si no hay ≥ 40
   inliers, **no hay bucle**. Y no se discute.

> Ojo a las UNIDADES del filtro temporal: se cuenta en KEYFRAMES, no en
> frames. La secuencia tiene 200 frames pero sólo ~18 keyframes; poner el gap
> en 20 (pensando en frames) hace que ningún candidato califique jamás y el
> cierre de bucle no dispare nunca — **en silencio**. Me pasó construyendo
> este nivel.

## Cómo correr

```bash
pip install -r requirements.txt
python genera_datos.py       # el corredor: ida y vuelta (200 frames)
python 13_slam.py            # el sistema completo + la ablacion
python verificacion.py       # el examen del nivel
```

Resultados en `salida/`: `corredor.png` — la trayectoria como **serie
temporal** x(t) y el error(t) con los bucles marcados. (Una trayectoria de
ida y vuelta se solapa consigo misma en planta y no se ve nada: es la lección
16 del repo padre.)

## Lo que este SLAM NO tiene (y dónde está)

Es una versión recortada a propósito (~350 líneas frente a las ~850 del
padre). Le faltan, y son los ejercicios de este nivel y el contenido de los
siguientes:

- **covisibilidad** (el mapa local es por recencia): ejercicio 1. Es más que
  eficiencia — es corrección.
- **relocalización** (aquí, perderse es definitivo): ejercicio 4.
- **matching guiado** por reproyección: nivel 14. Es *la* palanca en datos
  reales.
- BoW, hilo de mapeo, C++: nivel 18.

## Qué debes poder explicar al terminar

- Los tres estados y qué dispara cada transición.
- Por qué el mapa local no es sólo una optimización.
- Por qué el BA es imprescindible y el bucle es un refinamiento.
- Por qué la trayectoria online miente y la de keyframes no.
- Los tres filtros del cierre de bucle, y cuál es el que de verdad protege.
