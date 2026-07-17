# Nivel 18 — Ingeniería de tiempo real · electiva

**Objetivo**: el MÉTODO de optimización de sistemas — **perfilar, sustituir
solo el punto caliente, verificar la equivalencia**. No "hacerlo rápido":
saber DÓNDE, y probar que la versión rápida da LO MISMO.

## Paso 1: perfilar (la intuición se refuta)

Antes de correr el driver, apunta dónde crees TÚ que se va el tiempo. El
repo padre apostaba por cv2 (su docs/04); su perfil dijo: **BA 57% +
matching guiado 37%, cv2 8%**. El nuestro (fr2_xyz, 1000 frames, la tabla
que imprime el driver):

| etapa | % del tracking |
|---|---|
| matching guiado | **62.4%** |
| BA local | **32.6%** |
| frontend (ORB) | 1.1% |
| PnP | 0.6% |
| bucle: reconocimiento | **0.0%** |

Los mismos dos puntos calientes que el padre (en otro orden — nuestro guiado
es Python puro; el suyo ya era C++ cuando midió), y cv2 igual de inocente.
Y una sorpresa honesta: el reconocimiento de lugar — lo que BoW acelera —
perfila a **0.0%** con ~80 keyframes (la fuerza bruta acotada a 300
descriptores ya era barata a esta escala). BoW es un seguro para cuando la
base crece, no una ganancia HOY: *no optimices lo que el perfil no señala* —
Amdahl no perdona ni a las gemelas bonitas.

## Paso 2: la gemela (sustituir SOLO el punto caliente)

Una **gemela** es otra implementación del MISMO contrato. Este nivel enchufa
dos, elegidas porque corren en cualquier máquina:

1. **`ba_rapido.py` — el BA vectorizado**: exactamente los mismos cálculos
   del BA didáctico (nivel 11), en lotes de NumPy en vez de bucles de
   Python. Mismos FLOPs, menos intérprete (~µs por operación de Python,
   amortizados sobre miles de elementos por llamada). El tracker no se
   entera: `SLAM(K, ba_fn=bundle_adjustment_rapido)`.
2. **`bow.py` — bolsa de palabras visuales**: el reconocimiento de lugar
   deja de ser O(KFs)·(matching completo) y pasa a un histograma tf·idf con
   índice invertido (Sivic & Zisserman 2003). Para ORB, el centroide del
   k-medias es el VOTO DE MAYORÍA por bit (la mediana de Hamming).

El siguiente orden de magnitud existe y es el del padre: GTSAM (factor
graphs C++ con dispersidad de verdad, ~2× de fps del sistema), iSAM2
(incrementalidad: no re-optimizar lo que no tocaste), el matching guiado en
C++ con pybind11 (su 37% → ruido), y el hilo de mapeo. Su escalera completa:
**4.3 → 9.5 → 18.7 → 25.7 → 46.7 fps**. Exigen toolchain (conda / MSVC):
son los ejercicios 1, 2 y 4, con las referencias del padre.

## Paso 3: verificar (o no es una optimización)

La regla del nivel: **una gemela sin test de equivalencia es un bug con
buena prensa**. Lo que el examen verifica:

- El BA vectorizado da LO MISMO que el didáctico **a precisión de máquina**
  (dif. máxima medida: 4·10⁻¹⁶ en poses, mismo camino LM iteración a
  iteración) — y 4.4× más rápido ya en una ventana sintética.
- El BoW propone el MISMO candidato que la fuerza bruta (y el correcto).
- En el sistema completo (fr2_xyz): más rápido con el MISMO ATE
  (**0.76 cm = 0.76 cm** — paridad exacta al centésimo).

El estándar del padre llega más lejos (su lección 31): su gemela C++ se
verificó par a par hasta la **semántica de desempate de np.argmin**. Los
bugs de las gemelas viven en los empates y los bordes.

## La escalera medida (fr2_xyz, 600 frames del examen)

| configuración | tracking | BA global | ATE final-KF |
|---|---|---|---|
| referencia (nivel 14) | 2.4 fps | 252 s | 0.76 cm |
| + BA vectorizado + BoW | **3.1 fps** (1.31×) | **109 s** (2.3×) | **0.76 cm** |

La mejora del tracking parece modesta — hasta que miras el perfil: el BA
local es solo una parte del frame (ORB, guiado y PnP viven en C++ de OpenCV
desde el principio). El GBA — puro Python didáctico — se acelera 2.3× y
sigue bajando conforme crece el mapa. Amdahl manda: solo se acelera la
fracción que sustituyes.

## Cómo correr

```bash
pip install -r requirements.txt
python descarga_datos.py            # fr2_xyz (~2.1 GB) — compartido con el 14
python 18_tiempo_real.py            # perfil + escalera (1000 frames)
python verificacion.py              # equivalencias (sin dataset) + escalera
```

Los actos 1-2 del examen (los tests de equivalencia) NO necesitan dataset.

## Qué debes poder explicar al terminar

- Por qué perfilar va ANTES que optimizar (y qué apostabas tú antes de ver
  tu tabla).
- De dónde sale la velocidad de la gemela vectorizada si hace los mismos
  FLOPs.
- Qué verifica exactamente un test de equivalencia — y la diferencia entre
  equivalencia de RESULTADO (GTSAM: otro camino, mismo óptimo) y de
  TRAYECTORIA (la gemela numpy: mismo camino LM).
- Por qué el centroide de k-medias en Hamming es el voto de mayoría por bit.
- La ley de Amdahl aplicada a tu propia tabla de perfil.
