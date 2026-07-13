# Nivel 10 — PnP y mapa persistente: de odometría a SLAM

**Objetivo**: dejar de tirar todo lo aprendido en cada frame. En vez de
re-estimar la geometría desde cero entre pares consecutivos (nivel 08),
construyes un MAPA que persiste y localizas la cámara CONTRA él (PnP,
3D→2D). Es el salto arquitectónico que separa la odometría del SLAM.

**El número del nivel**: la misma secuencia, el mismo ORB, el mismo ratio
test — pero el ATE baja de **18.6 cm a 8.7 cm (53% mejor)**. No tocaste
ninguna perilla: cambiaste la arquitectura. El script corre AMBOS sistemas
sobre las mismas imágenes y te lo enseña lado a lado. (El repo padre midió
el mismo salto al pasar de su v0.1 a su v0.2: 13 → 7 cm.)

> Nota: la secuencia de este nivel es la del nivel 07/09 (con los tres
> planos repartidos), no exactamente la del 08 — por eso la VO 2D-2D marca
> aquí 18.6 cm y allí 16.0. Lo que importa es la comparación INTERNA:
> ambos sistemas corren sobre las mismas imágenes.

## Teoría mínima

**PnP (Perspective-n-Point).** Dado un conjunto de puntos 3D del mapa `{X_i}`
y los píxeles `{u_i}` donde los ves ahora, encuentra la pose que minimiza la
reproyección:

```
T* = argmin_T  Σ ‖ π(K, T_c_w · X_i) − u_i ‖²
```

Se resuelve con RANSAC (para separar inliers) + refinamiento no lineal. La
diferencia con la geometría epipolar del nivel 07 es profunda:

| | 2D-2D (nivel 07/08) | 3D-2D / PnP (este nivel) |
|---|---|---|
| Compara | frame ↔ frame anterior | frame ↔ **mapa** |
| La escala | se re-inventa (‖t‖=1) | se **hereda** del mapa |
| Evidencia | 2 vistas | todas las que vieron el punto |
| El error | se acumula sin freno | se ancla al mapa |

**Por qué esto mata la deriva.** En el nivel 08, cada pose se apoyaba en la
anterior: los errores se componían sin límite. Ahora la pose se calcula
contra un mapa que existe desde el principio: si el mapa es bueno, la pose
es buena, sin importar cuántos frames hayan pasado. La deriva no desaparece
(el mapa también se construye con error, y crece por sus bordes), pero deja
de ser una composición ciega.

**El keyframe.** No todos los frames merecen entrar al mapa: triangular
desde vistas casi idénticas da puntos sin paralaje (nivel 09). Se elige un
subconjunto — los **keyframes** — con una política: no antes de N frames, y
cuando el tracking empiece a quedarse sin puntos. Cada keyframe triangula
puntos nuevos contra el anterior.

**El gauge.** El primer par de vistas fija la escala arbitraria del sistema
(monocular: `‖t‖=1`, y por convención del curso la profundidad MEDIANA del
mapa inicial se normaliza a 1.0). Todo lo demás la HEREDA. Ojo con el matiz,
que confunde a todo el mundo: sólo los puntos de la inicialización tienen
mediana 1.0. Los que nacen en keyframes posteriores heredan esa escala pero
miran zonas más lejanas de la escena, así que la mediana del mapa completo
sale distinta (2.12 en esta corrida) — y eso es correcto, no un bug. El
gauge es un acto único de bautizo, no una propiedad que el mapa mantenga.
Ese gauge persigue al SLAM monocular hasta el nivel 15, donde un sensor de
profundidad lo convierte en una MEDICIÓN y deja de ser negociable.

**La regla que cuesta cara.** El repo padre la aprendió por las malas: un
keyframe insertado desde una pose dudosa (26 inliers) creó 584 puntos basura
y el sistema empezó a teletransportarse. *Nunca crear mapa desde una pose
incierta* — por eso hay un piso de salud (`KF_MIN_INLIERS`) antes de
insertar.

## Cómo correr

```bash
pip install -r requirements.txt
python genera_datos.py               # la misma secuencia del nivel 08
python 10_pnp_mapa.py                # el tracker PnP + la comparacion con VO
python verificacion.py               # el examen del nivel
```

Resultados en `salida/`: `trayectoria.txt`, `comparacion.png` (VO 2D-2D vs
PnP contra el mismo ground truth) y `mapa.ply`.

## Qué debes poder explicar al terminar

- Qué minimiza el PnP y en qué se diferencia de la matriz esencial.
- Por qué el mapa "ancla" la escala en vez de re-inventarla cada frame.
- Qué es un keyframe y por qué no todos los frames lo son.
- Qué pasa si insertas un keyframe desde una pose mala (pruébalo:
  ejercicio 3).
