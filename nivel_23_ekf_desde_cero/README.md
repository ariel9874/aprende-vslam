# Nivel 23 — El EKF desde cero (y el EKF de estado de error) · bonus

**Objetivo**: construir el filtro de Kalman DESPACIO — cuatro actos, cada
uno agrega UNA sola cosa — hasta llegar al **error-state EKF**: el filtro
con el que los VIO reales (ARKit/ARCore) resuelven exactamente el problema
que el grafo del nivel 22 resolvió como smoother. Mismo mundo, mismas
medidas: al final, filtro y smoother se comparan número contra número.

El nivel 21 ya *usó* un EKF-SLAM como backend (rápido, para el espectro de
marginalización). Este nivel lo *explica*, desde antes de que exista.

## Los cuatro actos (cada uno es un script que corre solo)

### Acto 1 — Fusionar dos números (`fusion_1d.py`)

Ni filtro, ni estado, ni matrices: dos mediciones del mismo número, con
incertidumbres distintas. La media ponderada por inversa de varianza,
reescrita como `x + K·(z − x)`, ES la ecuación de corrección de Kalman —
la "ganancia" es el peso de una media ponderada, y nada más.

**Verificado**: procesar 1000 mediciones de una en una == guardarlas todas
y promediar (difieren 0.0); la información (1/σ²) SUMA, y σ cae como
**1/√N** — la raíz del ruido shot del nivel 01, del lado del estimador.

### Acto 2 — El estado se mueve (`kalman_lineal.py`)

Un carrito con posición y velocidad; el sensor solo mide posición. Se
agrega la PREDICCIÓN: la incertidumbre crece al apostar y baja al medir
(la "respiración" de σ — está en la gráfica). Dos lecciones con número:

- **La velocidad se estima sin sensor de velocidad**: derivar la medición
  da 7.1 m/s de error; el filtro, 0.42 (**17×**). El canal es `P[0,1]`, la
  correlación que la predicción creó.
- **El secreto mejor guardado**: el KF es EXACTAMENTE el grafo de factores
  lineal (nivel 21) resuelto por recursión. Verificado a precisión de
  máquina: estado final a **1.1e-11**, y P == covarianza marginal del grafo
  a **4.4e-14** — marginalizar el pasado (Schur) es lo que el filtro hace
  paso a paso. Esa igualdad se rompe en cuanto el mundo es no lineal, y de
  esa ruptura nace toda la discusión filtro-vs-smoother del curso.

### Acto 3 — El mundo no es lineal (`ekf_localizacion.py`)

Localización con mapa conocido (el SLAM ya lo hizo el 21; quitarle el mapa
deja la atención entera en la ÚNICA novedad: linealizar). F = ∂f/∂x y
H = ∂h/∂x entran en las MISMAS ecuaciones del acto 2 — eso es todo el "E".

**Verificado**: dead reckoning 41 cm → EKF **2.0 cm** (20×). Y el bug
clásico, reproducido a propósito: sin `envolver()` la innovación del
rumbo, un landmark casi detrás del robot da z = +3.13, h = −3.13 — error
real 0.01 rad, "sorpresa" de 2π — y el filtro acaba en **80 cm (39×
peor)**. Es intermitente (solo dispara cruzando ±π): por eso es tan famoso.

### Acto 4 — El EKF de error (`ekf_de_error.py`) — el acto estrella

El mundo del nivel 22, duplicado bit a bit (misma semilla): IMU a 100 Hz
con sesgo 0.03, visión a 4 Hz, apagón de 5 s en plena curva. El truco de
los VIO reales: partir el estado en dos pisos —

- el **NOMINAL** (pose, velocidad, sesgo, mapa) se integra a 100 Hz con el
  modelo completo, sin linealizar nada (el punto medio del nivel 22);
- el **ERROR** — pequeño siempre, porque se corrige y se resetea — es
  quien tiene filtro de Kalman. La linealización siempre se evalúa donde
  vale (la herida del acto 3, curada por construcción), el filtro vive
  lejos de ±π, y en 3D el δθ plano (3 números) protege al cuaternión (4)
  en el nominal: la razón histórica del error-state (Solà).

| configuración | RMSE total | en el APAGÓN |
|---|---|---|
| IMU sola (dead reckoning) | 786 cm | — |
| coast del nivel 22 (referencia) | 18.5 cm | 62.2 cm |
| ESKF sin estimar el sesgo | 184 cm | 174 cm |
| **ESKF completo** | **7.2 cm** | **17.2 cm** |
| grafo VIO del nivel 22 (smoother) | 4.7 cm | 4.8 cm |

Cuatro lecciones medidas:

- **El sesgo se descubre EN VIVO**: 0.028 a los 12 s, 0.0294 al final (el
  grafo del 22 lo supo AL FINAL de optimizar; el filtro lo ve aparecer).
  Y en la gráfica hay un regalo: durante el apagón la curva del sesgo se
  queda PLANA — sin visión no hay quien lo corrija — y al volver, salta.
- **No modelar el sesgo rompe al filtro** (184 cm, 26× peor). El grafo del
  22 sin sesgo daba 4.3 cm: el smoother re-pondera todo al final; el
  filtro sobreconfiado se envenena a sí mismo. La palabra es CONSISTENCIA.
- **La novatada de la primera vuelta**: vuelta 1, 9.5 cm; vuelta 2,
  **3.5 cm**. El mapa converge — pero las poses de la vuelta 1 ya fueron
  emitidas con el mapa joven, y el filtro no puede corregir su pasado.
- **Filtro vs smoother, número contra número**: el ESKF cruza el apagón
  **3.6×** mejor que el coast (la IMU mide el giro a oscuras)... y el
  smoother lo cruza **3.6×** mejor que el ESKF (17.2 vs 4.8 cm): el apagón
  cae en plena novatada, y solo el smoother la perdona. Lección 25, por
  tercera y definitiva vez.

## Lo honesto

En 2D, θ ya es mínimo: no hay cuaternión que proteger, así que la ventaja
*numérica* del error-state sobre un EKF directo es pequeña aquí. Lo que el
nivel mide es la *arquitectura* (nominal no lineal a 100 Hz, error siempre
cerca de cero, inmunidad al wrap, sesgo en vivo); la historia completa del
cuaternión queda como lectura: Solà, *"Quaternion kinematics for the
error-state Kalman filter"* — con este nivel encima, se lee como novela.

## Cómo correr

```bash
pip install -r requirements.txt      # numpy + matplotlib
python fusion_1d.py                  # acto 1 (y asi, uno por uno...)
python 23_ekf.py                     # los 4 actos + la tabla + graficas
python verificacion.py               # el examen (14 checks, segundos)
```

## Qué debes poder explicar al terminar

- Por qué K es una media ponderada, y qué significa K→0 y K→1.
- Por qué el filtro estima variables que ningún sensor mide (¿qué papel
  juega la covarianza cruzada?).
- En qué sentido exacto el KF "es" el grafo lineal — y qué rompe esa
  igualdad.
- El bug del ángulo: por qué es intermitente y dónde va el `envolver()`.
- Los dos pisos del error-state: qué vive en el nominal, qué vive en el
  error, y por qué eso cura la debilidad del EKF.
- Por qué el filtro sin modelo de sesgo diverge donde el smoother apenas
  se despeina — y por qué la primera vuelta es siempre la peor.
