# Nivel 22 — El factor de IMU · bonus

**Objetivo**: entender por qué VIO (visión + inercia) ganó el mercado —
construyendo la **preintegración de IMU** desde cero y midiendo exactamente
qué compra. Es la secuela natural del nivel 21: un sensor nuevo = un factor
nuevo... salvo que ESTE sensor mide a 100 Hz y trae un mentiroso dentro (el
sesgo).

## El escenario (mundo_imu.py)

Un vehículo 2D "tipo dron" — SIN odometría — recorre el circuito del nivel
21 en continuo (1 m/s; curvas de verdad, con centrípeta). Sus sentidos:
visión a 4 Hz (landmarks) y una IMU a 100 Hz (giróscopo + acelerómetro) con
ruido y con un **sesgo de 0.03 rad/s que nadie le confiesa al grafo**. Y en
la peor curva, un **APAGÓN visual de 5 segundos** — la ráfaga de blur del
nivel 17, en versión controlada.

## Las tres ideas, cada una con su número

### 1. La preintegración (Lupton; Forster et al.)

Mil muestras de IMU entre dos keyframes no pueden ser mil variables. Se
integran RELATIVAS al keyframe i (independientes de dónde esté i en el
mundo) en tres deltas — δθ, δv, δp — que forman UN factor. El estado por
nodo crece a `[x, y, θ, vx, vy, b_g]`: el factor habla de velocidades y
sesgos, no solo de poses (y ese crecimiento es la razón de que los VIO
reales vivan en las ventanas marginalizadas del nivel 21).

Verificado: con IMU perfecta, encadenar los deltas reproduce el circuito
entero (62 s) a **0.02 cm**. Y una lección de construcción que quedó en el
código: con integración de Euler eran **20 cm** — el punto medio, al mismo
costo, compró tres órdenes de magnitud.

### 2. El truco del sesgo (la contribución de Forster)

Los deltas dependen del sesgo con que se integraron; si el optimizador
actualiza b_g, ¿re-integrar mil muestras por iteración? No: durante la
preintegración se acumulan también los jacobianos ∂δ/∂b_g, y el factor
corrige a PRIMER ORDEN. Verificado: en θ la corrección es **exacta**
(2·10⁻¹⁶ — δθ es lineal en el sesgo); en δv el error cae **402×** (queda a
segundo orden). Ese truco es lo que hace tratable el VIO en tiempo real.

### 3. Por qué se FUSIONA (y no se integra a secas)

| configuración | RMSE total | en el APAGÓN |
|---|---|---|
| IMU sola (dead reckoning) | **786 cm** | — |
| visión + coast (sin IMU) | 18.5 cm | **62.2 cm** |
| VIO sin estimar el sesgo | 4.3 cm | 6.6 cm |
| **VIO completo** | **4.7 cm** | **4.8 cm** |

- **La IMU sola no navega**: el sesgo rota el mundo entero (786 cm; con
  sesgo perfecto aún deriva 58 cm — la doble integración no perdona).
- **El apagón es EL argumento** (12.9×): el factor coast *asume* velocidad
  constante, y el robot giró 90° a oscuras. La IMU no asume: **midió** el
  giro (giróscopo) y la centrípeta (acelerómetro). Es la información que al
  nivel 17 le faltaba durante el blur — ningún detector la puede inventar;
  un sensor inercial la trae de fábrica.
- **El sesgo se descubre solo**: el grafo lo estima en 0.0297 (real: 0.03)
  sin que nadie se lo diga — lo deduce de que visión e IMU no cuadran de
  otro modo. Y paga exactamente donde debe: dentro del apagón (6.6 → 4.8),
  porque fuera la visión corrige al giróscopo mentiroso frame a frame.

## Lo que el 2D no puede enseñar (dicho honesto)

En 3D el acelerómetro mide además la GRAVEDAD: una referencia absoluta de
roll/pitch (por eso los VIO tienen solo 4 grados de libertad no observables,
no 6) y el ancla de ESCALA del monocular. Nuestro mundo 2D con landmarks
métricos no puede mostrar ninguna de las dos — quedan como lectura (Forster
et al., "On-Manifold Preintegration") y el ejercicio 4 se asoma a la escala.

## Cómo correr

```bash
pip install -r requirements.txt      # numpy + matplotlib
python 22_vio.py                     # la tabla + las graficas (segundos)
python verificacion.py               # el examen (10 checks, segundos)
```

## Qué debes poder explicar al terminar

- Por qué los deltas se integran relativos al keyframe i (¿qué se rompería
  si fueran absolutos?).
- El truco del sesgo: qué se acumula durante la preintegración y por qué la
  corrección es exacta en θ pero de primer orden en δv y δp.
- Por qué el estado por nodo crece — y qué presión pone eso sobre la
  arquitectura (nivel 21).
- La diferencia entre ASUMIR (coast) y MEDIR (IMU), con los números del
  apagón.
- Por qué el sesgo es observable cuando hay visión, y qué pasa con él en
  los tramos ciegos.
