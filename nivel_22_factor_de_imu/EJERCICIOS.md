# Ejercicios — Nivel 22

## 1. El apagón crece (fácil, la curva del nivel)

El apagón dura 5 s. Bárrelo: 2, 5, 10, 20 segundos (ajusta el radio en
`generar`) y grafica el error-en-apagón de coast vs VIO contra la duración.

**Objetivo**: dos curvas que divergen. El coast degrada catastróficamente en
cuanto el apagón contiene una curva; el VIO degrada SUAVE (la deriva de la
IMU en t^{3/2}). Encuentra el punto donde incluso el VIO supera 20 cm: ese
es el presupuesto de apagón de tu sensor — un número de diseño real (¿cuánto
blur tolera tu dron?).

## 2. La covarianza del preintegrado (medio — lo que simplificamos)

Nuestro factor usa una información DIAGONAL fija (`INFO_IMU`). La de verdad
se PROPAGA durante la preintegración: Σ_{k+1} = A_k·Σ_k·A_kᵀ + B_k·Q·B_kᵀ,
igual que la predicción del EKF del nivel 21. Impleméntala (A_k y B_k son
los jacobianos del paso respecto al estado delta y al ruido).

**Objetivo**: la información correcta depende de la DURACIÓN del tramo (más
muestras = más incertidumbre) y de su geometría. Mide si cambia el resultado
en este mundo — y explica por qué en un mundo tan homogéneo cambia poco
(todos los tramos duran igual). ¿Cuándo importaría mucho?

## 3. Sesgo que deriva (medio)

Nuestro sesgo es constante. Hazlo derivar: `b_g(t) = 0.03 + 0.01·sin(2πt/40)`
en la generación.

**Objetivo**: el random walk del grafo (`INFO_BIAS_RW`) ahora importa de
verdad: con información infinita (sesgo forzado constante) el grafo no puede
seguir la deriva; muy floja, el sesgo absorbe error que no le toca. Barre
`INFO_BIAS_RW` y grafica. Es la calibración clásica de todo VIO real (los
sigmas de random walk de la hoja de datos de tu IMU).

## 4. Asómate a la escala (difícil, conceptual)

Cambia `observar` a SOLO el ángulo al landmark (bearing-only: visión
monocular de juguete — sin rango, la visión pierde la escala en 2D).

**Objetivo**: corre visión-sola (coast) vs VIO. Sin rango, el grafo visual
puede encoger o agrandar el mundo casi gratis (mide la escala del mapa
resultante vs el real); con IMU, el acelerómetro fija la escala (δp está en
metros DE VERDAD). Es la razón de que el monocular+IMU sea la combinación
estándar de los teléfonos (ARKit/ARCore). Ojo: bearing-only necesita buena
inicialización — parte del ejercicio es descubrir cuánta.

## 5. Los jacobianos analíticos del factor (difícil — el rito de paso)

El grafo usa diferencias finitas (decisión didáctica del nivel 12). Deriva a
mano los jacobianos del residuo de IMU respecto a los 12 estados implicados
y sustitúyelos.

**Objetivo**: el test de equivalencia del nivel 18 contra la versión
numérica (tolerancia ~1e-5) y la medición de velocidad. Derivar estos
jacobianos en 3D, sobre la variedad, es literalmente el apéndice del paper
de Forster — hacerlo en 2D primero es el mejor entrenamiento que existe.
