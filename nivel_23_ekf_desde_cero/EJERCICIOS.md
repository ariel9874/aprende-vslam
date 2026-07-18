# Ejercicios — Nivel 23

## 1. La perilla Q (fácil — la experiencia real de tunear)

En `kalman_lineal.py`, barre `SIGMA_V` del filtro (no el del simulador:
deja el mundo quieto y miente solo en el modelo) de 0.01 a 1.0.

**Objetivo**: las dos patologías clásicas, medidas. Q muy chica: el filtro
se vuelve terco (cree demasiado en su modelo, ignora al sensor, se queda
ATRÁS de los cambios de velocidad). Q muy grande: nervioso (persigue al
ruido del sensor). Grafica RMSE contra Q: hay un valle — y su fondo está
en el Q VERDADERO del simulador. Tunear un filtro real es buscar ese valle
sin conocer la verdad.

## 2. NEES: ¿el filtro dice la verdad sobre sí mismo? (medio)

El error del acto 4 lo mides tú contra el ground truth; el filtro además
CREE saber su error (su P). El NEES lo confronta: ε = eᵀ·P⁻¹·e sobre la
pose, promediado en el tiempo, debe rondar la dimensión del estado (aquí
2 o 3, chi-cuadrado).

**Objetivo**: mide el NEES del ESKF completo (¿consistente?) y el del ESKF
sin sesgo estimado (ε explota: el filtro está seguro Y equivocado — la
definición operativa de inconsistencia). Es el instrumento con el que la
literatura mide filtros, y explica el 184 cm del acto 4 mejor que ninguna
prosa.

## 3. El jacobiano del reset (medio — la letra pequeña de Solà)

Tras inyectar el error al nominal, reseteamos δx a cero pero dejamos P
igual. La letra pequeña: el reset también transforma P (jacobiano G del
reset, ecs. 285-286 del paper de Solà; en 2D casi identidad, en 3D no).

**Objetivo**: implementa G y mide la diferencia en este mundo (spoiler:
pequeña — explica POR QUÉ: ¿de qué tamaño es δθ cuando se resetea?). Saber
cuándo una corrección fina importa y cuándo no es criterio de ingeniería,
no de fe.

## 4. El EKF iterado (difícil)

En el acto 3, la H se evalúa en el estado PREDICHO. El IEKF re-evalúa H en
el estado ya corregido y repite la corrección (2-3 vueltas).

**Objetivo**: con el ruido nominal casi no cambia nada (mide). Sube
`SIGMA_R` y `SIGMA_B` 5× y vuelve a medir: la brecha EKF vs IEKF se abre —
iterar es re-linealizar, o sea, un pasito hacia el smoother. Sitúalo en el
espectro del nivel 21: EKF → IEKF → ventana → grafo completo.

## 5. Devolverle el mapa... y quitarle el rango (difícil — el puente)

Dos movimientos sobre el acto 4: (a) ya tiene el mapa en el estado — ahora
cambia la observación a SOLO rumbo (bearing-only, la visión monocular de
juguete del ejercicio 4 del nivel 22); (b) compara contra el grafo del 22
con el mismo cambio.

**Objetivo**: el ESKF monocular+IMU sigue anclado en metros (el
acelerómetro trae la escala); mide cuánto empeora la inicialización de
landmarks con una sola dirección (pista: necesitarás dos vistas o una
prior de rango). Estás reconstruyendo, en 2D, la decisión de diseño
exacta de ARKit.
