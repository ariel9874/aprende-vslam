# Ejercicios — Nivel 12

## 1. El jacobiano analítico (medio — el que evitamos)

El grafo usa jacobianos NUMÉRICOS (12 evaluaciones por arista). Implementa el
analítico para SE(3): necesitas el jacobiano adjunto,
`∂/∂δ_i Log(T̂⁻¹T_i⁻¹T_j) ≈ −J_r⁻¹(e)·Adj(T_j⁻¹T_i)` y su pareja para j.

**Objetivo**: comprueba que coincide con el numérico a 1e-6 y cronometra
ambos con 30, 100 y 300 nodos. Decide tú si el ahorro justifica la
complejidad a esta escala (el repo padre decidió que NO en el grafo, y que SÍ
en el BA — donde hay miles de observaciones, no decenas de aristas).

## 2. ¿Dónde se reparte el error? (fácil, muy visual)

Grafica el error por nodo (‖pose_estimada − verdad‖ contra el índice del
nodo) ANTES y DESPUÉS del cierre de bucle.

**Objetivo**: verás que antes el error crece monótonamente (la deriva) y
después queda repartido en una curva suave con máximo en el medio. El grafo
no "arregla el último nodo": redistribuye. Explica por qué el nodo 15 acaba
con más error que el 29.

## 3. La información importa (medio)

Cambia la información de las aristas de odometría de `1e2·I` a `1e4·I` (o sea:
"confío mucho en mi odometría") y vuelve a cerrar el bucle.

**Objetivo**: el grafo ya no puede repartir el error (las odometrías son
"inviolables") y el bucle apenas corrige. Mide el ATE. Ahora hazlo al revés
(odometría floja, `1e0`). La información NO es un detalle de implementación:
es la declaración de en quién confías, y determina dónde se coloca el error.

## 4. El bucle parcial (medio)

Cierra el bucle sólo entre los nodos 20 y 5 (no entre el último y el
primero). Es decir: revisitas una zona intermedia, no el origen.

**Objetivo**: ¿qué parte de la trayectoria se corrige y cuál no? Los nodos
21-29 quedan "colgando" después del bucle. Este es el caso REAL del repo
padre (su lección 23): en secuencias largas todos los bucles son LOCALES,
y por eso la escala sigue derivando entre segmentos lejanos.

## 5. Sim(3) en el sitio equivocado (difícil — el error simétrico)

Repite el experimento de Strasdat pero con odometría MÉTRICA correcta (sin
deriva de escala, como daría un sensor RGB-D del nivel 15). Cierra el bucle
con un grafo Sim(3) — cuya medida de bucle mide, inevitablemente con ruido,
una escala relativa ligeramente distinta de 1.

**Objetivo**: mide el ATE y la escala del mapa tras varios bucles seguidos
(cierra 5 bucles simulados en cadena). Verás la escala DERIVAR: cada
corrección Sim(3) re-escala el mapa, el siguiente bucle mide la discrepancia
que el anterior creó, y la "corrige" otra vez. Es exactamente el bug que el
repo padre midió (escala 2.09, ATE 22 cm) antes de darse cuenta de que en
RGB-D el bucle debe ser SE(3). La moraleja del nivel, del revés.
