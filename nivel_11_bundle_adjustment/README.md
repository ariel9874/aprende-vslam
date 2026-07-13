# Nivel 11 — Bundle Adjustment: el corazón matemático

**Objetivo**: dejar de creer en el mapa. Hasta ahora el PnP (nivel 10)
estimaba la pose dando los puntos 3D por buenos, y la triangulación daba las
poses por buenas. El **bundle adjustment** admite que TODO tiene ruido y
busca la configuración conjunta — poses Y puntos a la vez — que mejor explica
todas las observaciones.

Es el algoritmo que hace funcionar la fotogrametría, el SLAM y la
reconstrucción 3D moderna. Lo vas a implementar entero: jacobianos
analíticos, Levenberg-Marquardt, kernel de Huber y complemento de Schur.

## Teoría mínima (la derivación completa está en `bundle_adjustment.py`)

**El problema.**

```
argmin_{T_k, X_p}   Σ_(k,p)   ρ( ‖ π(K, T_k⁻¹·X_p) − u_kp ‖² )
```

La observación `(k,p)` dice: el keyframe `k` vio el punto `p` en el píxel
`u_kp`. Se optimizan las poses `T_k` **y** los puntos `X_p` simultáneamente.

**Optimizar en una variedad.** Una rotación no se puede "sumar": `R + δ` ya
no es una rotación. Se optimiza en el espacio TANGENTE (el álgebra de Lie,
un espacio vectorial de verdad) y se vuelve al grupo con la exponencial:
`T ← T · Exp(δ)`, con `δ = [ρ, ω] ∈ R⁶`. Aquí es donde por fin aparecen
Exp/Log — cuando hacen falta, no antes.

**El complemento de Schur (por qué el BA escala).** El sistema normal tiene
estructura de flecha:

```
H = [ B   E  ]        B: cámaras ↔ cámaras   (6K × 6K)
    [ Eᵀ  C  ]        C: puntos ↔ puntos     (¡DIAGONAL POR BLOQUES 3×3!)
```

`C` es diagonal por bloques porque **dos puntos nunca comparten un residuo**:
los puntos sólo se hablan a través de las cámaras. Eso permite eliminar los
puntos casi gratis y resolver sólo por las cámaras:

```
(B − E·C⁻¹·Eᵀ)·δ_c = g_c − E·C⁻¹·g_p     ← sistema reducido, ¡sólo 6K × 6K!
δ_p = C_p⁻¹·(g_p − E_pᵀ·δ_c)             ← retro-sustitución, punto a punto
```

Con 5 keyframes y 2000 puntos, un sistema de ~6000 incógnitas colapsa a uno
de **30×30**. g2o, Ceres y GTSAM viven de este truco.

## Las tres lecciones que este nivel te ahorra (las tres se MIDEN aquí)

1. **El gauge monocular tiene 7 grados de libertad, no 6.** Fijar UNA cámara
   ancla rotación y traslación... pero la ESCALA sigue libre: escalar toda la
   escena deja los residuos idénticos. El optimizador deriva dentro de esa
   familia. Hay que fijar **DOS** cámaras. El script te lo demuestra con
   números (la firma es inconfundible: el error relativo sale idéntico en
   poses y en puntos).

2. **Los agujeros de costo enseñan a hacer trampa.** Si omites del costo las
   observaciones de puntos que caen DETRÁS de la cámara, el optimizador
   aprende a esconder los puntos incómodos ahí para borrar su residuo (el
   repo padre midió puntos volando a 15 000 unidades). Y una penalización
   tímida no basta: la trampa se refina. La penalización debe superar el
   residuo físicamente posible más grande.

3. **Un punto con UNA sola observación se desliza por su rayo.** Su bloque
   `C_p` tiene rango 2: no hay nada que fije su profundidad. Hay que
   excluirlo del BA (o el BA empeora el mapa en vez de mejorarlo).

## Cómo correr

```bash
pip install -r requirements.txt
python 11_bundle_adjustment.py     # los 4 experimentos, con sus numeros
python verificacion.py             # el examen del nivel
```

No hace falta dataset: el nivel corre sobre geometría sintética exacta
(sabemos la verdad, así que podemos medir el error de verdad).

## Qué debes poder explicar al terminar

- Qué minimiza el BA y por qué es distinto del PnP y de la triangulación.
- Por qué se optimiza en el tangente y qué es `T ← T·Exp(δ)`.
- Por qué `C` es diagonal por bloques y qué compra el Schur.
- Los 7 grados de libertad del gauge, y qué firma deja fijar sólo uno.
