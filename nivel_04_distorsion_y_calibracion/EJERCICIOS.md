# Ejercicios — Nivel 04

## 1. ¿Cuántas vistas hacen falta? (fácil)

Calibra usando solo 3, 5, 8 y las 14 vistas, y tabula: RMS de reproyección
y error de los intrínsecos contra la verdad.

**Objetivo**: la curva de saturación. ¿Desde cuántas vistas deja de mejorar?
Cuidado con la trampa: el RMS BAJA con pocas vistas (menos datos, más fácil
ajustarlos) mientras el error contra la verdad SUBE — sobreajuste puro, y la
razón de que el RMS solo no baste para juzgar una calibración.

## 2. El tablero plano y frontal no calibra (medio — el error clásico)

Modifica `pose_tablero` para que TODAS las vistas sean casi frontales
(rotaciones < 3°) y calibra.

**Objetivo**: mira cómo se disparan fx/fy y cx/cy. Explica por qué: con el
tablero paralelo al sensor, un tablero grande y lejos produce la MISMA
imagen que uno pequeño y cerca — la focal y la distancia se confunden
(degeneración). Las vistas INCLINADAS rompen esa ambigüedad. Por eso todo
tutorial insiste en "mueve e inclina el tablero".

## 3. Separar radial de tangencial (medio)

El script mide que una recta que pasa por el centro óptico apenas se curva
(0.22 px). Verifica la explicación: recalcula esa curvatura poniendo
`p1 = p2 = 0` en la verdad (solo distorsión radial).

**Objetivo**: debe caer a ~0 exacto. Acabas de aislar experimentalmente el
término tangencial. ¿Y qué le pasa a la recta del borde (u=15) al quitar el
tangencial?

## 4. Calibra TU cámara (medio — el ejercicio de verdad)

Imprime un tablero (o ábrelo en una tablet), graba 15-20 fotos con tu webcam
moviéndolo e inclinándolo, y corre el MISMO `04_calibracion.py` sobre tus
imágenes (sáltate la comparación con la verdad: no la tienes).

**Objetivo**: tu K y tu distorsión, con RMS < 0.5 px. Compara tu fx con la
resolución de tu cámara: fx ≈ ancho_px / (2·tan(FOV_horizontal/2)) — estima
tu campo de visión y compruébalo con una cinta métrica y una pared.

## 5. Lo que se paga por no rectificar (difícil — puente al nivel 07)

Toma la geometría epipolar del nivel 07 sobre datos reales de TUM y corre
la estimación de la pose CON y SIN rectificación previa.

**Objetivo**: cuantifica el sesgo en grados. Recuerda el orden de magnitud
que mediste aquí: la lente mueve los puntos hasta 23 px, y la geometría
epipolar umbraliza los inliers a 1 px. No rectificar no es "un poco de
error": es alimentar al estimador con puntos que ninguna geometría rígida
puede explicar.
