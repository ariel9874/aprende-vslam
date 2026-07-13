# Ejercicios — Nivel 00

Cada ejercicio tiene un objetivo medible. Hazlos sobre una copia de
`00_hola_pixeles.py` o en un script nuevo en esta carpeta.

## 1. El error del promedio (fácil)

Implementa el gris como promedio simple `(R+G+B)/3` y réstale el gris
ponderado. Guarda la diferencia como imagen (normalízala a 0–255 para verla).

**Objetivo**: encuentra EN QUÉ zonas de la imagen difieren más y explica por
qué (pista: busca objetos de color saturado; en un escritorio, los libros).

## 2. BGR vs RGB, el clásico (fácil)

Guarda la imagen intercambiando los canales B y R (`bgr[:, :, ::-1]`) y
ábrela con un visor. Describe qué colores se intercambian y por qué la piel
humana es el delator más famoso de este bug.

**Objetivo**: el píxel central impreso por tu script debe reportar el mismo
valor numérico en (B,G,R) que antes en (R,G,B).

## 3. Brillo y contraste lineales (medio)

Aplica `salida = a*gris + b` con (a=1.5, b=0) y (a=1.0, b=50). Dibuja el
histograma de las tres versiones en la misma figura.

**Objetivo**: explica con el histograma qué hace cada parámetro y CUÁNTOS
píxeles saturaste (valor 255) en cada caso — esa información ya no se
recupera, y en el nivel 05 verás que las esquinas saturadas no se detectan.

## 4. El vídeo es una pila de matrices (medio)

El dataset trae cientos de frames en `rgb/`. Carga los primeros 100, calcula
la media de gris de cada uno y grafícala contra el tiempo (el nombre del
archivo es el timestamp).

**Objetivo**: una curva de exposición de la secuencia. ¿Hay saltos? (La
cámara ajusta la exposición sola — ese "auto-exposure" perseguirá al curso
hasta el nivel 19.)
