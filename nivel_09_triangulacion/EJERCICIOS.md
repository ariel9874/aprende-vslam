# Ejercicios — Nivel 09

## 1. La DLT a mano (medio — el ejercicio del nivel)

No uses `cv2.triangulatePoints`: construye tú la matriz A (4×4) apilando las
dos ecuaciones de cada vista y resuelve con `np.linalg.svd`, quedándote con
la última fila de Vᵀ.

**Objetivo**: tus puntos deben coincidir con los de OpenCV a < 1e-9 m. (Es
el patrón del curso: implementar, verificar contra la referencia.)

## 2. Triangular con poses ESTIMADAS (medio — el puente al nivel 10)

El script usa las poses del ground truth para aislar la triangulación. Ahora
usa las que estima tu código del nivel 07 (E + recoverPose), con `||t||=1`.

**Objetivo**: el mapa sale con la FORMA correcta pero en otra escala (la
del gauge `||t||=1`). Mide el factor: divide la profundidad mediana de tu
mapa entre la del mapa con GT. Ese número es la escala arbitraria del
monocular — y es exactamente lo que el nivel 10 va a heredar y propagar en
vez de re-inventar en cada frame.

## 3. Curva ruido → error (medio)

Barre el ruido de píxel (0.1, 0.25, 0.5, 1, 2 px) para dos baselines (par
0→2 y par 0→30) y grafica el desplazamiento 3D mediano.

**Objetivo**: dos rectas en escala log-log, y comprueba que la pendiente es
1 (el error 3D es LINEAL en el ruido de píxel) mientras que la separación
vertical entre ellas la fija el baseline. Confirma `dZ = ε·Z²/(f·B)`.

## 4. Puntos en el infinito (medio)

Baja `MIN_PARALLAX_DEG` a 0.0 y triangula el par 0→1. Exporta el PLY.

**Objetivo**: abre la nube y describe dónde acaban los puntos sin paralaje
(pista: mira el rango de z — algunos se van a cientos de metros, y otros
DETRÁS de la cámara). Ese es el mapa basura que el filtro te ahorra, y la
razón de la regla del repo padre: *nunca crear mapa desde una pose
incierta*.

## 5. La incertidumbre no es una esfera (difícil)

Para un punto concreto, perturba su píxel en un círculo (32 direcciones,
0.5 px) y triangula cada vez. Grafica la nube de los 32 puntos 3D
resultantes en el plano XZ.

**Objetivo**: verás un ELIPSOIDE muy alargado en la dirección del rayo, no
una esfera: la incertidumbre lateral es de milímetros y la de profundidad
de decímetros. Por eso el bundle adjustment (nivel 11) NO puede tratar los
puntos como si tuvieran error isótropo — y por eso un punto con una sola
observación se desliza libremente por su rayo (la lección 7 del repo padre).
