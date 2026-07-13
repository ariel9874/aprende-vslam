# Ejercicios — Nivel 06

## 1. Cross-check vs ratio (fácil)

Implementa la verificación cruzada: aceptar el match A→B solo si el mejor
match de vuelta B→A cae en el mismo keypoint (`cv2.BFMatcher(...,
crossCheck=True)` lo hace por ti; impleméntalo tú con dos `knnMatch`).

**Objetivo**: tabla comparando ratio 0.75, cross-check, y AMBOS: número de
matches y % de sospechosos. ¿Cuál es más estricto? ¿Se complementan?

## 2. El metro que se encoge (medio — la anomalía que viste)

La curva del script muestra ~27% de "sospechosos" con ratio 0.50, MÁS que
con 0.75. Investiga si es basura real o un artefacto del detector de
outliers: para los supervivientes de ratio 0.50, imprime la mediana y el MAD
de los desplazamientos y compáralos con los de ratio 0.75.

**Objetivo**: demuestra con números que los matches de 0.50 son MÁS
coherentes (MAD menor) y que el % alto viene de que el umbral 3·MAD se
volvió minúsculo. Moraleja de metrología: un porcentaje sin su denominador
y su escala engaña — la misma lección que el repo padre aprendió midiendo
la trayectoria online vs final (su lección 25).

## 3. Hamming a mano (medio)

Calcula la distancia de Hamming entre dos descriptores ORB con numpy puro
(`np.unpackbits` + XOR + suma) y verifica contra la que reporta el matcher
para ese par.

**Objetivo**: coincidencia exacta, y de paso mide cuánto más lento es tu
numpy que el BFMatcher de OpenCV para los 2000×2000 pares (la razón de que
los descriptores binarios dominen el tiempo real).

## 4. La textura repetitiva, el enemigo natural (difícil)

Busca en la secuencia un frame con estructura repetitiva (persianas,
teclado, estantería) y empareja contra un frame cercano. Dibuja los matches
DESCARTADOS por el ratio (los que fallan por poco: d1/d2 entre 0.75 y 0.9).

**Objetivo**: verifica visualmente que se concentran en la textura repetida
— el ratio test funcionando exactamente como fue diseñado. Reporta qué
fracción de los descartados-por-poco cae en esas zonas.
