# Ejercicios — Nivel 05

## 1. El barrido de k (fácil)

Corre tu Harris con k en {0.02, 0.04, 0.08, 0.15} y dibuja el top-50 de cada
uno sobre la imagen.

**Objetivo**: describe qué pierde y qué gana el detector al subir k (pista:
mira la fórmula R = λ1·λ2 − k·(λ1+λ2)² — ¿a quién castiga el término de la
traza?). ¿En qué k empiezan a desaparecer las esquinas "débiles pero reales"?

## 2. Esquinas saturadas (fácil, conecta con el nivel 00)

Satura artificialmente una zona de la imagen (`gray[100:200, 300:450] = 255`)
y vuelve a detectar con ORB.

**Objetivo**: cuenta los keypoints dentro de la zona antes y después. El
gradiente de una zona saturada es cero: sin gradiente no hay tensor de
estructura, no hay esquina — por eso el auto-exposure del nivel 00 importa.

## 3. El mito de la invarianza a escala (medio)

El script midió que SIFT re-detecta solo ~25% de sus puntos a media
resolución (¡peor que GFTT!). Investiga: separa los keypoints de SIFT por
su atributo `kp.size` (escala) y mide la repetibilidad de los FINOS
(size < mediana) vs los GRUESOS (size >= mediana) por separado.

**Objetivo**: demuestra con números que los que mueren son los finos — sus
estructuras ya no están resueltas a 0.5x. Moraleja: "invariante a escala"
significa que el DESCRIPTOR empareja entre escalas distintas, no que la
detección sobreviva a perder la mitad de los píxeles.

## 4. Repetibilidad bajo movimiento REAL (medio)

En vez de rotar sintéticamente, usa dos frames reales separados por 1 s
(gap de 30 en `rgb/`). Ya no conoces la transformación exacta, así que mide
un proxy: % de keypoints del frame A cuyo descriptor tiene un match con
ratio < 0.75 en el frame B (adelanto del nivel 06).

**Objetivo**: tabla ORB vs SIFT con gap 10 / 30 / 90 frames. ¿Cuál degrada
más lento con el paso del tiempo (= cambio de punto de vista)?
