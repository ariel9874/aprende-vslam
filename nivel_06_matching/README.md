# Nivel 06 — Matching: emparejar y filtrar mentiras

**Objetivo**: emparejar los descriptores de dos imágenes y aprender la
primera línea de defensa contra los emparejamientos falsos: el ratio test
de Lowe. Sin geometría todavía (eso es el nivel 07) — aquí se mide cuánto
se puede limpiar solo con los descriptores.

## Teoría mínima

**Comparar descriptores.** ORB produce vectores binarios de 256 bits; la
distancia natural es **Hamming**: cuántos bits difieren (un XOR y contar
unos — por eso es tan rápido). Para descriptores flotantes (SIFT) se usa la
distancia euclídea.

**El vecino más cercano miente.** Para cada descriptor del frame A, el más
parecido del frame B es SU MATCH... aunque el punto físico ni siquiera sea
visible en B: el vecino más cercano siempre existe. ¿Cómo saber si es un
match de verdad o el menos malo de los impostores?

**El ratio test de Lowe (2004).** Compara el mejor candidato contra el
SEGUNDO mejor:

```
aceptar si   dist(mejor) < 0.75 · dist(segundo)
```

La intuición: si el punto es realmente re-observado, su descriptor tiene UN
doble claro en B y el segundo candidato queda lejos. Si el punto no está (o
la textura es repetitiva — un teclado, una persiana), el mejor y el segundo
son impostores igual de malos y sus distancias son parecidas → se descarta.
No mira la imagen: mira la AMBIGÜEDAD.

**Lo que el ratio no puede.** Un match puede ser inambiguo y aun así
geométricamente imposible. La limpieza definitiva la hace RANSAC con la
restricción epipolar (nivel 07). En este nivel medimos el residuo: cuántos
matches "sospechosos" (desplazamiento incoherente con la mayoría) sobreviven
al ratio.

## Cómo correr

```bash
pip install -r requirements.txt
python descarga_datos.py           # TUM fr1_xyz (el mismo de niveles 00/05)
python 06_matching.py              # dos frames separados 1 s
python verificacion.py             # el examen del nivel
```

¿Ya tienes fr1_xyz? Pasa `--root <carpeta>`. Otros knobs: `--gap` (frames
de separación) y `--ratio`.

Resultados en `salida/`: `matches_antes_despues.png` y `curva_ratio.png`.

## Qué debes poder explicar al terminar

- Por qué Hamming para binarios y qué cuesta calcularla.
- Qué ambigüedad mide el ratio test y por qué 0.75.
- Qué tipo de matches falsos NO puede eliminar el ratio (motivación del
  nivel 07).
