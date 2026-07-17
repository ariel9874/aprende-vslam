# Ejercicios — Nivel 17

## 1. DISK: el tercer contendiente (estrella)

`kornia` (que ya instalaste con lightglue) trae DISK: un extractor entrenado
con gradiente de política — la recompensa son los matches CORRECTOS tras
emparejar, así que optimiza el objetivo final del pipeline y no un proxy de
"esquinidad". Escribe `ExtractorDISK` en features.py (la referencia del repo
padre: `kornia.feature.DISK.from_pretrained("depth")`; espera 3 canales —
replica el gris) y repite la tabla de pares del examen.

**Objetivo**: la tabla con tres filas (ORB, SuperPoint, DISK). ¿DISK con
LightGlue de SuperPoint funciona? (Pista: no — LightGlue se entrena POR
extractor: `LightGlue(features="disk")`. Los pesos del matcher y del
descriptor son un matrimonio.)

## 2. El ratio de Lowe, como experimento (fácil, muy instructivo)

Barre `ratio_float` de 0.70 a 0.95 y, para un par nítido↔borroso fijo, mide:
matches supervivientes y E-inliers de cada umbral.

**Objetivo**: la curva completa. Verás el codo: con 0.75 sobreviven ~decenas
(y la reloc del nivel se moría de hambre: votos máx 20 con umbral 40); con
0.90, cientos. Explica POR QUÉ con la concentración de la medida: en 256
dims, la distancia al vecino correcto y al incorrecto se parecen cada vez
más — el cociente pierde contraste. El umbral es una propiedad del ESPACIO,
no una constante mágica.

## 3. La cadena hambrienta (medio)

Comenta el bloque de re-abastecimiento de `_puente` (el `_guided_match` con
la pose fresca) y corre superpoint sobre los primeros 150 frames, imprimiendo
cuántos pares con pid le llegan al puente en cada frame.

**Objetivo**: reproduce la agonía medida al construir el nivel: 33 → 27 →
22 → 15 pares y cadena muerta en 4 frames. Con el re-abastecimiento, la
cadena se sostiene (y aun así la ráfaga dura la mata — compara DÓNDE muere
cada versión). Un mecanismo de supervivencia sin fuente de reposición solo
pospone la muerte.

## 4. La paridad aburrida (medio — la otra mitad de la lección 29)

Corre superpoint sobre fr2_xyz (el dataset del nivel 14, `--root` y
`--max-frames 600`) y compara con los números ORB del nivel 14 (0 perdidos,
0.8 cm).

**Objetivo**: en una secuencia amable el deep NO aporta nada — y cuesta 3×
más. La lección 29 completa: SuperPoint pagaba SOLO en las fr1 handheld.
Elegir frontend es ingeniería de presupuesto, no de moda: ¿qué fracción de
TUS datos se parece a fr1_desk?

## 5. La init tramposa, reproducida (fácil)

Baja `MIN_INIT_POINTS` de 100 a 15 (la vara del nivel 14) y corre superpoint.

**Objetivo**: ve al sistema nacer muerto — la init "válida" de ~21 puntos
desde un par de rotación casi pura (flujo 26 px, baseline ~0) y el tracking
coasteando desde el frame 7. Después explica por qué ORB nunca tropezó con
esta trampa en tres niveles (14, 15, 16) y SuperPoint la pisó en el primer
intento. Moraleja para siempre: mejorar un componente EXPONE los supuestos
silenciosos de los demás.
