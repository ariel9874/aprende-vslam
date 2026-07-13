# Nivel 07 — Geometría epipolar: recuperar el movimiento

**Objetivo**: recuperar el movimiento de la cámara (R, t) entre dos vistas a
partir de sus matches — la matriz esencial, RANSAC y la descomposición — y
verificarlo contra ground truth EXACTO. Aquí aparece la limitación que
define al monocular: la escala de t es inobservable.

## Teoría mínima (la derivación completa está en el script)

**La restricción epipolar.** Un punto 3D X visto por dos cámaras genera dos
rayos que, junto con la línea que une ambos centros (el baseline), son
coplanares. Esa coplanaridad se escribe:

```
x̂₂ᵀ · E · x̂₁ = 0        con  E = [t]ₓ·R   (la matriz ESENCIAL)
```

E empaqueta la rotación y la DIRECCIÓN de la traslación. Tiene 5 grados de
libertad → bastan 5 correspondencias (el solver de Nistér).

**RANSAC.** Un solo match falso arruina un ajuste por mínimos cuadrados.
RANSAC muestrea 5 matches al azar, resuelve E, cuenta cuántos matches la
satisfacen, y se queda con el consenso mayor. Los que la satisfacen son los
*inliers* — la limpieza geométrica que el ratio test (nivel 06) no podía
hacer.

**De E a (R, t) — y sus trampas.** La SVD de E da CUATRO soluciones (dos
rotaciones × dos signos de t); solo una deja los puntos triangulados DELANTE
de ambas cámaras (quiralidad). Y ||t|| = 1 por convención: la restricción es
homogénea, una cámara sola no mide metros.

**Este nivel usa datos sintéticos** con ground truth exacto en el marco de
la propia cámara: el error de R se mide sin excusas (< 1 grado o algo está
mal). Sobre datos reales (ejercicio 4) el GT viene de un sistema de captura
en OTRO marco, y comparar requiere más cuidado.

## Cómo correr

```bash
pip install -r requirements.txt
python genera_datos.py             # la secuencia sintetica (GT exacto)
python 07_geometria_epipolar.py    # E, R, t, lineas epipolares y la trampa
python verificacion.py             # el examen del nivel
```

Resultados en `salida/`: `epipolares.png` (las líneas epipolares sobre
ambas vistas) y los errores impresos contra GT.

## Los dos experimentos del nivel

1. **Recuperación de pose** (frames 0 → 6): error de rotación < 1° y de
   dirección de t < 5° contra el GT exacto.
2. **La trampa de recoverPose** (frames 0 → 1, medida originalmente en el
   repo padre — su lección 1): la sobrecarga básica de OpenCV descarta como
   inliers los puntos a más de 50× el baseline. Con paso pequeño
   (profundidad/baseline ≈ 80–280 en esta escena) los inliers COLAPSAN
   aunque la geometría sea perfecta. La sobrecarga con `distanceThresh` lo
   arregla — y verás la diferencia en números.

## Qué debes poder explicar al terminar

- De dónde sale x̂₂ᵀ·E·x̂₁ = 0 (coplanaridad) y qué es una línea epipolar.
- Por qué 5 puntos, por qué RANSAC, y qué es un inlier geométrico.
- Las 4 soluciones de la SVD y el test de quiralidad.
- Por qué ||t|| = 1 no es un bug, y qué haría falta para tener metros.
- El caso degenerado de rotación pura (sin paralaje no hay t observable).
