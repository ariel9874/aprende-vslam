# Nivel 19 — Mapa denso 3DGS · electiva

**Objetivo**: escribir un rasterizador de **3D Gaussian Splatting
diferenciable** desde cero — y con él, el "renderiza y compara": un mapa que
se ajusta por descenso de gradiente hasta re-sintetizar lo que la cámara vio.

## La idea en una línea

Todo el curso ha construido mapas DISPERSOS (puntos 3D): sirven para
localizar, no para re-sintetizar. 3DGS cambia la primitiva: el mapa es un
conjunto de **gaussianas 3D** (media, covarianza, opacidad, color) y el
render las proyecta y mezcla en una imagen. Como TODO el pipeline es
diferenciable, el mapa se optimiza igual que el BA del nivel 11 — pero el
residuo es fotométrico (la imagen entera), no la reproyección de puntos.

## Las cuatro piezas del rasterizador (`render_gaussianas.py`, ~160 líneas)

1. **La primitiva**: Σ = R·S·Sᵀ·Rᵀ — definida positiva POR CONSTRUCCIÓN, por
   eso se optimizan rotación (cuaternión libre, se normaliza dentro) y
   escalas (en log), nunca Σ directa. Opacidad en logit, color en logit:
   cada parámetro vive donde el optimizador puede moverse sin restricciones.
2. **EWA splatting** (Zwicker 2001): Σ' = J·W·Σ·Wᵀ·Jᵀ. El jacobiano J es
   *exactamente el d_pi del BA del nivel 11* — la misma linealización de la
   proyección, ahora propagando covarianzas en vez de residuos.
3. **El peso por píxel**: g = exp(−½·δᵀΣ'⁻¹δ) sobre la rejilla de CENTROS
   de píxel (i + 0.5 — ver abajo por qué esa media línea importa tanto).
4. **α-blending por transmitancia** (front-to-back): C = Σ cᵢ·aᵢ·Tᵢ con
   Tᵢ = Π<sub>j<i</sub>(1−aⱼ) — un producto acumulado exclusivo, tensorial
   y diferenciable.

Es la REFERENCIA legible (como el BA NumPy): densa, O(N·H·W), en PyTorch
puro. Las gemelas de rendimiento (tiles, gsplat/CUDA) son la historia del
nivel 18 otra vez — el padre las tiene, con test de equivalencia.

## Los números (medidos con este código)

| acto del examen | medido | criterio |
|---|---|---|
| gradiente autograd vs dif. finitas | dif. máx **2.7·10⁻⁹** | < 10⁻³ |
| sobreajuste de una vista (1500 iters) | **58.0 dB** | > 30 dB |
| medio píxel entre convenciones | **29.0 dB** | < 40 dB (¡debe discrepar!) |

- El **sobreajuste a 58 dB** prueba que el rasterizador y sus gradientes
  funcionan. El techo del padre con datos REALES fue **~21 dB** (fr1_desk
  full-res, paridad con el estado del arte) — y su lección 41: el cuello no
  era la capacidad del mapa, era la FOTOMETRÍA de los datos (motion blur,
  auto-exposición — el nivel 01 te lo advirtió).
- El **medio píxel** (lección 40 del padre): el píxel i cubre [i, i+1) y su
  centro está en i+0.5. Sus dos gemelas (referencia vs gsplat) discrepaban a
  ~25 dB hasta alinear esa convención — 25 → 60 dB con UNA línea. El acto 4
  del examen reproduce la discrepancia a propósito: dos implementaciones que
  no comparten convención NO pasan un test de equivalencia (nivel 18).

## Cómo correr

```bash
pip install -r requirements.txt      # torch (con CUDA si tienes GPU)
python 19_gaussianas.py              # el sobreajuste, con gráficas (32 s GPU)
python verificacion.py               # el examen (~1 min GPU; CPU: minutos)
```

Sin dataset: todo el nivel es sintético y autocontenido. El salto a datos
reales (sembrar las gaussianas desde el mapa del nivel 15 y re-sintetizar
fr1_desk) es el ejercicio 5, con las referencias del padre.

## Qué debes poder explicar al terminar

- Por qué se optimizan R y S (y no Σ), y qué gana cada parametrización
  (log-escala, logit-opacidad).
- Qué propaga exactamente EWA — y qué tiene que ver su J con el BA.
- Por qué la transmitancia se calcula como producto EXCLUSIVO y qué pasaría
  con el gradiente si fuera inclusivo.
- Por qué verificar el gradiente contra diferencias finitas es EL test (¿qué
  falla silenciosamente sin él?).
- La lección del medio píxel — y por qué 58 dB en sintético no contradice
  los 21 dB del padre en real.
