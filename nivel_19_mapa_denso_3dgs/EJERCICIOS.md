# Ejercicios — Nivel 19

## 1. La transmitancia inclusiva (fácil, revelador)

Cambia el producto acumulado EXCLUSIVO por el inclusivo (usa `T` en vez de
`T_excl` en el blending) y corre el examen.

**Objetivo**: explica qué se rompe y por qué. Con el inclusivo, cada
gaussiana se auto-ocluye (su propio (1−a) la atenúa) — el render palidece y,
peor, el gradiente de la opacidad queda sesgado a la baja. Un error de UN
índice en un cumprod, invisible a simple vista, cazable con el acto 3.

## 2. DILATION: el anti-aliasing que no es opcional (fácil)

Pon `DILATION = 0.0` y sobreajusta. Después pruébalo con 3.0.

**Objetivo**: con 0, algunas gaussianas colapsan por debajo de un píxel
(determinantes ~0, inversas inestables, picos que parpadean entre centros de
píxel); con 3.0, todo sale borroso y el PSNR se estanca. Mide ambos. El 0.3
del código es un compromiso medido, no un número mágico.

## 3. Densificar y podar (medio — lo que 3DGS real hace y aquí falta)

Nuestro sobreajuste usa n fijo de gaussianas. 3DGS real DENSIFICA (clona/
divide las gaussianas con gradiente grande) y PODA (elimina opacidad ~0)
cada ~100 iteraciones. Implementa una versión mínima: cada 200 iters, borra
las gaussianas con sigmoid(opacidad) < 0.02 y duplica (con jitter) las del
5% superior de ‖∂L/∂μ‖.

**Objetivo**: misma calidad con menos gaussianas, o más calidad con las
mismas. Mide PSNR y n a lo largo del entrenamiento. (Ojo: al cambiar el
número de parámetros hay que reconstruir el optimizador — el estado de Adam
no sobrevive.)

## 4. La pose también es diferenciable (medio)

El render es diferenciable respecto a T_w_c. Congela un mapa sobreajustado,
perturba la pose (unos grados y centímetros) y recupérala por descenso de
gradiente sobre la pérdida fotométrica.

**Objetivo**: tracking FOTOMÉTRICO — sin keypoints, sin matching, sin PnP.
Es la mitad "SLAM" de los sistemas 3DGS-SLAM (MonoGS): la misma pieza que el
`update_poses` del padre. Reporta el error de pose vs iteraciones y el radio
de convergencia (¿desde cuánta perturbación ya no vuelve?).

## 5. Datos reales: el techo fotométrico (difícil; GPU)

Siembra gaussianas desde el mapa métrico del nivel 15 (fr1_desk: posiciones
y colores de los puntos, escala inicial ~1.5% de su profundidad) y optimiza
contra 5-10 keyframes reales con sus poses.

**Objetivo**: la lección 41 del padre en tus manos: el PSNR se estanca
~20-22 dB y NO es falta de capacidad — es motion blur + auto-exposición
(cada keyframe vio la escena con otra fotometría). Compara el PSNR de un
keyframe nítido vs uno borroso. El padre llegó a 21.0 dB full-res (paridad
con el estado del arte en esa secuencia) usando gsplat vía Docker — su
lección 40 documenta el porqué del contenedor (el mangling nvcc↔MSVC).
