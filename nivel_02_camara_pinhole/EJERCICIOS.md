# Ejercicios — Nivel 02

## 1. Caras ocultas (medio)

Tu cubo alámbrico dibuja las 12 aristas siempre, incluso las de la cara
trasera. Implementa el descarte más simple: una cara es visible si su
normal (hacia afuera) apunta hacia la cámara (`normal · centro_de_cara < 0`
en el marco de la cámara).

**Objetivo**: el cubo gira mostrando solo 6-9 aristas según el ángulo — tu
primer z-culling. ¿Por qué basta el signo del producto punto?

## 2. La ilusión del gran angular (fácil)

Renderiza DOS cubos: uno a 2 m y otro a 8 m, con fx=260 y con fx=1040
(ajusta la distancia global para que el cubo cercano ocupe lo mismo en
ambas).

**Objetivo**: reproduce el efecto "vértigo" del cine (dolly zoom): con
gran angular el cubo lejano se ve diminuto; con teleobjetivo, casi igual
que el cercano. Explícalo con u − cx = fx·X/Z: la perspectiva la fija la
POSICIÓN (los Z relativos), no la focal.

## 3. El punto principal descentrado (fácil)

Renderiza con cx=200 en vez de 318.6 y compara.

**Objetivo**: describe qué se movió y qué NO (pista: nada gira ni se
deforma — ¿por qué un cx malo es tan difícil de notar a simple vista y tan
dañino para la geometría de los niveles 07+?).

## 4. Tu propio objeto (medio)

Sustituye el cubo por otro modelo alámbrico hecho a mano (una pirámide, una
casa de 10 vértices) y anímalo.

**Objetivo**: video de tu objeto girando. Si te animas: proyecta también la
SOMBRA sobre un plano (los vértices con y = constante) — es otra proyección,
pero paralela. ¿Qué fila de K "desaparece" en una proyección paralela?
