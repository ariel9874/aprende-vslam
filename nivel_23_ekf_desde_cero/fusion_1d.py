"""ACTO 1 — Fusionar dos números: la esencia del filtro de Kalman.

Todavía no hay filtro, ni estado, ni matrices. Solo la pregunta primordial:
tienes DOS mediciones del mismo número, con incertidumbres distintas.
¿Cómo se combinan? La respuesta — la media ponderada por inversa de
varianza — contiene al filtro de Kalman ENTERO: la famosa "ganancia" K es
el peso de esa media, y nada más.

─── La matemática: la media ponderada óptima ─────────────────────────────────
Dos mediciones del mismo x: z₁ con desviación σ₁ y z₂ con σ₂ (gaussianas,
independientes). El x más probable minimiza los cuadrados PESADOS:

    J(x) = (x − z₁)²/σ₁² + (x − z₂)²/σ₂²

dJ/dx = 0 despeja la media ponderada por INFORMACIÓN (información ≡ 1/σ²):

    x̂ = (z₁/σ₁² + z₂/σ₂²) / (1/σ₁² + 1/σ₂²),      1/σ̂² = 1/σ₁² + 1/σ₂²

La segunda ecuación es la llave de todo lo recursivo: la información SUMA.
(Es la misma suma ΣJᵀΛJ de la matriz de información del nivel 21 — aquí en
escalar, donde se ve desnuda.)

Ahora reagrupa la primera alrededor de z₁ (hazlo a mano: son tres líneas):

    x̂ = z₁ + K·(z₂ − z₁),        K = σ₁² / (σ₁² + σ₂²)

...y esa ES la ecuación de corrección del filtro de Kalman. K pesa la
SORPRESA (z₂ − z₁: la "innovación"). Si lo que ya tenías es muy incierto
(σ₁ grande), K→1: créete lo nuevo. Si es muy seguro, K→0: ignóralo.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def fusionar(x1: float, s1: float, x2: float, s2: float
             ) -> Tuple[float, float]:
    """Fusiona dos estimaciones gaussianas (x1, s1) y (x2, s2).

    Devuelve (x, s) fusionados — escrito en la forma K·innovación para que
    reconozcas al filtro de Kalman cuando llegue (acto 2)."""
    K = s1 ** 2 / (s1 ** 2 + s2 ** 2)
    x = x1 + K * (x2 - x1)
    s = np.sqrt(1.0 / (1.0 / s1 ** 2 + 1.0 / s2 ** 2))
    return float(x), float(s)


def estimar_recursivo(zs: np.ndarray, sigma: float
                      ) -> Tuple[float, float, np.ndarray]:
    """Procesa N mediciones UNA POR UNA, sin guardar ninguna.

    Tras cada medición el estimado se actualiza y la medición SE TIRA:
    memoria constante. Devuelve (x, s, historial) con el historial de
    (x, s) tras cada paso — para ver a sigma caer."""
    x, s = float(zs[0]), float(sigma)
    hist = [(x, s)]
    for z in zs[1:]:
        x, s = fusionar(x, s, float(z), sigma)
        hist.append((x, s))
    return x, s, np.array(hist)


def main() -> None:
    rng = np.random.default_rng(23)
    verdad = 7.30                       # la pared esta a 7.30 m (nadie lo sabe)

    print("ACTO 1: fusionar dos numeros\n")

    # ── dos instrumentos, una pared ──────────────────────────────────────────
    s_laser, s_ultra = 0.02, 0.10       # el laser es 5x mejor
    z_laser = verdad + rng.normal(0, s_laser)
    z_ultra = verdad + rng.normal(0, s_ultra)
    x, s = fusionar(z_ultra, s_ultra, z_laser, s_laser)
    K = s_ultra ** 2 / (s_ultra ** 2 + s_laser ** 2)
    print(f"  ultrasonido: {z_ultra:.4f} m (sigma {s_ultra})")
    print(f"  laser      : {z_laser:.4f} m (sigma {s_laser})")
    print(f"  fusionados : {x:.4f} m (sigma {s:.4f}) con K = {K:.3f}")
    print(f"  K dice: creete el laser al {100*K:.0f}% -- es la media")
    print("  ponderada por 1/sigma^2, escrita como x + K*(z - x)\n")

    # ── mil mediciones, procesadas de una en una ─────────────────────────────
    N, s_z = 1000, 0.5
    zs = verdad + rng.normal(0, s_z, N)
    x_rec, s_rec, hist = estimar_recursivo(zs, s_z)
    x_batch = float(np.mean(zs))        # el batch: guardarlo TODO y promediar
    print(f"  {N} mediciones (sigma {s_z}), procesadas de una en una:")
    print(f"  recursivo {x_rec:.6f} vs batch (promedio) {x_batch:.6f}"
          f"  -> difieren {abs(x_rec - x_batch):.1e}")
    print(f"  sigma final: {s_rec:.4f} vs sigma/sqrt(N) = {s_z/np.sqrt(N):.4f}")
    print("  la informacion SUMA: N mediciones = N veces mas informacion,")
    print("  y sigma cae como 1/sqrt(N) -- la misma raiz del ruido shot")
    print("  que mediste en el nivel 01, ahora del lado del estimador")


if __name__ == "__main__":
    main()
