"""PREINTEGRACIÓN de IMU: mil muestras, UN factor.

El problema que resuelve: la IMU mide a 100-1000 Hz y el grafo tiene nodos a
4-30 Hz. Meter una variable por muestra revienta el grafo; integrar "hacia
adelante" desde la pose i exigiría RE-integrar todo cada vez que el
optimizador mueva la pose i (que es en cada iteración).

─── La matemática: integrar RELATIVO al marco i (Lupton; Forster et al.) ─────
Los deltas se acumulan en el marco del nodo i — independientes de dónde esté
i en el mundo:

    δθ_{k+1} = δθ_k + (ω_k − b_g)·dt
    δv_{k+1} = δv_k + R(δθ_k)·a_k·dt
    δp_{k+1} = δp_k + δv_k·dt + ½·R(δθ_k)·a_k·dt²

El factor entre los nodos i, j compara estos deltas con lo que los ESTADOS
implican:

    r_θ = (θ_j − θ_i) − δθ̂
    r_v = R(θ_i)ᵀ·(v_j − v_i) − δv̂
    r_p = R(θ_i)ᵀ·(p_j − p_i − v_i·Δt) − δp̂

─── El truco del sesgo (la contribución de Forster) ──────────────────────────
Los deltas dependen del sesgo b_g con el que se integró. Si el optimizador
actualiza b_g, ¿re-integrar mil muestras? NO: durante la preintegración se
acumulan también los JACOBIANOS de los deltas respecto al sesgo,

    ∂δθ/∂b_g = −Σdt        ∂δv/∂b_g, ∂δp/∂b_g  (regla de la cadena abajo)

y el factor corrige a PRIMER ORDEN sin re-integrar:

    δθ̂(b) ≈ δθ̂(b̄) + (∂δθ/∂b_g)·(b − b̄)

El error de esa aproximación es O((b−b̄)²) — el examen lo mide. Este truco
es lo que hace tratable el VIO en tiempo real: el factor se fabrica UNA vez.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from mundo_imu import DT, R2


def preintegrar(tramo: List[Tuple[float, np.ndarray]],
                bias_gyro: float = 0.0) -> Dict:
    """Los deltas (δθ, δv, δp) del tramo, integrados con el sesgo `bias_gyro`,
    MÁS sus jacobianos respecto al sesgo (el truco de la cabecera)."""
    d_th = 0.0
    d_v = np.zeros(2)
    d_p = np.zeros(2)
    J_th = 0.0                    # d(δθ)/d(b_g)
    J_v = np.zeros(2)             # d(δv)/d(b_g)
    J_p = np.zeros(2)             # d(δp)/d(b_g)
    for w_med, a_med in tramo:
        w_hat = w_med - bias_gyro
        # PUNTO MEDIO: rotar con el angulo a mitad del paso. Con Euler puro
        # (rotar con el angulo al INICIO) la discretizacion acumulaba ~20 cm
        # en los 62 s del circuito — medido construyendo este nivel. Mismo
        # costo, un orden mas de precision.
        th_m = d_th + 0.5 * w_hat * DT
        R = R2(th_m)
        # dR/dθ · a  (la derivada de la rotacion, aplicada a la aceleracion)
        dRa = np.array([-np.sin(th_m) * a_med[0] - np.cos(th_m) * a_med[1],
                        np.cos(th_m) * a_med[0] - np.sin(th_m) * a_med[1]])
        # jacobianos ANTES de actualizar (regla de la cadena en orden);
        # d(th_m)/d(b_g) = J_th − dt/2  (el punto medio tambien siente el bias)
        J_m = J_th - 0.5 * DT
        J_p = J_p + J_v * DT + 0.5 * dRa * J_m * DT ** 2
        J_v = J_v + dRa * J_m * DT
        J_th = J_th - DT
        # los deltas
        d_p = d_p + d_v * DT + 0.5 * (R @ a_med) * DT ** 2
        d_v = d_v + (R @ a_med) * DT
        d_th = d_th + w_hat * DT
    return {"d_th": d_th, "d_v": d_v, "d_p": d_p,
            "J_th": J_th, "J_v": J_v, "J_p": J_p,
            "bias_ref": bias_gyro, "dt_total": len(tramo) * DT}


def corregir_por_bias(pre: Dict, bias: float) -> Tuple[float, np.ndarray,
                                                       np.ndarray]:
    """Los deltas ajustados a un sesgo NUEVO, a primer orden, SIN re-integrar
    (δ(b) ≈ δ(b̄) + J·(b − b̄))."""
    db = bias - pre["bias_ref"]
    return (pre["d_th"] + pre["J_th"] * db,
            pre["d_v"] + pre["J_v"] * db,
            pre["d_p"] + pre["J_p"] * db)


def integrar_muerto(tramos, pose0: np.ndarray, vel0: np.ndarray,
                    bias: float = 0.0) -> np.ndarray:
    """DEAD RECKONING: encadenar los deltas sin ningún otro sensor. La
    posición integra DOS veces el ruido del acelerómetro (y el sesgo del
    giro rota la velocidad entera): la deriva crece como t^(3/2)-t^2, no
    linealmente — por eso nadie navega solo con IMU barata."""
    poses = [pose0.copy()]
    p, th, v = pose0[:2].copy(), float(pose0[2]), vel0.copy()
    for tramo in tramos:
        pre = preintegrar(tramo, bias)
        R = R2(th)
        p = p + v * pre["dt_total"] + R @ pre["d_p"]
        v = v + R @ pre["d_v"]
        th = th + pre["d_th"]
        poses.append(np.array([p[0], p[1], th]))
    return np.array(poses)
