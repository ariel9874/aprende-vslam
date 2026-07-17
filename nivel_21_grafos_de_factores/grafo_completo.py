"""GRAFO COMPLETO: poses Y landmarks como variables. El patrón oro.

Es el bundle adjustment del nivel 11 en su forma de grafo de factores, con
todo a la vista. Variables Θ = {p_0..p_N, l_0..l_M}; factores:

    prior     : e = p_0 − p̂_0                    (ancla el gauge)
    odometría : e = between(p_i, p_j) ⊖ ẑ_ij     (cadena)
    landmark  : e = observar(p_i, l_j) − ẑ       (la retícula que cose todo)

─── La matemática: MAP = mínimos cuadrados no lineales ───────────────────────
Cada medida es gaussiana: p(z|Θ) ∝ exp(−½‖e(Θ)‖²_Λ). Maximizar el producto
de factores = minimizar la suma de Mahalanobis:

    Θ* = argmin Σ_k e_k(Θ)ᵀ Λ_k e_k(Θ)

Gauss-Newton: linealizar e(Θ+δ) ≈ e + J·δ y resolver las ecuaciones normales

    H·δ = −g      con  H = Σ JᵀΛJ  (la MATRIZ DE INFORMACIÓN),  g = Σ JᵀΛe

H es DISPERSA: cada factor toca 1-2 variables. En este nivel la ensamblamos
densa (el problema es chico) pero GUARDAMOS su patrón: dibujarlo es entender
por qué GTSAM/g2o existen (la eliminación explota exactamente ese patrón).

Los jacobianos, analíticos (en SE(2) caben en tres líneas cada uno — son la
versión de juguete de los del nivel 11):

    ∂between/∂p_i = [[−c, −s,  m_y], [ s, −c, −m_x], [0, 0, −1]]
    ∂between/∂p_j = [[ c,  s, 0], [−s,  c, 0], [0, 0, 1]]
    ∂obs/∂p       = las dos primeras filas de ∂between/∂p_i
    ∂obs/∂l       = R(θ)ᵀ

con (m_x, m_y) = between/obs ya calculado (reaparece en su propia derivada).
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from mundo import (SIGMA_OBS, SIGMA_ODO_TH, SIGMA_ODO_XY, between, componer,
                   envolver, observar)

# Información de cada tipo de factor: Λ = diag(1/σ²).
INFO_ODO = np.diag([1 / SIGMA_ODO_XY ** 2, 1 / SIGMA_ODO_XY ** 2,
                    1 / SIGMA_ODO_TH ** 2])
INFO_OBS = np.eye(2) / SIGMA_OBS ** 2
INFO_PRIOR = np.eye(3) * 1e8


def _J_between(p_i, p_j):
    """Jacobianos analíticos del factor de odometría (teoría arriba)."""
    c, s = np.cos(p_i[2]), np.sin(p_i[2])
    m = between(p_i, p_j)
    J_i = np.array([[-c, -s, m[1]], [s, -c, -m[0]], [0, 0, -1.0]])
    J_j = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])
    return J_i, J_j


def _J_obs(p, l):
    """Jacobianos analíticos del factor de observación."""
    c, s = np.cos(p[2]), np.sin(p[2])
    o = observar(p, l)
    J_p = np.array([[-c, -s, o[1]], [s, -c, -o[0]]])
    J_l = np.array([[c, s], [-s, c]])
    return J_p, J_l


def linearizar(mundo: dict, poses: np.ndarray, lms: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray, float]:
    """(H, g, costo) del sistema completo en el punto (poses, lms).

    Orden del estado: [p_0..p_N (3 c/u) | l_0..l_M (2 c/u)]. Esta función es
    LA pieza compartida del nivel: la ventana marginalizada (Schur) opera
    sobre exactamente esta H — la marginalización es álgebra sobre la
    matriz de información, no un algoritmo nuevo.
    """
    N, M = len(poses), len(lms)
    n = 3 * N + 2 * M
    H = np.zeros((n, n))
    g = np.zeros(n)
    costo = 0.0

    def suma(idx_a, J_a, idx_b, J_b, e, Lam):
        nonlocal costo
        costo += float(e @ Lam @ e)
        bloques = [(idx_a, J_a)] + ([(idx_b, J_b)] if J_b is not None else [])
        for ia, Ja in bloques:
            g[ia[0]:ia[1]] += Ja.T @ Lam @ e
            for ib, Jb in bloques:
                H[ia[0]:ia[1], ib[0]:ib[1]] += Ja.T @ Lam @ Jb

    # prior (gauge): SIN el, H es singular — mover TODO el mundo junto no
    # cambia ningun residuo relativo (el gauge de los niveles 11/12).
    e0 = np.array([poses[0][0], poses[0][1], envolver(poses[0][2])])
    suma((0, 3), np.eye(3), None, None, e0, INFO_PRIOR)

    for i, j, z in mundo["odo"]:
        e = between(poses[i], poses[j]) - z
        e[2] = envolver(e[2])
        J_i, J_j = _J_between(poses[i], poses[j])
        suma((3 * i, 3 * i + 3), J_i, (3 * j, 3 * j + 3), J_j, e, INFO_ODO)

    for i, j, z in mundo["obs"]:
        e = observar(poses[i], lms[j]) - z
        J_p, J_l = _J_obs(poses[i], lms[j])
        il = (3 * N + 2 * j, 3 * N + 2 * j + 2)
        suma((3 * i, 3 * i + 3), J_p, il, J_l, e, INFO_OBS)

    return H, g, costo


def optimizar(mundo: dict, iteraciones: int = 15
              ) -> Dict[str, np.ndarray]:
    """Gauss-Newton sobre el grafo completo, desde la odometría pura."""
    poses = mundo["inicial"].copy()
    # Los landmarks se inicializan desde su PRIMERA observacion (como los
    # puntos del nivel 15 desde la profundidad: invertir el modelo).
    M = len(mundo["landmarks"])
    lms = np.zeros((M, 2))
    vistos = set()
    for i, j, z in mundo["obs"]:
        if j not in vistos:
            p = poses[i]
            c, s = np.cos(p[2]), np.sin(p[2])
            lms[j] = [p[0] + c * z[0] - s * z[1], p[1] + s * z[0] + c * z[1]]
            vistos.add(j)

    N = len(poses)
    costos = []
    for _ in range(iteraciones):
        H, g, costo = linearizar(mundo, poses, lms)
        costos.append(costo)
        delta = np.linalg.solve(H + 1e-9 * np.eye(len(H)), -g)
        for i in range(N):
            poses[i, :2] += delta[3 * i:3 * i + 2]
            poses[i, 2] = envolver(poses[i, 2] + delta[3 * i + 2])
        lms += delta[3 * N:].reshape(-1, 2)
        if np.linalg.norm(delta) < 1e-10:
            break
    H, _, costo = linearizar(mundo, poses, lms)
    costos.append(costo)
    return {"poses": poses, "landmarks": lms, "H": H, "costos": costos}
