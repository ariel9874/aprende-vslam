"""VENTANA DESLIZANTE con MARGINALIZACIÓN: olvidar sin tirar.

El grafo completo crece sin límite; un sistema en línea optimiza solo una
VENTANA de variables recientes. La pregunta del archivo: al sacar el pasado
de la ventana, ¿qué haces con su información? Dos respuestas, medidas:

  - CORTAR: borrar los factores viejos y ya. La información se TIRA.
  - MARGINALIZAR: comprimir el pasado en un PRIOR gaussiano sobre la
    frontera de la ventana. La información se CONSERVA (exacta, en el punto
    de linealización).

─── La matemática: marginalizar = complemento de Schur ───────────────────────
Sobre las ecuaciones normales H·δ = −g, parte el estado en A (lo que sale)
y B (lo que queda):

    [H_AA  H_AB] [δ_A]   [−g_A]
    [H_BA  H_BB] [δ_B] = [−g_B]

Eliminar δ_A da el sistema REDUCIDO sobre B:

    (H_BB − H_BA·H_AA⁻¹·H_AB)·δ_B = −(g_B − H_BA·H_AA⁻¹·g_A)

Es EXACTAMENTE el mismo Schur del BA (nivel 11) — allí marginalizaba puntos
dentro de una iteración; aquí marginaliza EL PASADO para siempre. Sobre el
sistema LINEAL es exacto a precisión de máquina (el examen lo verifica).

─── Los dos precios (los que pagan los VIO reales) ───────────────────────────
1. FILL-IN: H era dispersa; el bloque reducido H_BB' es DENSO — marginalizar
   un landmark acopla entre sí a TODAS las poses que lo vieron. El examen lo
   cuenta en bloques no-nulos.
2. PUNTO DE LINEALIZACIÓN CONGELADO: el prior quedó evaluado donde se
   marginalizó; la ventana se sigue re-linealizando, el prior no puede.
   (Es el problema que FEJ — First Estimate Jacobians — gestiona en OKVIS/
   MSCKF; aquí se hereda tal cual y se mide su costo.)
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from grafo_completo import linearizar
from mundo import envolver


def indices_a_marginalizar(mundo: dict, corte: int, N: int, M: int):
    """El estado se parte en A (sale) y B (queda): salen las poses < corte y
    los landmarks cuyas observaciones viven TODAS en el pasado marginalizado."""
    lm_futuro = {j for i, j, _ in mundo["obs"] if i >= corte}
    idx_A, idx_B = [], []
    for i in range(N):
        destino = idx_A if i < corte else idx_B
        destino.extend(range(3 * i, 3 * i + 3))
    for j in range(M):
        destino = idx_B if j in lm_futuro else idx_A
        destino.extend(range(3 * N + 2 * j, 3 * N + 2 * j + 2))
    return np.array(idx_A), np.array(idx_B), lm_futuro


def schur(H: np.ndarray, g: np.ndarray, idx_A: np.ndarray, idx_B: np.ndarray):
    """El complemento de Schur: (H', g') del sistema reducido sobre B."""
    H_AA = H[np.ix_(idx_A, idx_A)]
    H_AB = H[np.ix_(idx_A, idx_B)]
    H_BB = H[np.ix_(idx_B, idx_B)]
    K = np.linalg.solve(H_AA + 1e-12 * np.eye(len(idx_A)), H_AB)
    return H_BB - H_AB.T @ K, g[idx_B] - H_AB.T @ np.linalg.solve(
        H_AA + 1e-12 * np.eye(len(idx_A)), g[idx_A])


def optimizar_ventana(mundo: dict, corte: int, marginalizar: bool = True,
                      iteraciones: int = 15) -> Dict:
    """La ventana [corte..N) optimizada con el pasado marginalizado (o
    simplemente cortado, para medir la diferencia).

    El prior se construye UNA vez, linealizado en la odometría inicial (el
    punto de linealización congelado de la cabecera), y la ventana se
    re-linealiza libremente por encima.
    """
    poses = mundo["inicial"].copy()
    N, M = len(poses), len(mundo["landmarks"])

    # landmarks iniciales desde su primera observacion (como grafo_completo)
    lms = np.zeros((M, 2))
    vistos = set()
    for i, j, z in mundo["obs"]:
        if j not in vistos:
            p = poses[i]
            c, s = np.cos(p[2]), np.sin(p[2])
            lms[j] = [p[0] + c * z[0] - s * z[1], p[1] + s * z[0] + c * z[1]]
            vistos.add(j)

    idx_A, idx_B, lm_futuro = indices_a_marginalizar(mundo, corte, N, M)
    lin0 = np.concatenate([poses.reshape(-1), lms.reshape(-1)])   # punto congelado

    if marginalizar:
        H, g, _ = linearizar(mundo, poses, lms)
        H_pr, g_pr = schur(H, g, idx_A, idx_B)
    else:
        # CORTAR: sin prior del pasado; solo un ancla debil en la primera
        # pose de la ventana (donde la dejo la odometria) para fijar gauge.
        H_pr = np.zeros((len(idx_B), len(idx_B)))
        H_pr[:3, :3] = np.eye(3) * 1e6
        g_pr = np.zeros(len(idx_B))

    # El sub-mundo de la ventana: solo factores integramente dentro de B.
    sub = {"odo": [(i, j, z) for i, j, z in mundo["odo"] if i >= corte],
           "obs": [(i, j, z) for i, j, z in mundo["obs"]
                   if i >= corte and j in lm_futuro]}

    for _ in range(iteraciones):
        Hf, gf, _ = _linearizar_ventana(sub, poses, lms, corte, N, M,
                                        idx_B, lm_futuro)
        # el prior gaussiano: e = x_B − lin0_B  ->  H_pr·(δ + desvio) = ...
        desvio = np.concatenate([poses.reshape(-1),
                                 lms.reshape(-1)])[idx_B] - lin0[idx_B]
        Hf += H_pr
        gf += g_pr + H_pr @ desvio
        delta = np.linalg.solve(Hf + 1e-9 * np.eye(len(Hf)), -gf)
        _aplicar(delta, poses, lms, corte, N, lm_futuro)
        if np.linalg.norm(delta) < 1e-10:
            break
    return {"poses": poses, "landmarks": lms, "H_prior": H_pr,
            "idx_B": idx_B}


def _linearizar_ventana(sub, poses, lms, corte, N, M, idx_B, lm_futuro):
    """La misma linealización del grafo completo, restringida a la ventana
    (reusa los jacobianos analíticos de grafo_completo vía `linearizar`
    sobre un mundo recortado, y extrae el bloque B)."""
    H, g, costo = linearizar({"odo": sub["odo"], "obs": sub["obs"]},
                             poses, lms)
    # quitar el prior de gauge que `linearizar` pone en la pose 0 GLOBAL
    # (esta fuera de la ventana): su bloque no pertenece a B, asi que basta
    # con extraer el bloque B.
    return H[np.ix_(idx_B, idx_B)], g[idx_B], costo


def _aplicar(delta, poses, lms, corte, N, lm_futuro):
    k = 0
    for i in range(corte, N):
        poses[i, :2] += delta[k:k + 2]
        poses[i, 2] = envolver(poses[i, 2] + delta[k + 2])
        k += 3
    for j in sorted(lm_futuro):
        lms[j] += delta[k:k + 2]
        k += 2
