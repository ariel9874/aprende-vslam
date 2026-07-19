"""FILTRO (EKF-SLAM): marginalizar TODO el pasado, en cada paso.

DUPLICADO del nivel 21: es el tercer competidor de la tabla final del nivel
(filtro vs iSAM vs batch — las tres respuestas a "llegó una medida nueva").

El extremo del espectro. El estado es SOLO la pose actual + los landmarks;
cada pose anterior se marginaliza en cuanto llega la siguiente. Es el mismo
Schur de la ventana llevado al límite — y con un precio nuevo: cada medida
se linealiza UNA vez, en el estado que había al llegar, y esa linealización
queda sellada en la covarianza para siempre. Un smoother re-linealiza al
converger; un filtro no puede volver atrás.

─── La matemática: predecir y corregir ───────────────────────────────────────
Estado x = [p (3) | l_1 (2) | l_2 (2) | ...], covarianza P.

PREDICCIÓN (odometría z): p ← p ⊕ z, y a primer orden

    P ← F·P·Fᵀ + G·Q·Gᵀ      F = ∂(p⊕z)/∂p (solo el bloque de la pose),
                              G = ∂(p⊕z)/∂z,  Q = diag(σ²_odo)

CORRECCIÓN (observación z del landmark j): innovación ν = z − obs(p, l_j),

    S = H·P·Hᵀ + R,   K = P·Hᵀ·S⁻¹,   x ← x ⊞ K·ν,   P ← (I − K·H)·P

H solo tiene bloques en la pose y en l_j... pero K los ACOPLA a todos: tras
unas vueltas, P es densa — el fill-in de marginalizar, otra vez, ahora como
correlación explícita entre landmarks. Esa correlación es la que cierra el
bucle "gratis": corregir un landmark viejo arrastra a los demás.

La trayectoria del filtro es ONLINE por construcción (cada pose se emite y
se marginaliza: nadie la reescribirá jamás) — compárala con la del smoother
y estás midiendo otra vez la lección 25 del curso.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from mundo import (SIGMA_OBS, SIGMA_ODO_TH, SIGMA_ODO_XY, componer, envolver,
                   observar)

Q = np.diag([SIGMA_ODO_XY ** 2, SIGMA_ODO_XY ** 2, SIGMA_ODO_TH ** 2])
R_OBS = np.eye(2) * SIGMA_OBS ** 2


def correr(mundo: dict) -> Dict[str, np.ndarray]:
    """EKF-SLAM sobre las mismas medidas que los grafos. Devuelve la
    trayectoria ONLINE (la pose filtrada en cada paso) y el mapa final."""
    obs_por_pose: Dict[int, list] = {}
    for i, j, z in mundo["obs"]:
        obs_por_pose.setdefault(i, []).append((j, z))

    x = np.zeros(3)                       # la pose actual
    P = np.zeros((3, 3))
    idx: Dict[int, int] = {}              # landmark -> offset en el estado
    lms = np.zeros(0)
    tray = [x.copy()]

    def corregir(j, z):
        nonlocal x, P, lms
        n = 3 + len(lms)
        if j not in idx:
            # INICIALIZACION del landmark: invertir el modelo de observacion
            # (l = p ⊕ z) y AGRANDAR el estado, propagando covarianza.
            c, s = np.cos(x[2]), np.sin(x[2])
            l = np.array([x[0] + c * z[0] - s * z[1],
                          x[1] + s * z[0] + c * z[1]])
            J_p = np.array([[1, 0, -s * z[0] - c * z[1]],
                            [0, 1, c * z[0] - s * z[1]]])
            J_z = np.array([[c, -s], [s, c]])
            P_nuevo = np.zeros((n + 2, n + 2))
            P_nuevo[:n, :n] = P
            P_nuevo[n:, :3] = J_p @ P[:3, :3]
            P_nuevo[:3, n:] = P_nuevo[n:, :3].T
            P_nuevo[n:, 3:n] = J_p @ P[:3, 3:]
            P_nuevo[3:n, n:] = P_nuevo[n:, 3:n].T
            P_nuevo[n:, n:] = J_p @ P[:3, :3] @ J_p.T + J_z @ R_OBS @ J_z.T
            P = P_nuevo
            idx[j] = len(lms)
            lms = np.concatenate([lms, l])
            return
        k = 3 + idx[j]
        l = lms[idx[j]:idx[j] + 2]
        nu = z - observar(x, l)
        c, s = np.cos(x[2]), np.sin(x[2])
        o = observar(x, l)
        H = np.zeros((2, 3 + len(lms)))
        H[:, :3] = np.array([[-c, -s, o[1]], [s, -c, -o[0]]])
        H[:, k:k + 2] = np.array([[c, s], [-s, c]])
        S = H @ P @ H.T + R_OBS
        K = P @ H.T @ np.linalg.inv(S)
        dx = K @ nu
        x[:2] += dx[:2]
        x[2] = envolver(x[2] + dx[2])
        lms += dx[3:]
        P = (np.eye(len(P)) - K @ H) @ P

    # las observaciones de la pose 0, antes del primer paso
    for j, z in obs_por_pose.get(0, []):
        corregir(j, z)

    for i, _, z in mundo["odo"]:
        # PREDICCION
        c, s = np.cos(x[2]), np.sin(x[2])
        F = np.array([[1, 0, -s * z[0] - c * z[1]],
                      [0, 1, c * z[0] - s * z[1]],
                      [0, 0, 1.0]])
        G = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        x = componer(x, z)
        P[:3, :3] = F @ P[:3, :3] @ F.T + G @ Q @ G.T
        if len(lms):
            P[:3, 3:] = F @ P[:3, 3:]
            P[3:, :3] = P[:3, 3:].T
        # CORRECCION con lo que esta pose ve
        for j, z_o in obs_por_pose.get(i + 1, []):
            corregir(j, z_o)
        tray.append(x.copy())

    mapa = np.full((len(mundo["landmarks"]), 2), np.nan)
    for j, k in idx.items():
        mapa[j] = lms[k:k + 2]
    return {"tray": np.array(tray), "landmarks": mapa, "P": P}
