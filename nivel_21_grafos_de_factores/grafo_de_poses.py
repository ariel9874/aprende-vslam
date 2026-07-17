"""GRAFO DE POSES: los landmarks nunca entran al estado.

Es el backend del nivel 12, visto ahora como una DECISIÓN sobre la forma del
grafo: cada re-observación de landmarks compartidos entre dos poses lejanas
se COMPRIME en un solo factor relativo pose-a-pose (el "cierre de bucle"),
y los landmarks desaparecen del estado.

─── Qué se gana y qué se pierde ──────────────────────────────────────────────
GANA: el estado pasa de 3N+2M a 3N variables, y los factores de decenas de
observaciones a unos pocos relativos — es lo que hace al grafo de poses tan
barato que se corre en caliente al cerrar un bucle (nivel 13).

PIERDE: la retícula. En el grafo completo, DOS poses cualesquiera que vieron
el mismo landmark están acopladas a través de él (la covisibilidad ES la
estructura de H — nivel 11). Al comprimir, solo sobreviven los acoplos que
alguien decidió convertir en factor: los pares (i, j) elegidos como "bucle".
Toda la información restante de los landmarks se tira. La diferencia se MIDE
en el driver — y es la razón de que ORB-SLAM corra el grafo de poses en
caliente pero el BA completo como refinador (niveles 13/14).

─── Cómo nace un factor de bucle aquí ────────────────────────────────────────
Igual que en el curso (matching + verificación geométrica), en versión 2D:
si las poses i y j (lejanas en el tiempo) comparten ≥ 3 landmarks, sus dos
conjuntos de observaciones {z_i}, {z_j} son la MISMA constelación vista desde
dos sitios. El ajuste rígido 2D (Procrustes/Umeyama, nivel 08) entre ambas
constelaciones da la transformación relativa ẑ_ij — el factor.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from grafo_completo import INFO_ODO, _J_between
from mundo import between, envolver

GAP_BUCLE = 25          # poses de separacion minima (el filtro temporal)
MIN_COMUNES = 3         # landmarks compartidos para intentar el ajuste
CADENCIA = 8            # un candidato de bucle cada tantas poses (costo)
INFO_BUCLE = np.diag([400.0, 400.0, 4000.0])


def _ajuste_rigido_2d(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """La pose relativa (dx, dy, dθ) que lleva los puntos B sobre los A —
    el Umeyama del nivel 08, en 2D y sin escala."""
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    Ac, Bc = A - ca, B - cb
    U, _, Vt = np.linalg.svd(Bc.T @ Ac)
    S = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ S @ U.T
    t = ca - R @ cb
    return np.array([t[0], t[1], np.arctan2(R[1, 0], R[0, 0])])


def factores_de_bucle(mundo: dict) -> List[Tuple[int, int, np.ndarray]]:
    """Los cierres de bucle sintetizados desde las re-observaciones."""
    vistos: Dict[int, Dict[int, np.ndarray]] = {}
    for i, j, z in mundo["obs"]:
        vistos.setdefault(i, {})[j] = z
    bucles = []
    ultimo = -10 ** 9
    for j in sorted(vistos):
        if j - ultimo < CADENCIA:
            continue
        for i in sorted(vistos):
            if j - i < GAP_BUCLE:
                break
            comunes = sorted(set(vistos[i]) & set(vistos[j]))
            if len(comunes) < MIN_COMUNES:
                continue
            A = np.array([vistos[i][k] for k in comunes])
            B = np.array([vistos[j][k] for k in comunes])
            bucles.append((i, j, _ajuste_rigido_2d(A, B)))
            ultimo = j
            break
    return bucles


def optimizar(mundo: dict, con_bucles: bool = True, iteraciones: int = 15
              ) -> Dict[str, np.ndarray]:
    """Gauss-Newton SOLO sobre poses: odometría + bucles (sin landmarks)."""
    poses = mundo["inicial"].copy()
    N = len(poses)
    factores = [(i, j, z, INFO_ODO) for i, j, z in mundo["odo"]]
    bucles = factores_de_bucle(mundo) if con_bucles else []
    factores += [(i, j, z, INFO_BUCLE) for i, j, z in bucles]

    for _ in range(iteraciones):
        H = np.zeros((3 * N, 3 * N))
        g = np.zeros(3 * N)
        H[:3, :3] += np.eye(3) * 1e8            # prior: el gauge
        g[:3] += np.eye(3) * 1e8 @ np.array(
            [poses[0][0], poses[0][1], envolver(poses[0][2])])
        for i, j, z, Lam in factores:
            e = between(poses[i], poses[j]) - z
            e[2] = envolver(e[2])
            J_i, J_j = _J_between(poses[i], poses[j])
            for (a, Ja) in ((i, J_i), (j, J_j)):
                g[3 * a:3 * a + 3] += Ja.T @ Lam @ e
                for (b, Jb) in ((i, J_i), (j, J_j)):
                    H[3 * a:3 * a + 3, 3 * b:3 * b + 3] += Ja.T @ Lam @ Jb
        delta = np.linalg.solve(H + 1e-9 * np.eye(3 * N), -g)
        for i in range(N):
            poses[i, :2] += delta[3 * i:3 * i + 2]
            poses[i, 2] = envolver(poses[i, 2] + delta[3 * i + 2])
        if np.linalg.norm(delta) < 1e-10:
            break
    return {"poses": poses, "bucles": bucles}
