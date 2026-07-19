"""El mundo común del nivel: un robot 2D, un circuito, landmarks — y ruido.

DUPLICADO del nivel 21 (la regla del curso: duplicar > importar). Mismas
medidas, misma semilla — por eso los números de este nivel se comparan
directamente con la tabla del 21: si los datos cambian, la comparación miente.

Por qué 2D (SE(2)) cuando todo el curso fue 3D: la lección de este nivel es
la FORMA DEL GRAFO, no la maquinaria de la variedad. En SE(2) una pose son 3
números (x, y, θ), los jacobianos caben en tres líneas, y las matrices de
información se pueden DIBUJAR — que es como se entienden.

─── La matemática: SE(2) en cinco líneas ─────────────────────────────────────
Una pose p = (x, y, θ). Componer (p_i ⊕ d = el delta d visto desde p_i):

    x' = x + cosθ·dx − sinθ·dy
    y' = y + sinθ·dx + cosθ·dy          θ' = θ + dθ  (envuelto a (−π, π])

El delta relativo entre dos poses (la "medida" de odometría) es la inversa:
between(p_i, p_j) = R(θ_i)ᵀ·(t_j − t_i) en traslación, θ_j − θ_i en ángulo.
El envolvimiento del ángulo es el único "espacio tangente" que SE(2) exige —
la versión de juguete del Log de SE(3) del nivel 12.

─── Las medidas (los factores en potencia) ───────────────────────────────────
1. ODOMETRÍA: between(p_i, p_i+1) + ruido gaussiano. Deriva sin límite.
2. OBSERVACIONES DE LANDMARK: la posición del landmark EN EL MARCO DEL
   ROBOT (R(θ)ᵀ·(l − t)) + ruido, si está a menos de ALCANCE. Es el análogo
   2D de "el keyframe observa el punto en el píxel (u, v)".
3. El robot recorre el circuito DOS veces: la segunda vuelta re-observa los
   landmarks de la primera — ahí viven los bucles.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def envolver(a: np.ndarray | float):
    """Ángulo(s) a (−π, π] — el 'Log' de juguete de SE(2)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def componer(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    """p ⊕ d: aplicar el delta d (en el marco de p) a la pose p."""
    c, s = np.cos(p[2]), np.sin(p[2])
    return np.array([p[0] + c * d[0] - s * d[1],
                     p[1] + s * d[0] + c * d[1],
                     envolver(p[2] + d[2])])


def between(p_i: np.ndarray, p_j: np.ndarray) -> np.ndarray:
    """El delta que lleva de p_i a p_j, expresado en el marco de p_i."""
    c, s = np.cos(p_i[2]), np.sin(p_i[2])
    dx, dy = p_j[0] - p_i[0], p_j[1] - p_i[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy,
                     envolver(p_j[2] - p_i[2])])


def observar(p: np.ndarray, l: np.ndarray) -> np.ndarray:
    """El landmark l visto desde la pose p (en el marco del robot)."""
    c, s = np.cos(p[2]), np.sin(p[2])
    dx, dy = l[0] - p[0], l[1] - p[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy])


# ── el mundo ─────────────────────────────────────────────────────────────────

ALCANCE = 3.5                 # radio del sensor de landmarks (m)
SIGMA_ODO_XY = 0.06           # ruido de odometria (m por paso)
SIGMA_ODO_TH = 0.010          # (rad por paso)
SIGMA_OBS = 0.05              # ruido de la observacion de landmark (m)


def generar(semilla: int = 7, vueltas: int = 2):
    """El circuito: un rectángulo de 8x5 m recorrido `vueltas` veces, con 14
    landmarks repartidos. Devuelve un dict con TODO lo que los backends
    necesitan (y el ground truth, que ninguno debe mirar salvo para evaluar).
    """
    rng = np.random.default_rng(semilla)

    # Ground truth: pasos de ~0.5 m por el perimetro del rectangulo.
    esquinas = [(0, 0), (8, 0), (8, 5), (0, 5)]
    gt: List[np.ndarray] = []
    p = np.array([0.0, 0.0, 0.0])
    gt.append(p.copy())
    for _ in range(vueltas):
        for k in range(4):
            x0, y0 = esquinas[k]
            x1, y1 = esquinas[(k + 1) % 4]
            largo = np.hypot(x1 - x0, y1 - y0)
            n = int(round(largo / 0.5))
            rumbo = np.arctan2(y1 - y0, x1 - x0)
            for _ in range(n):
                giro = envolver(rumbo - p[2])
                p = componer(p, np.array([0.5, 0.0, float(giro)]))
                gt.append(p.copy())
    gt = np.array(gt)

    # Landmarks: junto al circuito (dentro y fuera), fijos.
    landmarks = np.array([
        [1.5, 1.2], [4.0, -1.0], [6.5, 1.2], [9.0, 2.5], [6.5, 3.8],
        [4.0, 6.0], [1.5, 3.8], [-1.0, 2.5], [2.5, -0.8], [8.8, 0.2],
        [8.8, 4.8], [5.5, 5.8], [0.2, 5.6], [-0.8, 0.2]])

    # Odometria ruidosa entre poses consecutivas.
    odo = []
    for i in range(len(gt) - 1):
        z = between(gt[i], gt[i + 1])
        z += rng.normal(0, [SIGMA_ODO_XY, SIGMA_ODO_XY, SIGMA_ODO_TH])
        odo.append((i, i + 1, z))

    # Observaciones de landmarks al alcance.
    obs: List[Tuple[int, int, np.ndarray]] = []   # (idx_pose, idx_landmark, z)
    for i, pg in enumerate(gt):
        for j, l in enumerate(landmarks):
            if np.hypot(l[0] - pg[0], l[1] - pg[1]) < ALCANCE:
                z = observar(pg, l) + rng.normal(0, SIGMA_OBS, 2)
                obs.append((i, j, z))

    # Trayectoria por odometria pura (la inicializacion de todos los grafos).
    ini = [np.array([0.0, 0.0, 0.0])]
    for _, _, z in odo:
        ini.append(componer(ini[-1], z))

    return {"gt": gt, "landmarks": landmarks, "odo": odo, "obs": obs,
            "inicial": np.array(ini)}


def rmse_xy(tray: np.ndarray, gt: np.ndarray) -> float:
    """RMSE de posición (sin alinear: el gauge lo fija el prior en la pose 0
    — todos los backends comparten el mismo ancla, comparacion justa)."""
    d = tray[:, :2] - gt[:len(tray), :2]
    return float(np.sqrt((d ** 2).sum(axis=1).mean()))
