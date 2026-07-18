"""El grafo VIO: visión + factores de IMU preintegrados (o coast, para medir).

El estado por keyframe crece: ya no basta la pose. El factor de IMU habla de
VELOCIDADES (integra aceleración) y de SESGOS (hay que descubrirlos), así
que cada nodo es

    [x, y, θ, v_x, v_y, b_g]        (6 valores; + landmarks de 2)

Ese crecimiento del estado es EL precio de la IMU — y la razón de que los
VIO reales vivan en ventanas marginalizadas (nivel 21): más estado por nodo,
más presión por olvidar el pasado.

Factores:
    IMU      : el preintegrado (preintegracion.py), con la corrección de
               primer orden del sesgo dentro del residuo.
    random walk del sesgo: b_g(j) ≈ b_g(i) — el sesgo puede derivar despacio.
    visión   : observar(pose, landmark) − ẑ (el mismo del nivel 21).
    COAST    : la alternativa a la IMU para comparar — un factor DÉBIL de
               velocidad constante (p_j ≈ p_i + v_i·Δt, v_j ≈ v_i, θ_j ≈ θ_i).
               Es el `_coast` del nivel 13 convertido en factor: honesto
               sobre lo que asume, y ciego a los giros.

─── Decisión didáctica: jacobianos NUMÉRICOS ─────────────────────────────────
Como en el grafo de poses del nivel 12: cada factor toca pocas variables y
el problema es chico; las diferencias finitas cuestan nada y no permiten
equivocarse de convención. (Los analíticos del residuo de IMU son un clásico
de los papers de VIO — y el ejercicio 5.)
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from mundo_imu import R2, SIGMA_OBS, envolver, observar
from preintegracion import corregir_por_bias, preintegrar

D_NODO = 6
EPS = 1e-6

# informaciones (diagonales; la propagacion completa de covarianza del
# preintegrado es el ejercicio 2)
INFO_IMU = np.diag([250.0, 250.0, 2500.0, 40.0, 40.0])   # [p(2), θ, v(2)]
INFO_BIAS_RW = np.array([[1e6]])
INFO_OBS = np.eye(2) / SIGMA_OBS ** 2
INFO_COAST = np.diag([2.0, 2.0, 0.5, 4.0, 4.0])


def _residuo_imu(xi: np.ndarray, xj: np.ndarray, pre: Dict) -> np.ndarray:
    """[r_p(2), r_θ, r_v(2)] — la teoría de preintegracion.py."""
    p_i, th_i, v_i, b_i = xi[:2], xi[2], xi[3:5], xi[5]
    p_j, th_j, v_j = xj[:2], xj[2], xj[3:5]
    d_th, d_v, d_p = corregir_por_bias(pre, b_i)
    Rt = R2(th_i).T
    r_p = Rt @ (p_j - p_i - v_i * pre["dt_total"]) - d_p
    r_th = envolver(th_j - th_i - d_th)
    r_v = Rt @ (v_j - v_i) - d_v
    return np.array([r_p[0], r_p[1], float(r_th), r_v[0], r_v[1]])


def _residuo_coast(xi: np.ndarray, xj: np.ndarray, dt: float) -> np.ndarray:
    r_p = xj[:2] - xi[:2] - xi[3:5] * dt
    r_th = envolver(xj[2] - xi[2])
    r_v = xj[3:5] - xi[3:5]
    return np.array([r_p[0], r_p[1], float(r_th), r_v[0], r_v[1]])


def optimizar(mundo: Dict, usar_imu: bool = True, estimar_bias: bool = True,
              iteraciones: int = 20) -> Dict:
    """Gauss-Newton sobre nodos [p, θ, v, b_g] + landmarks, con factores de
    IMU (o de coast) entre keyframes consecutivos y visión donde la haya."""
    N = len(mundo["gt"])
    M = len(mundo["landmarks"])
    dt = mundo["dt_kf"]

    # preintegrar cada tramo UNA vez (con sesgo de referencia 0)
    pres = [preintegrar(t, 0.0) for t in mundo["tramos"]]

    # inicializacion: dead-reckoning de la propia IMU (¡sin mirar el GT!);
    # para coast, quieto en el origen no funciona — usa la misma init.
    from preintegracion import integrar_muerto
    poses0 = integrar_muerto(mundo["tramos"], np.zeros(3),
                             np.array([1.0, 0.0]))
    x = np.zeros(D_NODO * N + 2 * M)
    for i in range(N):
        x[D_NODO * i:D_NODO * i + 3] = poses0[i]
        x[D_NODO * i + 3:D_NODO * i + 5] = R2(poses0[i][2]) @ [1.0, 0.0]

    # landmarks: desde su primera observacion
    vistos = set()
    for i, j, z in mundo["obs"]:
        if j not in vistos:
            p = x[D_NODO * i:D_NODO * i + 3]
            x[D_NODO * N + 2 * j:D_NODO * N + 2 * j + 2] = \
                p[:2] + R2(p[2]) @ z
            vistos.add(j)

    # ── la lista de factores: (indices de variables, funcion, informacion) ──
    factores = []
    n_i = lambda i: list(range(D_NODO * i, D_NODO * i + D_NODO))
    l_j = lambda j: [D_NODO * N + 2 * j, D_NODO * N + 2 * j + 1]

    # prior del nodo 0 (gauge + velocidad inicial + sesgo inicial libre-ish)
    factores.append((n_i(0),
                     lambda v: v[:5] - np.array([0, 0, 0, 1.0, 0]),
                     np.diag([1e8, 1e8, 1e8, 1e4, 1e4])))

    for i in range(N - 1):
        pre = pres[i]
        if usar_imu:
            factores.append((n_i(i) + n_i(i + 1),
                             (lambda pre_: lambda v: _residuo_imu(
                                 v[:6], v[6:], pre_))(pre), INFO_IMU))
        else:
            factores.append((n_i(i) + n_i(i + 1),
                             lambda v: _residuo_coast(v[:6], v[6:], dt),
                             INFO_COAST))
        # el sesgo: random walk si se estima; clavado a 0 si no
        if estimar_bias and usar_imu:
            factores.append(([D_NODO * i + 5, D_NODO * (i + 1) + 5],
                             lambda v: np.array([v[1] - v[0]]), INFO_BIAS_RW))
        else:
            factores.append(([D_NODO * i + 5],
                             lambda v: np.array([v[0]]),
                             np.array([[1e8]])))

    for i, j, z in mundo["obs"]:
        factores.append((n_i(i)[:3] + l_j(j),
                         (lambda z_: lambda v: observar(v[:3], v[3:]) - z_)(z),
                         INFO_OBS))

    # ── Gauss-Newton con jacobianos numericos ────────────────────────────────
    n = len(x)
    for _ in range(iteraciones):
        H = np.zeros((n, n))
        g = np.zeros(n)
        for idx, fn, Lam in factores:
            v0 = x[idx]
            e = fn(v0)
            J = np.zeros((len(e), len(idx)))
            for k in range(len(idx)):
                v = v0.copy()
                v[k] += EPS
                J[:, k] = (fn(v) - e) / EPS
            ii = np.array(idx)
            g[ii] += J.T @ Lam @ e
            H[np.ix_(ii, ii)] += J.T @ Lam @ J
        delta = np.linalg.solve(H + 1e-6 * np.eye(n), -g)
        x += delta
        for i in range(N):
            x[D_NODO * i + 2] = envolver(x[D_NODO * i + 2])
        if np.linalg.norm(delta) < 1e-8:
            break

    poses = np.stack([x[D_NODO * i:D_NODO * i + 3] for i in range(N)])
    biases = np.array([x[D_NODO * i + 5] for i in range(N)])
    lms = x[D_NODO * N:].reshape(-1, 2)
    return {"poses": poses, "biases": biases, "landmarks": lms}
