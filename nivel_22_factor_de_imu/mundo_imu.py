"""El mundo del nivel: un vehículo 2D con IMU, visión... y un APAGÓN.

A diferencia del nivel 21 (odometría de ruedas), este vehículo es "tipo
dron": NO tiene odometría. Sus dos sentidos son la VISIÓN (landmarks, solo
en los keyframes, a ~4 Hz) y una IMU (giróscopo + acelerómetro, a 100 Hz,
con ruido y con SESGO). Y en el tramo más incómodo del circuito — una
CURVA — la visión se apaga: la ráfaga de blur del nivel 17, en versión de
juguete controlada.

─── La matemática: el uniciclo y su IMU ──────────────────────────────────────
El vehículo avanza con rapidez v(t) y gira con velocidad angular ω(t):

    ẋ = v·cosθ,   ẏ = v·sinθ,   θ̇ = ω

Su IMU en el marco del CUERPO mide:

    giróscopo     : ω + b_g + ruido          (b_g: el SESGO, constante aquí)
    acelerómetro  : [v̇, v·ω] + ruido         (tangencial, CENTRÍPETA)

La componente centrípeta v·ω es la gracia del 2D: en las curvas el
acelerómetro "siente" el giro — exactamente la información que el apagón
visual necesita. (Lo que el 2D NO puede enseñar: la gravedad como referencia
de actitud — eso es 3D puro y queda anotado en el README.)

Se simula a 100 Hz con el modelo exacto y se toman KEYFRAMES cada 25
muestras (4 Hz). El ground truth existe a ambas frecuencias; los backends
solo ven medidas.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# ── SE(2), duplicado del nivel 21 a propósito (regla 2 del curso) ────────────


def envolver(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def observar(p: np.ndarray, l: np.ndarray) -> np.ndarray:
    """El landmark l visto desde la pose p (marco del robot)."""
    c, s = np.cos(p[2]), np.sin(p[2])
    dx, dy = l[0] - p[0], l[1] - p[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy])


def R2(th: float) -> np.ndarray:
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


# ── parametros del sensor ────────────────────────────────────────────────────

DT = 0.01                  # 100 Hz
POR_KF = 25                # un keyframe cada 25 muestras (4 Hz)
ALCANCE = 3.5
SIGMA_OBS = 0.05
SIGMA_GYRO = 0.02          # rad/s por muestra
SIGMA_ACC = 0.10           # m/s^2 por muestra
BIAS_GYRO_REAL = 0.03      # rad/s — el sesgo que el grafo debera DESCUBRIR


def generar(semilla: int = 11, con_apagon: bool = True,
            sin_ruido: bool = False) -> Dict:
    """El circuito del nivel 21 (8x5 m, dos vueltas) recorrido en continuo:
    v = 1 m/s en las rectas, y en cada esquina una curva de 90 grados a
    ω = 1.2 rad/s (sin frenar: curva de verdad, con centrípeta).

    El APAGÓN: la visión se apaga durante la SEGUNDA curva de la primera
    vuelta — unos 5 s que incluyen el giro completo. El coast por velocidad
    constante no puede saber que ahí se giró; el giróscopo lo midió.
    """
    rng = np.random.default_rng(semilla)
    # sin_ruido: IMU perfecta y sin sesgo — SOLO para el test de que la
    # preintegracion reproduce la verdad (el examen, acto 1).
    s_gyro = 0.0 if sin_ruido else SIGMA_GYRO
    s_acc = 0.0 if sin_ruido else SIGMA_ACC
    s_obs = 0.0 if sin_ruido else SIGMA_OBS
    bias = 0.0 if sin_ruido else BIAS_GYRO_REAL
    v_recta, w_curva = 1.0, 1.2
    t_curva = (np.pi / 2) / w_curva

    # perfil (v, ω) muestra a muestra: [recta 8m, curva, recta 5m, curva] x2
    perfil: List[Tuple[float, float]] = []
    for _ in range(2):
        for largo in (8.0, 5.0, 8.0, 5.0):
            perfil += [(v_recta, 0.0)] * int(round(largo / v_recta / DT))
            perfil += [(v_recta, w_curva)] * int(round(t_curva / DT))

    # integrar el modelo EXACTO (la verdad continua)
    x = np.zeros(3)
    vel = np.array([v_recta, 0.0])
    gt_fino = [x.copy()]
    imu = []                       # (omega_medida, acc_medida) por muestra
    for k, (v, w) in enumerate(perfil):
        a_cuerpo = np.array([0.0, v * w])          # v̇=0: solo centripeta
        imu.append((w + bias + rng.normal(0, s_gyro) if s_gyro else w + bias,
                    a_cuerpo + rng.normal(0, s_acc, 2) if s_acc
                    else a_cuerpo.copy()))
        th = x[2] + w * DT * 0.5                    # punto medio (exacto aqui)
        x = np.array([x[0] + v * np.cos(th) * DT,
                      x[1] + v * np.sin(th) * DT,
                      envolver(x[2] + w * DT)])
        gt_fino.append(x.copy())
    gt_fino = np.array(gt_fino)

    # keyframes: pose Y velocidad de la verdad
    idx_kf = np.arange(0, len(gt_fino), POR_KF)
    gt = gt_fino[idx_kf]
    vel_gt = []
    for i in idx_kf:
        j = min(i, len(perfil) - 1)
        v, _ = perfil[j]
        vel_gt.append(R2(gt_fino[j][2]) @ np.array([v, 0.0]))
    vel_gt = np.array(vel_gt)

    # tramos de IMU entre keyframes consecutivos
    tramos = []
    for a, b in zip(idx_kf[:-1], idx_kf[1:]):
        tramos.append([imu[k] for k in range(a, b)])

    # landmarks (los del 21) y observaciones en keyframes
    landmarks = np.array([
        [1.5, 1.2], [4.0, -1.0], [6.5, 1.2], [9.0, 2.5], [6.5, 3.8],
        [4.0, 6.0], [1.5, 3.8], [-1.0, 2.5], [2.5, -0.8], [8.8, 0.2],
        [8.8, 4.8], [5.5, 5.8], [0.2, 5.6], [-0.8, 0.2]])

    # el apagon: la 2a curva de la vuelta 1 (+-2.5 s alrededor de su centro)
    n_kf = len(gt)
    t_kf = idx_kf * DT
    centro_apagon = (8.0 / v_recta) + t_curva + (5.0 / v_recta) + t_curva / 2
    apagados = set()
    if con_apagon:
        apagados = {i for i in range(n_kf)
                    if abs(t_kf[i] - centro_apagon) < 2.5}

    obs = []
    for i, pg in enumerate(gt):
        if i in apagados:
            continue
        for j, l in enumerate(landmarks):
            if np.hypot(l[0] - pg[0], l[1] - pg[1]) < ALCANCE:
                z = observar(pg, l)
                if s_obs:
                    z = z + rng.normal(0, s_obs, 2)
                obs.append((i, j, z))

    return {"gt": gt, "vel_gt": vel_gt, "gt_fino": gt_fino,
            "landmarks": landmarks, "obs": obs, "tramos": tramos,
            "apagados": sorted(apagados), "dt_kf": POR_KF * DT}


def rmse_xy(tray: np.ndarray, gt: np.ndarray) -> float:
    d = tray[:, :2] - gt[:len(tray), :2]
    return float(np.sqrt((d ** 2).sum(axis=1).mean()))
