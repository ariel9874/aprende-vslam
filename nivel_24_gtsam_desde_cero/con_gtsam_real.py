#!/usr/bin/env python3
"""ACTO 5: el mismo mundo, resuelto con GTSAM DE VERDAD.

Corre DENTRO del contenedor (docker compose up --build): no hay wheel de
gtsam para Windows/Python 3.13 (verificado 2026-07: pip solo publica
manylinux y mac), así que el contenedor ES el entorno — el patrón del
nivel 20.

El diccionario de traducción (la tabla completa vive en el README):

    nuestro concepto (niveles 21/24)      la clase de GTSAM
    ─────────────────────────────────     ─────────────────────────────
    lista de factores no lineales         NonlinearFactorGraph
    dict clave -> valor                   Values
    clave ('x', i) / ('l', j)             symbol_shorthand.X(i) / L(j)
    factor de odometría (between)         BetweenFactorPose2
    factor de observación                 BearingRangeFactor2D
    Λ = diag(1/σ²)                        noiseModel.Diagonal.Sigmas
    prior del gauge (1e8)                 PriorFactorPose2 (σ = 1e-4)
    gauss_newton / batch                  LevenbergMarquardtOptimizer
    ISAMJuguete.paso()                    ISAM2.update()

ADAPTACIÓN HONESTA de la medida: nuestro factor de observación es el
landmark en el marco del robot (dx, dy); el factor 2D estándar de GTSAM es
BearingRangeFactor2D. Son la MISMA información en coordenadas polares:
bearing = atan2(dy, dx), range = |z|, con σ_bearing = σ_obs/range (el ruido
isotrópico de 5 cm, visto en polar). Por eso el acuerdo esperado es de
MILÍMETROS de ATE, no de épsilon de máquina — y así se verifica.
"""

from __future__ import annotations

import time

import numpy as np

import eliminacion as el
from mundo import SIGMA_OBS, SIGMA_ODO_TH, SIGMA_ODO_XY, componer, generar, rmse_xy

import gtsam
from gtsam import symbol_shorthand

X = symbol_shorthand.X
L = symbol_shorthand.L

RUIDO_PRIOR = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4, 1e-4, 1e-4]))
RUIDO_ODO = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([SIGMA_ODO_XY, SIGMA_ODO_XY, SIGMA_ODO_TH]))


def ruido_obs(rango: float):
    """El ruido isotropico de 5 cm, expresado en (bearing, range)."""
    return gtsam.noiseModel.Diagonal.Sigmas(
        np.array([SIGMA_OBS / max(rango, 1e-3), SIGMA_OBS]))


def factor_obs(i: int, j: int, z: np.ndarray):
    b = float(np.arctan2(z[1], z[0]))
    r = float(np.hypot(z[0], z[1]))
    return gtsam.BearingRangeFactor2D(X(i), L(j), gtsam.Rot2(b), r,
                                      ruido_obs(r))


def poses_de(valores: gtsam.Values, n: int) -> np.ndarray:
    return np.array([[valores.atPose2(X(i)).x(), valores.atPose2(X(i)).y(),
                      valores.atPose2(X(i)).theta()] for i in range(n)])


def main() -> int:
    m = generar()
    gt = m["gt"]
    N = len(m["inicial"])
    print(f"GTSAM {gtsam.__version__ if hasattr(gtsam, '__version__') else ''}"
          f" sobre el mundo del nivel 21: {N} poses, "
          f"{len(m['obs'])} observaciones")

    # ── nuestro numero de referencia (el batch del nivel 21/24) ──────────────
    t0 = time.perf_counter()
    nuestro = el.gauss_newton_batch(m)
    t_nuestro = time.perf_counter() - t0
    ate_nuestro = rmse_xy(nuestro["poses"], gt)

    # ── batch con GTSAM: el grafo entero + Levenberg-Marquardt ───────────────
    grafo = gtsam.NonlinearFactorGraph()
    grafo.add(gtsam.PriorFactorPose2(X(0), gtsam.Pose2(0, 0, 0), RUIDO_PRIOR))
    for i, j, z in m["odo"]:
        grafo.add(gtsam.BetweenFactorPose2(X(i), X(j),
                                           gtsam.Pose2(*z), RUIDO_ODO))
    for i, j, z in m["obs"]:
        grafo.add(factor_obs(i, j, z))

    inicial = gtsam.Values()
    for i, p in enumerate(m["inicial"]):
        inicial.insert(X(i), gtsam.Pose2(*p))
    vistos = set()
    for i, j, z in m["obs"]:                 # misma init que el nivel 21
        if j not in vistos:
            p = m["inicial"][i]
            c, s = np.cos(p[2]), np.sin(p[2])
            inicial.insert(L(j), gtsam.Point2(p[0] + c * z[0] - s * z[1],
                                              p[1] + s * z[0] + c * z[1]))
            vistos.add(j)

    t0 = time.perf_counter()
    resultado = gtsam.LevenbergMarquardtOptimizer(grafo, inicial).optimize()
    t_batch = time.perf_counter() - t0
    ate_batch = rmse_xy(poses_de(resultado, N), gt)

    # ── ISAM2: el acto 4, pero el de verdad ──────────────────────────────────
    obs_por_pose = {}
    for i, j, z in m["obs"]:
        obs_por_pose.setdefault(i, []).append((j, z))

    isam = gtsam.ISAM2()                     # defaults: umbral 0.1, skip 10
    tray = []
    en_mapa = set()
    t0 = time.perf_counter()
    for i in range(N):
        nuevo = gtsam.NonlinearFactorGraph()
        vals = gtsam.Values()
        if i == 0:
            nuevo.add(gtsam.PriorFactorPose2(X(0), gtsam.Pose2(0, 0, 0),
                                             RUIDO_PRIOR))
            p = np.zeros(3)
        else:
            z = m["odo"][i - 1][2]
            pa = isam.calculateEstimate().atPose2(X(i - 1))
            p = componer(np.array([pa.x(), pa.y(), pa.theta()]), z)
            nuevo.add(gtsam.BetweenFactorPose2(X(i - 1), X(i),
                                               gtsam.Pose2(*z), RUIDO_ODO))
        vals.insert(X(i), gtsam.Pose2(*p))
        for j, z_o in obs_por_pose.get(i, []):
            nuevo.add(factor_obs(i, j, z_o))
            if j not in en_mapa:
                c, s = np.cos(p[2]), np.sin(p[2])
                vals.insert(L(j), gtsam.Point2(p[0] + c * z_o[0] - s * z_o[1],
                                               p[1] + s * z_o[0] + c * z_o[1]))
                en_mapa.add(j)
        isam.update(nuevo, vals)
        pi = isam.calculateEstimate().atPose2(X(i))
        tray.append([pi.x(), pi.y(), pi.theta()])
    t_isam = time.perf_counter() - t0
    ate_online = rmse_xy(np.array(tray), gt)
    ate_final = rmse_xy(poses_de(isam.calculateEstimate(), N), gt)

    # ── el veredicto ─────────────────────────────────────────────────────────
    dif_mm = abs(ate_batch - ate_nuestro) * 1000
    print(f"\n{'backend':32s} {'ATE':>8s} {'tiempo':>8s}")
    print(f"{'nuestro batch (nivel 21/24)':32s} {100*ate_nuestro:6.2f}cm "
          f"{t_nuestro:7.2f}s")
    print(f"{'GTSAM LevenbergMarquardt':32s} {100*ate_batch:6.2f}cm "
          f"{t_batch:7.2f}s")
    print(f"{'GTSAM ISAM2 online':32s} {100*ate_online:6.2f}cm "
          f"{t_isam:7.2f}s")
    print(f"{'GTSAM ISAM2 final':32s} {100*ate_final:6.2f}cm")
    print(f"\nGTSAM vs nuestro: dif de ATE = {dif_mm:.1f} mm (mismo mundo, "
          "misma respuesta)")
    ok = dif_mm < 10.0 and ate_final < 0.15
    print(f"ACTO5_{'OK' if ok else 'FALLO'} dif_mm={dif_mm:.2f} "
          f"ate_gtsam_cm={100*ate_batch:.2f} ate_nuestro_cm="
          f"{100*ate_nuestro:.2f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
