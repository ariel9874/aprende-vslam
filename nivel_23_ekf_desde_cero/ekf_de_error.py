"""ACTO 4 — El EKF de ERROR (error-state): el filtro de los VIO reales.

El mismo mundo del nivel 22 (mismas medidas, bit a bit): IMU a 100 Hz con
sesgo, visión a 4 Hz, apagón de 5 s en plena curva. El grafo VIO del 22 lo
resolvió como SMOOTHER; esto es como lo resuelve un TELÉFONO (ARKit/ARCore):
con un filtro — pero no un EKF cualquiera.

─── La matemática: dos pisos ─────────────────────────────────────────────────
El truco es partir el estado en dos:

    NOMINAL  x = [p, v, θ, b_g]  +  el mapa      (grande, no lineal)
    ERROR    δx = [δp, δv, δθ, δb_g, δl...]      (pequeño SIEMPRE)

con la verdad = nominal ⊞ error. El reparto de trabajo:

1. El NOMINAL se integra a 100 Hz con el modelo COMPLETO, sin linealizar
   nada (punto medio — el integrador del nivel 22, que reproduce el
   circuito a 0.02 cm). Predecir no aproxima: esa era la herida del EKF.

2. El ERROR es quien tiene filtro de Kalman: su dinámica linealizada
   (jacobiano A, abajo) propaga SOLO la covarianza — el error medio es 0
   hasta que llega una medición. Y como el error se corrige cada vez que
   la visión habla, NUNCA crece: la linealización siempre se evalúa donde
   es válida. La debilidad del acto 3, curada por construcción.

3. Cuando la visión corrige (δx = K·ν), el error se INYECTA al nominal
   (p += δp, θ = envolver(θ + δθ), ...) y se RESETEA a cero. El filtro
   vive siempre alrededor del origen: lejos de todo ±π (la inmunidad al
   bug del acto 3) — y en 3D, δθ vive plano en el tangente (3 números)
   mientras el cuaternión (4, con su norma) vive protegido en el nominal.
   Esa es LA razón histórica del error-state (Solà, "Quaternion
   kinematics for the error-state Kalman filter": la lectura del nivel).

La dinámica del error (derívala perturbando el paso nominal a 1er orden):

    δp' = δp + δv·dt                        A[p,v] = I·dt
    δv' = δv + (J₉₀·R·a)·δθ·dt              A[v,θ] = J₉₀·a_mundo·dt
    δθ' = δθ − δb_g·dt                      A[θ,b] = −dt   (el canal del sesgo)
    δb'  = δb_g                              (+ random walk pequeño)

J₉₀ = [[0,−1],[1,0]] (girar 90°): dR/dθ = J₉₀·R. El mapa entra al estado
igual que en el filtro EKF-SLAM del nivel 21 (mismo mundo desconocido que
resolvió el grafo del 22 — la comparación filtro-vs-smoother es justa).
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from mundo_imu import (BIAS_GYRO_REAL, DT, R2, SIGMA_ACC, SIGMA_GYRO,
                       SIGMA_OBS, envolver, observar)

J90 = np.array([[0.0, -1.0], [1.0, 0.0]])
R_OBS = np.eye(2) * SIGMA_OBS ** 2
Q_BIAS = (3e-5) ** 2       # random walk del sesgo por muestra (deja aprender)


def correr(m: Dict, estimar_bias: bool = True) -> Dict:
    """El ESKF-VIO completo. Devuelve la trayectoria ONLINE en keyframes,
    el historial del sesgo (para verlo converger EN VIVO) y el mapa."""
    # nominal
    p = np.zeros(2)
    v = np.array([1.0, 0.0])
    th = 0.0
    bg = 0.0
    lms: Dict[int, np.ndarray] = {}        # landmark j -> posicion estimada
    orden = []                             # j en el orden del estado
    # error: covarianza sobre [dp(2), dv(2), dth, dbg, dl...]
    s_bg0 = 0.05 if estimar_bias else 0.0
    P = np.diag([0.01, 0.01, 0.3, 0.3, 0.01, s_bg0]) ** 2

    obs_por_kf: Dict[int, list] = {}
    for i, j, z in m["obs"]:
        obs_por_kf.setdefault(i, []).append((j, z))

    def corregir(j, z):
        nonlocal p, v, th, bg, P
        n = 6 + 2 * len(orden)
        R = R2(th)
        if j not in lms:
            # INICIALIZAR el landmark: l = p + R·z, y agrandar P con el
            # jacobiano de esa formula (el patron del filtro del nivel 21)
            l = p + R @ z
            Jx = np.zeros((2, n))
            Jx[:, 0:2] = np.eye(2)         # dl/d(dp)
            Jx[:, 4] = J90 @ (R @ z)       # dl/d(dth)
            P_nuevo = np.zeros((n + 2, n + 2))
            P_nuevo[:n, :n] = P
            P_nuevo[n:, :n] = Jx @ P
            P_nuevo[:n, n:] = (Jx @ P).T
            P_nuevo[n:, n:] = Jx @ P @ Jx.T + R @ R_OBS @ R.T
            P = P_nuevo
            lms[j] = l
            orden.append(j)
            return
        k = 6 + 2 * orden.index(j)
        l = lms[j]
        nu = z - observar(np.array([p[0], p[1], th]), l)
        Rt = R.T
        H = np.zeros((2, len(P)))
        H[:, 0:2] = -Rt                    # dz/d(dp)
        H[:, 4] = -Rt @ J90 @ (l - p)      # dz/d(dth)
        H[:, k:k + 2] = Rt                 # dz/d(dl)
        S = H @ P @ H.T + R_OBS
        K = P @ H.T @ np.linalg.inv(S)
        d = K @ nu                         # el ERROR estimado...
        p = p + d[0:2]                     # ...se INYECTA al nominal...
        v = v + d[2:4]
        th = float(envolver(th + d[4]))
        if estimar_bias:
            bg = bg + d[5]
        for idx, jj in enumerate(orden):
            lms[jj] = lms[jj] + d[6 + 2 * idx:8 + 2 * idx]
        P = (np.eye(len(P)) - K @ H) @ P   # ...y se RESETEA a cero
        # (el jacobiano fino del reset ~ I en 2D: ejercicio 3)

    for j, z in obs_por_kf.get(0, []):
        corregir(j, z)
    tray = [np.array([p[0], p[1], th])]
    hist_bias = [bg]

    for k_tramo, tramo in enumerate(m["tramos"]):
        for w_med, a_med in tramo:
            # ── el NOMINAL: modelo completo, punto medio, sin linealizar ──
            w_hat = w_med - bg
            th_m = th + 0.5 * w_hat * DT
            a_w = R2(th_m) @ a_med
            p = p + v * DT + 0.5 * a_w * DT ** 2
            v = v + a_w * DT
            th = float(envolver(th + w_hat * DT))
            # ── el ERROR: solo su covarianza (el error medio sigue en 0) ──
            A = np.eye(6)
            A[0:2, 2:4] = np.eye(2) * DT
            dRa = J90 @ a_w
            A[0:2, 4] = 0.5 * dRa * DT ** 2
            A[2:4, 4] = dRa * DT
            A[2:4, 5] = -0.5 * dRa * DT ** 2
            A[4, 5] = -DT                  # el canal por el que se APRENDE b_g
            P[:6, :] = A @ P[:6, :]
            P[:, :6] = P[:, :6] @ A.T
            P[0:2, 0:2] += np.eye(2) * (0.5 * SIGMA_ACC * DT ** 2) ** 2
            P[2:4, 2:4] += np.eye(2) * (SIGMA_ACC * DT) ** 2
            P[4, 4] += (SIGMA_GYRO * DT) ** 2
            if estimar_bias:
                P[5, 5] += Q_BIAS
        # ── la VISION corrige el error (si este keyframe no esta apagado) ──
        for j, z in obs_por_kf.get(k_tramo + 1, []):
            corregir(j, z)
        tray.append(np.array([p[0], p[1], th]))
        hist_bias.append(bg)

    mapa = np.full((len(m["landmarks"]), 2), np.nan)
    for j, l in lms.items():
        mapa[j] = l
    return {"poses": np.array(tray), "biases": np.array(hist_bias),
            "landmarks": mapa}


def integrar_solo_imu(m: Dict) -> np.ndarray:
    """Dead reckoning: el nominal sin correcciones (la referencia del 22)."""
    p, v, th = np.zeros(2), np.array([1.0, 0.0]), 0.0
    tray = [np.array([p[0], p[1], th])]
    for tramo in m["tramos"]:
        for w_med, a_med in tramo:
            th_m = th + 0.5 * w_med * DT
            a_w = R2(th_m) @ a_med
            p = p + v * DT + 0.5 * a_w * DT ** 2
            v = v + a_w * DT
            th = float(envolver(th + w_med * DT))
        tray.append(np.array([p[0], p[1], th]))
    return np.array(tray)


def main() -> None:
    from mundo_imu import generar, rmse_xy
    print("ACTO 4: el EKF de error (el mundo del nivel 22, con filtro)\n")
    m = generar()
    apag = np.array(m["apagados"])

    def rmse_en(idx, poses):
        d = poses[idx, :2] - m["gt"][idx, :2]
        return float(np.sqrt((d ** 2).sum(axis=1).mean()))

    dr = integrar_solo_imu(m)
    r = correr(m)
    r_sinb = correr(m, estimar_bias=False)
    e_dr = rmse_xy(dr, m["gt"])
    e, e_a = rmse_xy(r["poses"], m["gt"]), rmse_en(apag, r["poses"])
    e_sb, e_sb_a = (rmse_xy(r_sinb["poses"], m["gt"]),
                    rmse_en(apag, r_sinb["poses"]))

    mitad = len(m["gt"]) // 2          # la segunda vuelta del circuito
    e_v1 = rmse_en(np.arange(1, mitad), r["poses"])
    e_v2 = rmse_en(np.arange(mitad, len(m["gt"])), r["poses"])

    print(f"  IMU sola (dead reckoning)   : {100*e_dr:.0f} cm")
    print(f"  ESKF sin estimar el sesgo   : {100*e_sb:.1f} cm | "
          f"apagon {100*e_sb_a:.1f} cm")
    print(f"  ESKF completo               : {100*e:.1f} cm | "
          f"apagon {100*e_a:.1f} cm")
    print(f"  sesgo descubierto EN VIVO   : {r['biases'][-1]:.4f} "
          f"(real {BIAS_GYRO_REAL})\n")
    print(f"  la novatada de la primera vuelta: vuelta 1 {100*e_v1:.1f} cm |"
          f" vuelta 2 {100*e_v2:.1f} cm")
    print("  (el mapa del filtro nace incierto y converge; pero las poses de")
    print("  la vuelta 1 ya fueron EMITIDAS con el mapa joven -- el filtro no")
    print("  puede corregir su pasado; el smoother, si: leccion 25)\n")
    print("  las referencias del nivel 22 sobre ESTE MISMO mundo y semilla:")
    print("  coast (vision sin IMU) 62.2 cm en el apagon; el grafo VIO")
    print(f"  (smoother) 4.7 cm total / 4.8 en el apagon. El filtro cruza el")
    print(f"  apagon {62.2/(100*e_a):.1f}x mejor que el coast -- y el smoother")
    print(f"  lo cruza {100*e_a/4.8:.1f}x mejor que el filtro: el apagon cae")
    print("  en plena novatada, y solo el smoother la perdona.")


if __name__ == "__main__":
    main()
