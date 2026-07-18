"""ACTO 3 — El mundo no es lineal: el EKF (localización con mapa conocido).

El robot del curso (el uniciclo del nivel 21) recorre su circuito con un
mapa de landmarks CONOCIDO — localización pura, sin SLAM: el nivel 21 ya
hizo el SLAM; quitarle el mapa deja el estado en 3 variables y la atención
entera en la ÚNICA novedad de este acto: la linealización.

─── La matemática: las mismas ecuaciones, evaluadas en otro sitio ────────────
El modelo de movimiento y el de medición ya no son matrices:

    f(x, u):  x' = x + v·dt·cosθ,  y' = y + v·dt·sinθ,  θ' = θ + ω·dt
    h(x, l):  r = √(dx² + dy²),    b = atan2(dy, dx) − θ     (rango-rumbo)

El EKF hace UNA cosa nueva: linealiza alrededor del estimado actual,

    F = ∂f/∂x = [[1, 0, −v·dt·sinθ],          G = ∂f/∂u (para el ruido
                 [0, 1,  v·dt·cosθ],               de los controles)
                 [0, 0,  1        ]]

    H = ∂h/∂x = [[−dx/r,  −dy/r,   0],        (deriva r y b respecto a
                 [ dy/r², −dx/r², −1]]         x, y, θ — hazlo a mano)

y mete F y H en las MISMAS ecuaciones del acto 2. Eso es todo el "E" del
EKF. El precio (invisible hoy, medido en el nivel 21): cada linealización
se evalúa en el estimado del momento y queda SELLADA en P para siempre.

Y una trampa nueva que el acto 2 no tenía: el rumbo es un ÁNGULO. La
innovación ν = z − h puede dar 6.27 rad cuando el error real es 0.01
(los dos lados de ±π). Sin envolver la innovación, el filtro recibe una
"sorpresa" de 2π donde no había ninguna — y explota. Aquí está la versión
correcta y la versión _MAL, las dos medidas (como la conjugación del
nivel 20: el bug clásico, reproducido a propósito).
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

DT = 0.1
SIGMA_V = 0.10             # ruido del control v (m/s por paso)
SIGMA_W = 0.05             # ruido del control omega (rad/s por paso)
SIGMA_R = 0.10             # ruido del rango (m)
SIGMA_B = 0.05             # ruido del rumbo (rad)
ALCANCE = 3.5
Q_U = np.diag([SIGMA_V ** 2, SIGMA_W ** 2])
R_Z = np.diag([SIGMA_R ** 2, SIGMA_B ** 2])

# los landmarks del circuito del nivel 21 -- pero esta vez el mapa SE CONOCE
LANDMARKS = np.array([
    [1.5, 1.2], [4.0, -1.0], [6.5, 1.2], [9.0, 2.5], [6.5, 3.8],
    [4.0, 6.0], [1.5, 3.8], [-1.0, 2.5], [2.5, -0.8], [8.8, 0.2],
    [8.8, 4.8], [5.5, 5.8], [0.2, 5.6], [-0.8, 0.2]])


def envolver(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def mover(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """f(x, u): el uniciclo, dt hacia adelante."""
    v, w = u
    return np.array([x[0] + v * DT * np.cos(x[2]),
                     x[1] + v * DT * np.sin(x[2]),
                     envolver(x[2] + w * DT)])


def h_rango_rumbo(x: np.ndarray, l: np.ndarray) -> np.ndarray:
    dx, dy = l[0] - x[0], l[1] - x[1]
    return np.array([np.hypot(dx, dy),
                     envolver(np.arctan2(dy, dx) - x[2])])


def generar(semilla: int = 23) -> Dict:
    """El circuito 8x5 m del nivel 21, dos vueltas, en continuo. El robot
    recibe CONTROLES ruidosos (su odometro) y mediciones rango-rumbo a los
    landmarks a la vista."""
    rng = np.random.default_rng(semilla)
    v_recta, w_curva = 1.0, 1.2
    t_curva = (np.pi / 2) / w_curva
    perfil: List[Tuple[float, float]] = []
    for _ in range(2):
        for largo in (8.0, 5.0, 8.0, 5.0):
            perfil += [(v_recta, 0.0)] * int(round(largo / v_recta / DT))
            perfil += [(v_recta, w_curva)] * int(round(t_curva / DT))

    x = np.zeros(3)
    gt, controles, medidas = [x.copy()], [], []
    for v, w in perfil:
        x = mover(x, np.array([v, w]))
        gt.append(x.copy())
        controles.append(np.array([v, w]) + rng.normal(0, [SIGMA_V, SIGMA_W]))
        vistos = []
        for j, l in enumerate(LANDMARKS):
            if np.hypot(l[0] - x[0], l[1] - x[1]) < ALCANCE:
                z = h_rango_rumbo(x, l) + rng.normal(0, [SIGMA_R, SIGMA_B])
                z[1] = envolver(z[1])
                vistos.append((j, z))
        medidas.append(vistos)
    return {"gt": np.array(gt), "controles": controles, "medidas": medidas}


def correr_ekf(m: Dict, con_envolver: bool = True) -> np.ndarray:
    """El EKF de localización. `con_envolver=False` es la versión _MAL:
    idéntica salvo UNA línea — no envuelve la innovación del rumbo."""
    x = np.zeros(3)                        # el arranque se conoce
    P = np.diag([0.01, 0.01, 0.01]) ** 2
    tray = [x.copy()]
    for u, vistos in zip(m["controles"], m["medidas"]):
        # ── predecir con el control (linealizar f) ──
        v, _ = u
        c, s = np.cos(x[2]), np.sin(x[2])
        F = np.array([[1, 0, -v * DT * s],
                      [0, 1, v * DT * c],
                      [0, 0, 1.0]])
        G = np.array([[DT * c, 0], [DT * s, 0], [0, DT]])
        x = mover(x, u)
        P = F @ P @ F.T + G @ Q_U @ G.T
        # ── corregir con cada landmark a la vista (linealizar h) ──
        for j, z in vistos:
            l = LANDMARKS[j]
            dx, dy = l[0] - x[0], l[1] - x[1]
            r2 = dx ** 2 + dy ** 2
            r = np.sqrt(r2)
            H = np.array([[-dx / r, -dy / r, 0.0],
                          [dy / r2, -dx / r2, -1.0]])
            nu = z - h_rango_rumbo(x, l)
            if con_envolver:
                nu[1] = envolver(nu[1])    # LA linea (quitala y mide)
            S = H @ P @ H.T + R_Z
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ nu
            x[2] = envolver(x[2])
            P = (np.eye(3) - K @ H) @ P
        tray.append(x.copy())
    return np.array(tray)


def dead_reckoning(m: Dict) -> np.ndarray:
    """Integrar los controles y rezar: la referencia sin correcciones."""
    x = np.zeros(3)
    tray = [x.copy()]
    for u in m["controles"]:
        x = mover(x, u)
        tray.append(x.copy())
    return np.array(tray)


def rmse_xy(tray: np.ndarray, gt: np.ndarray) -> float:
    d = tray[:, :2] - gt[:len(tray), :2]
    return float(np.sqrt((d ** 2).sum(axis=1).mean()))


def main() -> None:
    print("ACTO 3: el EKF (localizacion en el circuito del nivel 21)\n")
    m = generar()
    dr = dead_reckoning(m)
    ekf = correr_ekf(m)
    mal = correr_ekf(m, con_envolver=False)
    e_dr, e_ekf = rmse_xy(dr, m["gt"]), rmse_xy(ekf, m["gt"])
    e_mal = rmse_xy(mal, m["gt"])
    print(f"  dead reckoning (integrar y rezar): {100*e_dr:.1f} cm")
    print(f"  EKF con mapa conocido            : {100*e_ekf:.1f} cm "
          f"({e_dr/e_ekf:.0f}x mejor)")
    print("  mismas ecuaciones del acto 2 -- solo cambio DONDE se evaluan\n")
    print(f"  ...y la version _MAL (sin envolver la innovacion del rumbo):")
    print(f"  {100*e_mal:.0f} cm ({e_mal/e_ekf:.0f}x peor que la correcta).")
    print("  Un landmark casi detras del robot: z = +3.13, h = -3.13. El")
    print("  error real era 0.01 rad; la resta ciega dijo 6.27. El filtro")
    print("  se creyo esa 'sorpresa' de 2*pi -- y el bug es intermitente:")
    print("  solo dispara cuando un rumbo cruza +-pi (por eso es tan famoso)")


if __name__ == "__main__":
    main()
