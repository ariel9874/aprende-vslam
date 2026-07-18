"""ACTO 2 — El estado se MUEVE: el filtro de Kalman lineal.

Un carrito sobre un riel, con posición y velocidad. El sensor solo mide la
POSICIÓN, con ruido. Al acto 1 se le agrega UNA sola cosa: entre medición
y medición, el estado cambia — hay que PREDECIR antes de corregir.

─── La matemática: predecir y corregir ───────────────────────────────────────
Estado x = [p, v], covarianza P (2×2: ya no basta un σ — p y v se acoplan).

PREDICCIÓN (el modelo, dt hacia adelante): p ← p + v·dt, v ← v. En matrices:

    x ← F·x,   P ← F·P·Fᵀ + Q        F = [[1, dt], [0, 1]]

Q es el ruido de PROCESO: los empujones que el modelo no conoce. La
incertidumbre CRECE — predecir es apostar.

CORRECCIÓN (llega z = posición medida, ruido σ_z): el acto 1, en matrices:

    ν = z − H·x                       H = [1, 0]: SOLO se mide p
    S = H·P·Hᵀ + σ_z²                 la incertidumbre de la innovación
    K = P·Hᵀ·S⁻¹                      la media ponderada, otra vez
    x ← x + K·ν,   P ← (I − K·H)·P    la incertidumbre BAJA

K ahora tiene DOS filas: la innovación de posición corrige TAMBIÉN la
velocidad. ¿Por qué canal? P[0,1], la covarianza cruzada: la predicción
acopló p con v ("si v estaba mal, p quedó mal"), y esa correlación es la
que permite estimar una variable que NINGÚN sensor mide.

Y el secreto mejor guardado: este recursivo resuelve EXACTAMENTE el mismo
problema que un grafo de factores LINEAL (nivel 21) con factores de proceso
y de medición — el estado final del filtro ES la última pose del smoother.
Aquí se verifica a precisión de máquina; en cuanto el mundo se vuelve no
lineal (acto 3), esa igualdad se rompe — y ahí nace toda la discusión
filtro-vs-smoother del curso.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Dict

import numpy as np

DT = 0.1
N = 400                    # pasos (40 s de carrito)
SIGMA_V = 0.15             # ruido de proceso en v por paso (empujones)
SIGMA_P = 0.005            # temblor chico en p (mantiene Q invertible)
SIGMA_Z = 0.50             # el sensor de posicion es MALO a proposito
F = np.array([[1.0, DT], [0.0, 1.0]])
Q = np.diag([SIGMA_P ** 2, SIGMA_V ** 2])
H = np.array([[1.0, 0.0]])


def simular(semilla: int = 23) -> Dict:
    """El carrito real: velocidad que deriva (empujones) y sensor ruidoso."""
    rng = np.random.default_rng(semilla)
    x = np.array([0.0, 1.0])                  # arranca a 1 m/s
    xs, zs = [], []
    for _ in range(N):
        x = F @ x + rng.normal(0, [SIGMA_P, SIGMA_V])
        xs.append(x.copy())
        zs.append(x[0] + rng.normal(0, SIGMA_Z))
    return {"verdad": np.array(xs), "z": np.array(zs)}


def filtrar(zs: np.ndarray) -> Dict:
    """El filtro de Kalman: predecir, corregir, repetir. Memoria: un x y un P."""
    x = np.array([zs[0], 0.0])                # arranca de la primera medicion
    P = np.diag([SIGMA_Z ** 2, 2.0 ** 2])     # y sin idea de la velocidad
    hist_x, hist_P = [], []
    for z in zs:
        # ── predecir: la incertidumbre crece ──
        x = F @ x
        P = F @ P @ F.T + Q
        sigma_antes = np.sqrt(P[0, 0])
        # ── corregir: la incertidumbre baja ──
        nu = z - H @ x                        # la innovacion (la sorpresa)
        S = H @ P @ H.T + SIGMA_Z ** 2
        K = P @ H.T / S                       # la media ponderada, en matrices
        x = x + (K * nu).ravel()
        P = (np.eye(2) - K @ H) @ P
        hist_x.append(x.copy())
        hist_P.append((sigma_antes, np.sqrt(P[0, 0])))
    return {"x": np.array(hist_x), "P": P,
            "respiracion": np.array(hist_P)}   # (sigma pre, sigma post) por paso


def resolver_batch(zs: np.ndarray) -> Dict:
    """El MISMO problema como grafo de factores lineal (nivel 21), resuelto
    de una sola vez: N+1 estados, factores de proceso y de medición.

    ─── La matemática: el grafo lineal ──────────────────────────────────────
    Incógnita X = [x_0 | x_1 | ... | x_N] (todos los estados a la vez).
    Cada factor aporta JᵀΛJ a la información A y JᵀΛr al vector b:

      prior     : x_0 = x̂_0                        (Λ = P_0⁻¹)
      proceso   : x_{k+1} − F·x_k = 0               (Λ = Q⁻¹)
      medición  : H·x_k = z_k                       (Λ = 1/σ_z²)

    Todo es lineal: A·X = b se resuelve UNA vez, sin iterar. La covarianza
    del último estado es el bloque final de A⁻¹ — y debe ser IGUAL al P del
    filtro: marginalizar el pasado (Schur, nivel 21) es lo que el filtro
    hace paso a paso.
    ─────────────────────────────────────────────────────────────────────────
    """
    n_x = 2 * (N + 1)
    A = np.zeros((n_x, n_x))
    b = np.zeros(n_x)
    # prior sobre x_0 (el MISMO del filtro: primera medicion, v desconocida)
    P0_inv = np.linalg.inv(np.diag([SIGMA_Z ** 2, 2.0 ** 2]))
    A[0:2, 0:2] += P0_inv
    b[0:2] += P0_inv @ np.array([zs[0], 0.0])
    Q_inv = np.linalg.inv(Q)
    for k in range(N):
        i, j = 2 * k, 2 * (k + 1)
        # proceso: J = [-F | I] sobre los bloques k y k+1
        A[i:i+2, i:i+2] += F.T @ Q_inv @ F
        A[i:i+2, j:j+2] += -F.T @ Q_inv
        A[j:j+2, i:i+2] += -Q_inv @ F
        A[j:j+2, j:j+2] += Q_inv
        # medicion de z_k sobre el bloque k+1
        A[j:j+2, j:j+2] += H.T @ H / SIGMA_Z ** 2
        b[j:j+2] += (H.T * zs[k]).ravel() / SIGMA_Z ** 2
    X = np.linalg.solve(A, b)
    cov = np.linalg.inv(A)
    return {"x_final": X[-2:], "P_final": cov[-2:, -2:],
            "tray": X.reshape(-1, 2)[1:]}


def main() -> None:
    print("ACTO 2: el filtro de Kalman lineal (el carrito)\n")
    sim = simular()
    kf = filtrar(sim["z"])

    e_crudo = float(np.sqrt(np.mean((sim["z"] - sim["verdad"][:, 0]) ** 2)))
    e_filtro = float(np.sqrt(np.mean((kf["x"][:, 0] - sim["verdad"][:, 0]) ** 2)))
    e_vel = float(np.sqrt(np.mean((kf["x"][:, 1] - sim["verdad"][:, 1]) ** 2)))
    # la alternativa ingenua a "estimar v": derivar la medicion
    v_diff = np.diff(sim["z"]) / DT
    e_diff = float(np.sqrt(np.mean((v_diff - sim["verdad"][1:, 1]) ** 2)))
    print(f"  posicion: sensor crudo {100*e_crudo:.1f} cm | "
          f"filtrada {100*e_filtro:.1f} cm ({e_crudo/e_filtro:.1f}x mejor)")
    print(f"  velocidad SIN sensor de velocidad: derivar la medicion da "
          f"{e_diff:.1f} m/s de error;")
    print(f"  el filtro, {e_vel:.2f} m/s ({e_diff/e_vel:.0f}x mejor) -- el "
          "canal es P[0,1]:")
    print("  la correlacion p-v que la prediccion creo\n")

    r = kf["respiracion"][-1]
    print(f"  la respiracion de sigma (ultimo paso): predice {100*r[0]:.1f} ->"
          f" corrige {100*r[1]:.1f} cm (crece al apostar, baja al medir)\n")

    lote = resolver_batch(sim["z"])
    d_x = float(np.abs(kf["x"][-1] - lote["x_final"]).max())
    d_P = float(np.abs(kf["P"] - lote["P_final"]).max())
    print(f"  el mismo problema como grafo lineal ({2*(N+1)} incognitas,"
          " resuelto de una vez):")
    print(f"  |estado final del filtro - ultimo estado del grafo| = {d_x:.1e}")
    print(f"  |P del filtro - covarianza marginal del grafo|     = {d_P:.1e}")
    print("  el filtro ES el grafo lineal, resuelto por recursion -- esa")
    print("  igualdad se rompera en cuanto el mundo sea no lineal (acto 3)")


if __name__ == "__main__":
    main()
