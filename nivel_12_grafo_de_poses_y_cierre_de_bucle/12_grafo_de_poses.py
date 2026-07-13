#!/usr/bin/env python3
"""
Nivel 12 — Grafo de poses y cierre de bucle
===========================================

La deriva (nivel 08) por fin se puede DESHACER: cuando el sistema reconoce
un sitio por el que ya pasó, esa restricción se propaga hacia atrás por toda
la trayectoria. Cuatro experimentos:

    1. Exp/Log de SE(3) y Sim(3), verificados contra la exponencial de
       matrices por serie (la verdad numerica, sin formulas cerradas)
    2. el CIERRE DE BUCLE: 1.09 m de deriva -> 5 cm
    3. STRASDAT (el experimento del nivel): con deriva de ESCALA, un grafo
       SE(3) EMPEORA la trayectoria; uno Sim(3) la arregla
    4. HUBER: un falso positivo de bucle no debe doblar el grafo entero

Uso:
    python 12_grafo_de_poses.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lie import (hat, se3_exp, se3_inv, se3_log, sim3_exp, sim3_inv, sim3_log,
                 so3_exp)
from pose_graph import GrafoDePoses

AQUI = Path(__file__).resolve().parent


# ─────────────────────────── utilidades ──────────────────────────────────────

def expm_serie(A: np.ndarray, order: int = 30) -> np.ndarray:
    """Exponencial de matrices por SERIE (scaling-and-squaring).

    Es la "verdad numerica" contra la que se validan las formulas cerradas de
    Lie: exp(A) = I + A + A²/2! + ... calculado a lo bruto. Si nuestras
    formulas (Rodrigues, V, W) estan bien, deben coincidir con esto.
    """
    n = max(0, int(np.ceil(np.log2(max(np.linalg.norm(A), 1e-9)))) + 2)
    A2 = A / (2.0 ** n)
    X = np.eye(A.shape[0])
    term = np.eye(A.shape[0])
    for k in range(1, order):
        term = term @ A2 / k
        X = X + term
    for _ in range(n):
        X = X @ X
    return X


def poses_circulo(n: int, radio: float = 5.0) -> list[np.ndarray]:
    """Trayectoria circular cerrada: la cámara vuelve por donde vino."""
    poses = []
    for k in range(n):
        ang = 2 * np.pi * k / n
        T = np.eye(4)
        T[:3, :3] = so3_exp(np.array([0.0, ang, 0.0]))     # gira siguiendo el arco
        T[:3, 3] = np.array([radio * np.cos(ang), 0.0, radio * np.sin(ang)])
        poses.append(T)
    return poses


def parte_rigida(S: np.ndarray) -> np.ndarray:
    """Quita la escala de una matriz Sim(3): deja la SE(3) que hay dentro."""
    s = float(np.linalg.det(S[:3, :3])) ** (1.0 / 3.0)
    T = np.eye(4)
    T[:3, :3] = S[:3, :3] / s
    T[:3, 3] = S[:3, 3]
    return T


def ate(poses: dict, gt: list) -> float:
    """RMSE de posición contra el ground truth (sin alinear: el nodo 0 está
    anclado en su valor real, así que los marcos ya coinciden)."""
    d = [np.linalg.norm(poses[k][:3, 3] - gt[k][:3, 3]) for k in range(len(gt))]
    return float(np.sqrt(np.mean(np.square(d))))


# ─────────────────────────── los experimentos ────────────────────────────────

def exp1_lie() -> tuple[float, float]:
    """Exp/Log verificados contra la exponencial de matrices por serie."""
    rng = np.random.default_rng(5)
    print("1. El algebra de Lie (formulas cerradas vs serie de matrices):")

    # SE(3): Exp(xi) debe ser exp del elemento de algebra [[ [w]x, rho ],[0,0]]
    err_se3 = 0.0
    for _ in range(20):
        xi = rng.normal(0, 0.8, 6)
        xi_hat = np.zeros((4, 4))
        xi_hat[:3, :3] = hat(xi[3:6])
        xi_hat[:3, 3] = xi[:3]
        err_se3 = max(err_se3, float(np.abs(se3_exp(xi) - expm_serie(xi_hat)).max()))
        # y Log invierte Exp exactamente
        err_se3 = max(err_se3, float(np.abs(se3_log(se3_exp(xi)) - xi).max()))

    # Sim(3): el elemento de algebra lleva ademas lambda*I en la diagonal
    err_sim3 = 0.0
    casos = [rng.normal(0, 0.8, 7) for _ in range(20)]
    casos += [np.zeros(7),                                       # todo cero
              np.r_[rng.normal(0, 1, 3), np.zeros(3), 0.4],      # theta~0, lam grande
              np.r_[rng.normal(0, 1, 3), rng.normal(0, 1, 3), 1e-12],  # lam~0
              np.r_[0.5, -0.2, 0.1, 1e-12, 0, 0, -0.6]]          # theta~0 y lam<0
    for xi in casos:
        xi_hat = np.zeros((4, 4))
        xi_hat[:3, :3] = xi[6] * np.eye(3) + hat(xi[3:6])
        xi_hat[:3, 3] = xi[:3]
        S = sim3_exp(xi)
        err_sim3 = max(err_sim3, float(np.abs(S - expm_serie(xi_hat)).max()))
        err_sim3 = max(err_sim3, float(np.abs(sim3_log(S) - xi).max()))
        err_sim3 = max(err_sim3, float(np.abs(sim3_inv(S) @ S - np.eye(4)).max()))

    print(f"   SE(3):  Exp/Log vs serie, error max {err_se3:.2e}")
    print(f"   Sim(3): Exp/Log vs serie, error max {err_sim3:.2e}")
    print("   (incluidos los casos limite: theta~0, lambda~0, y ambos a la vez.")
    print("    Cada uno tiene su rama de Taylor, y es donde las formulas")
    print("    ingenuas mueren dividiendo por cero.)")
    return err_se3, err_sim3


def exp2_cierre_de_bucle(salida: Path) -> tuple[float, float]:
    """El cierre de bucle deshace la deriva acumulada."""
    salida.mkdir(exist_ok=True)     # el examen puede llamarnos sin main()
    rng = np.random.default_rng(3)
    n = 30
    gt = poses_circulo(n)

    # Odometria RUIDOSA: cada paso relativo lleva un pequeno error que se
    # ACUMULA al componerlo (el nivel 08, otra vez).
    medidas, cadena = [], [gt[0].copy()]
    for k in range(n - 1):
        T_rel = se3_inv(gt[k]) @ gt[k + 1]
        ruido = se3_exp(np.r_[rng.normal(0, 0.02, 3), rng.normal(0, 0.01, 3)])
        T_med = T_rel @ ruido
        medidas.append(T_med)
        cadena.append(cadena[-1] @ T_med)

    deriva = float(np.linalg.norm(cadena[-1][:3, 3] - gt[-1][:3, 3]))
    ate_antes = ate({k: cadena[k] for k in range(n)}, gt)

    # El grafo: nodos = poses estimadas; aristas = odometria + EL BUCLE.
    # El bucle dice: "el nodo n-1 esta justo al lado del 0, y esta es su
    # transformacion relativa" (lo que mediria una relocalizacion por PnP).
    g = GrafoDePoses("se3")
    g.add_pose(0, cadena[0], fixed=True)          # gauge: anclar el primero
    for k in range(1, n):
        g.add_pose(k, cadena[k])
    for k in range(n - 1):
        g.add_odometry(k, k + 1, medidas[k], np.eye(6) * 1e2)
    T_loop = se3_inv(gt[-1]) @ gt[0]              # la medida del bucle (limpia)
    g.add_loop(n - 1, 0, T_loop, np.eye(6) * 1e4)

    hist: list[float] = []
    opt = g.optimize(iterations=30, historial=hist)
    ate_despues = ate(opt, gt)

    print(f"\n2. CIERRE DE BUCLE ({n} poses en circulo, odometria ruidosa):")
    print(f"   deriva final (nodo {n-1} vs su verdad): {deriva:6.2f} m")
    print(f"   ATE de la trayectoria: {ate_antes:6.2f} m  ->  {ate_despues:6.3f} m")
    print(f"   costo del grafo:  {hist[0]:9.1f}  ->  {hist[-1]:7.1f}")
    print("   Fijate en lo que ha pasado: UNA restriccion (el nodo 29 esta")
    print("   junto al 0) ha corregido las 30 poses. El grafo REPARTE el error")
    print("   por toda la cadena, en vez de amontonarlo al final.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        g_xy = np.array([T[:3, 3] for T in gt])
        a_xy = np.array([cadena[k][:3, 3] for k in range(n)])
        d_xy = np.array([opt[k][:3, 3] for k in range(n)])
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.plot(g_xy[:, 0], g_xy[:, 2], "k--", lw=1.5, label="verdad")
        ax.plot(a_xy[:, 0], a_xy[:, 2], "-o", ms=3, color="tab:orange",
                label=f"odometria (ATE {ate_antes:.2f} m)")
        ax.plot(d_xy[:, 0], d_xy[:, 2], "-o", ms=3, color="tab:blue",
                label=f"tras el bucle (ATE {ate_despues:.3f} m)")
        ax.plot([a_xy[-1, 0], a_xy[0, 0]], [a_xy[-1, 2], a_xy[0, 2]], "r:",
                lw=2, label="restriccion de bucle")
        ax.set_xlabel("x [m]"), ax.set_ylabel("z [m]")
        ax.set_title("Una restriccion corrige toda la trayectoria")
        ax.axis("equal"), ax.grid(True, alpha=0.3), ax.legend()
        fig.savefig(salida / "cierre_de_bucle.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass
    return ate_antes, ate_despues


def exp3_strasdat() -> tuple[float, float, float]:
    """EL experimento del nivel (Strasdat et al., RSS 2010).

    La odometria MONOCULAR no solo deriva en posicion: deriva en ESCALA (el
    gauge del nivel 10 se va corrompiendo). Aqui cada paso encoge un 1%.
    Un grafo SE(3) no tiene DONDE meter esa inconsistencia: sus nodos son
    rigidos. Un grafo Sim(3) tiene el septimo grado de libertad exactamente
    para ella.
    """
    n, lam_bias = 24, -0.01                     # -1% de escala POR PASO
    gt = poses_circulo(n)

    # Cadena con escala derivante: cada medida relativa lleva s = e^lam.
    sim_med, cadena = [], [np.eye(4)]
    for k in range(n - 1):
        S_rel = (se3_inv(gt[k]) @ gt[k + 1]).copy()
        S_rel[:3, :3] *= np.exp(lam_bias)       # <- la deriva de escala
        sim_med.append(S_rel)
        cadena.append(cadena[-1] @ S_rel)
    cadena = [gt[0] @ S for S in cadena]        # arrancar en la pose real

    T_loop = se3_inv(gt[-1]) @ gt[0]            # el bucle: limpio, sin escala

    ate_odom = ate({k: parte_rigida(cadena[k]) for k in range(n)}, gt)

    # ── grafo SE(3): la odometria que registraria un sistema CIEGO a la escala
    rig = [parte_rigida(S) for S in cadena]
    g_se3 = GrafoDePoses("se3")
    g_se3.add_pose(0, rig[0], fixed=True)
    for k in range(1, n):
        g_se3.add_pose(k, rig[k])
    for k in range(n - 1):
        g_se3.add_odometry(k, k + 1, se3_inv(rig[k]) @ rig[k + 1], np.eye(6) * 1e2)
    g_se3.add_loop(n - 1, 0, T_loop, np.eye(6) * 1e4)
    ate_se3 = ate(g_se3.optimize(iterations=30), gt)

    # ── grafo Sim(3): MISMOS datos, un grado de libertad mas por nodo
    g_sim = GrafoDePoses("sim3")
    g_sim.add_pose(0, cadena[0], fixed=True)
    for k in range(1, n):
        g_sim.add_pose(k, cadena[k])
    for k in range(n - 1):
        g_sim.add_odometry(k, k + 1, sim_med[k], np.eye(7) * 1e2)
    loop_sim = np.eye(4)
    loop_sim[:3, :4] = T_loop[:3, :4]           # medida rigida: s = 1
    g_sim.add_loop(n - 1, 0, loop_sim, np.eye(7) * 1e4)
    res = g_sim.optimize(iterations=30)
    ate_sim3 = ate({k: parte_rigida(res[k]) for k in res}, gt)

    print(f"\n3. STRASDAT: deriva de ESCALA del 1% por paso ({n} poses):")
    print(f"   {'odometria (sin optimizar)':>30s}: ATE {ate_odom:6.2f} m")
    print(f"   {'grafo SE(3)':>30s}: ATE {ate_se3:6.2f} m")
    print(f"   {'grafo Sim(3)':>30s}: ATE {ate_sim3:6.2f} m")
    if ate_se3 > ate_odom:
        print("   El grafo SE(3) EMPEORA la trayectoria. No es un bug: es que le")
        print("   estas pidiendo lo imposible. Sus nodos son RIGIDOS, asi que la")
        print("   unica forma que tiene de cerrar el bucle es DEFORMAR la")
        print("   geometria (mover traslaciones que estaban bien) para absorber")
        print("   un error que es de ESCALA. Reparte la mentira en vez de")
        print("   corregirla.")
    print("   El Sim(3) tiene el 7o grado de libertad: cada nodo puede")
    print("   RE-ESCALARSE, asi que la correccion del bucle se reparte como lo")
    print("   que es (escala) y la trayectoria se recupera. Ese es el resultado")
    print("   de Strasdat (RSS 2010), y la razon de que el SLAM monocular use")
    print("   grafos Sim(3). OJO, adelanto del nivel 15: en RGB-D la escala es")
    print("   una MEDICION, no un gauge, y el bucle vuelve a ser SE(3). Usar")
    print("   Sim(3) alli es el error SIMETRICO, y el repo padre lo pago caro:")
    print("   cada bucle re-escalaba el mapa metrico y el error se componia")
    print("   (22 cm de ATE, escala 2.09). El grupo correcto depende de QUIEN")
    print("   fija la escala.")
    return ate_odom, ate_se3, ate_sim3


def exp4_falso_positivo() -> dict:
    """Un FALSO POSITIVO de bucle: cuanto puede Huber, y cuanto NO puede.

    El resultado medido es incomodo y por eso vale la pena: Huber con un
    umbral razonable NO salva al grafo. Es la misma leccion del nivel 11
    (Huber DEGRADA, no rechaza), ahora en el backend.
    """
    rng = np.random.default_rng(9)
    n = 30
    gt = poses_circulo(n)

    medidas, cadena = [], [gt[0].copy()]
    for k in range(n - 1):
        T_rel = se3_inv(gt[k]) @ gt[k + 1]
        ruido = se3_exp(np.r_[rng.normal(0, 0.02, 3), rng.normal(0, 0.01, 3)])
        T_med = T_rel @ ruido
        medidas.append(T_med)
        cadena.append(cadena[-1] @ T_med)

    T_loop = se3_inv(gt[-1]) @ gt[0]
    # EL FALSO POSITIVO: el sistema "reconoce" que el nodo 15 es el mismo sitio
    # que el 2 (dos pasillos identicos). Es MENTIRA: estan en lados opuestos.
    T_falso = se3_inv(gt[15]) @ gt[2]
    T_falso = T_falso @ se3_exp(np.r_[3.0, 0.0, 2.0, 0.0, 0.5, 0.0])  # basura

    def correr(huber: float, con_falso: bool = True) -> float:
        g = GrafoDePoses("se3")
        g.HUBER_DELTA = huber
        g.add_pose(0, cadena[0], fixed=True)
        for k in range(1, n):
            g.add_pose(k, cadena[k])
        for k in range(n - 1):
            g.add_odometry(k, k + 1, medidas[k], np.eye(6) * 1e2)
        g.add_loop(n - 1, 0, T_loop, np.eye(6) * 1e4)      # el bucle BUENO
        if con_falso:
            g.add_loop(15, 2, T_falso, np.eye(6) * 1e4)    # el FALSO
        return ate(g.optimize(iterations=30), gt)

    print("\n4. UN FALSO POSITIVO de bucle (el sistema cree que el nodo 15 es")
    print("   el mismo sitio que el 2, y se equivoca). Que puede hacer Huber?")

    r = {
        "rechazado (verificacion)": correr(1.0, con_falso=False),
        "cuadratico (sin Huber)": correr(1e9),
        "Huber delta=1.0": correr(1.0),
        "Huber delta=0.01": correr(0.01),
    }
    for nombre in ["cuadratico (sin Huber)", "Huber delta=1.0",
                   "Huber delta=0.01", "rechazado (verificacion)"]:
        print(f"   {nombre:>26s}: ATE {r[nombre]:6.3f} m")

    print("   Lee la tabla con cuidado, porque no dice lo que esperabas:")
    print("   - El bucle falso, sin robustez, destroza el grafo (3.5 m).")
    print("   - Huber con un umbral razonable (1.0) APENAS ayuda (3.3 m). El")
    print("     residuo del falso es enorme, pero Huber no lo anula: lo deja")
    print("     empujando con fuerza CONSTANTE (w=d/|r| => w*|r| = d), y esa")
    print("     fuerza sigue compitiendo con los 29 factores de odometria.")
    print("   - Bajando el umbral a 0.01 SI se recupera... pero eso tambien")
    print("     amansa al bucle BUENO (su residuo inicial es 58, tampoco es")
    print("     pequeno): estas comprando robustez a costa de la correccion")
    print("     legitima. Es una perilla peligrosa, no una solucion.")
    print("   La defensa REAL es no meter la arista: VERIFICAR el bucle con")
    print("   geometria antes de creerselo (matching + PnP + contar inliers;")
    print("   el repo padre exige 40). Huber es la SEGUNDA linea de defensa,")
    print("   no la primera. En el nivel 13 lo veras hecho.")
    return r


def main() -> int:
    salida = AQUI / "salida"
    salida.mkdir(exist_ok=True)

    exp1_lie()
    exp2_cierre_de_bucle(salida)
    exp3_strasdat()
    exp4_falso_positivo()

    print(f"\nGrafica en {salida / 'cierre_de_bucle.png'}")
    print("Ahora corre `python verificacion.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
