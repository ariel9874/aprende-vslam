#!/usr/bin/env python3
"""
Nivel 21 (bonus) — Grafos de factores: cuatro formas del mismo problema
=======================================================================

Un robot, un circuito, las MISMAS medidas — y cuatro backends que difieren
solo en la FORMA del grafo:

    completo   poses + landmarks: el patron oro (y el mas caro)
    poses      landmarks comprimidos en factores relativos (nivel 12)
    ventana    el pasado MARGINALIZADO en un prior denso (Schur)
    filtro     EKF: marginalizar cada pose al instante (el extremo)

    python 21_grafos.py          # la tabla + las graficas en salida/
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import filtro_ekf
import grafo_completo
import grafo_de_poses
import ventana_marginalizada as ventana
from mundo import generar, rmse_xy

AQUI = Path(__file__).resolve().parent


def main() -> int:
    m = generar()
    gt = m["gt"]
    N = len(gt)
    corte = N // 2
    print(f"El mundo: {N} poses, {len(m['landmarks'])} landmarks, "
          f"{len(m['odo'])} factores de odometria, {len(m['obs'])} "
          "observaciones\n")

    filas = []

    def medir(nombre, fn):
        t0 = time.perf_counter()
        r = fn()
        dt = time.perf_counter() - t0
        filas.append((nombre, r["rmse"], dt, r.get("nota", "")))
        return r

    filas.append(("odometria pura (sin optimizar)",
                  rmse_xy(m["inicial"], gt), 0.0, "la deriva de siempre"))

    rc = medir("grafo COMPLETO (poses+landmarks)", lambda: {
        **(res := grafo_completo.optimizar(m)),
        "rmse": rmse_xy(res["poses"], gt)})

    rp = medir("grafo de POSES (con bucles)", lambda: {
        **(res := grafo_de_poses.optimizar(m)),
        "rmse": rmse_xy(res["poses"], gt),
        "nota": f"{len(res['bucles'])} bucles sintetizados"})

    rv = medir("VENTANA marginalizada (2a mitad)", lambda: {
        **(res := ventana.optimizar_ventana(m, corte, marginalizar=True)),
        "rmse": rmse_xy(res["poses"][corte:], gt[corte:]),
        "nota": "prior de Schur sobre la frontera"})

    rx = medir("ventana CORTADA (sin prior)", lambda: {
        **(res := ventana.optimizar_ventana(m, corte, marginalizar=False)),
        "rmse": rmse_xy(res["poses"][corte:], gt[corte:]),
        "nota": "la informacion del pasado, tirada"})

    rf = medir("FILTRO EKF (trayectoria online)", lambda: {
        **(res := filtro_ekf.correr(m)),
        "rmse": rmse_xy(res["tray"], gt),
        "nota": "cada pose se emite y se sella"})

    print(f"{'backend':36s} {'RMSE':>8s} {'tiempo':>8s}  nota")
    for nombre, rmse, dt, nota in filas:
        print(f"{nombre:36s} {100*rmse:6.1f}cm {dt:7.2f}s  {nota}")

    print("\nLa jerarquia no es casual: es cuanta informacion conserva cada")
    print("forma del grafo — y cuanto pagas por conservarla.")

    # ── graficas: trayectorias + los patrones de dispersion ─────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
        ax = axs[0]
        ax.plot(gt[:, 0], gt[:, 1], "k--", lw=1.5, label="verdad")
        ax.plot(m["inicial"][:, 0], m["inicial"][:, 1], color="0.6",
                lw=1, label="odometria")
        ax.plot(rc["poses"][:, 0], rc["poses"][:, 1], lw=1.4,
                label="grafo completo")
        ax.plot(rf["tray"][:, 0], rf["tray"][:, 1], lw=1.0,
                label="EKF (online)")
        ax.scatter(m["landmarks"][:, 0], m["landmarks"][:, 1], marker="*",
                   c="tab:red", s=60, label="landmarks")
        ax.legend(fontsize=8), ax.axis("equal"), ax.grid(alpha=0.3)
        ax.set_title("el mismo mundo, cuatro estimadores")

        H = rc["H"]
        axs[1].spy(np.abs(H) > 1e-9, markersize=0.6)
        axs[1].set_title("H del grafo completo: DISPERSA\n"
                         "(flecha: poses | landmarks)")
        Hp = rv["H_prior"]
        axs[2].spy(np.abs(Hp) > 1e-9, markersize=0.8)
        axs[2].set_title("el prior tras marginalizar: DENSO\n"
                         "(el fill-in: el precio de olvidar)")
        salida = AQUI / "salida"
        salida.mkdir(exist_ok=True)
        fig.savefig(salida / "grafos.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\nGraficas: {salida / 'grafos.png'}")
    except ImportError:
        pass

    print("Ahora corre `python verificacion.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
