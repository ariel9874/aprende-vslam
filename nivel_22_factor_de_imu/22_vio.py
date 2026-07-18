#!/usr/bin/env python3
"""
Nivel 22 (bonus) — El factor de IMU: por qué VIO ganó el mercado
================================================================

Un vehículo 2D sin odometría: visión a 4 Hz + IMU a 100 Hz con sesgo. Y en
la peor curva del circuito, un APAGÓN visual de 5 segundos (la ráfaga de
blur del nivel 17, en versión controlada). Cuatro corridas lo cuentan todo:

    python 22_vio.py         # la tabla + las graficas en salida/
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import grafo_vio
from mundo_imu import BIAS_GYRO_REAL, generar, rmse_xy
from preintegracion import integrar_muerto

AQUI = Path(__file__).resolve().parent


def rmse_en(idx, poses, gt):
    d = poses[idx, :2] - gt[idx, :2]
    return float(np.sqrt((d ** 2).sum(axis=1).mean()))


def main() -> int:
    m = generar()
    gt = m["gt"]
    apagados = np.array(m["apagados"])
    print(f"El mundo: {len(gt)} keyframes a 4 Hz, {len(m['tramos'])} tramos "
          f"de IMU a 100 Hz,\n{len(m['obs'])} observaciones -- y "
          f"{len(apagados)} keyframes APAGADOS en plena curva\n")

    filas = []

    # 1) la IMU sola (dead reckoning): por que nadie navega solo con IMU
    dr = integrar_muerto(m["tramos"], np.zeros(3), np.array([1.0, 0.0]))
    filas.append(("IMU sola (dead reckoning)", rmse_xy(dr, gt), None,
                  "el sesgo rota el mundo; la deriva es cuadratica"))

    def correr(nombre, nota, **kw):
        t0 = time.perf_counter()
        r = grafo_vio.optimizar(m, **kw)
        filas.append((nombre, rmse_xy(r["poses"], gt),
                      rmse_en(apagados, r["poses"], gt), nota))
        return r

    r_coast = correr("vision + coast (sin IMU)",
                     "el factor de velocidad constante no sabe girar",
                     usar_imu=False)
    r_sinb = correr("VIO sin estimar el sesgo",
                    "el giroscopo miente 0.03 rad/s y nadie lo corrige",
                    usar_imu=True, estimar_bias=False)
    r_vio = correr("VIO completo (sesgo estimado)",
                   f"sesgo recuperado: {0:.0f}", usar_imu=True)
    filas[-1] = (filas[-1][0], filas[-1][1], filas[-1][2],
                 f"sesgo recuperado {r_vio['biases'][-1]:.4f} "
                 f"(real {BIAS_GYRO_REAL})")

    print(f"{'configuracion':30s} {'RMSE total':>10s} {'en el APAGON':>13s}")
    for nombre, r_t, r_a, nota in filas:
        apag = f"{100*r_a:10.1f}cm" if r_a is not None else "         -"
        print(f"{nombre:30s} {100*r_t:8.1f}cm {apag}   {nota}")

    print("\nLa curva del apagon es EL argumento: el coast asume velocidad")
    print("constante y el robot giro 90 grados a oscuras. La IMU no asume:")
    print("MIDE el giro (giroscopo) y la centripeta (acelerometro).")

    # ── graficas ─────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(13, 5))
        ax = axs[0]
        ax.plot(gt[:, 0], gt[:, 1], "k--", lw=1.5, label="verdad")
        ax.plot(r_coast["poses"][:, 0], r_coast["poses"][:, 1], lw=1.2,
                label="vision + coast")
        ax.plot(r_vio["poses"][:, 0], r_vio["poses"][:, 1], lw=1.2,
                label="VIO")
        ax.plot(gt[apagados, 0], gt[apagados, 1], "r.", ms=8,
                label="apagon visual")
        ax.scatter(m["landmarks"][:, 0], m["landmarks"][:, 1], marker="*",
                   c="tab:red", s=50)
        ax.legend(fontsize=9), ax.axis("equal"), ax.grid(alpha=0.3)
        ax.set_title("el apagon en la curva: coast vs IMU")
        ax2 = axs[1]
        for r, lab in ((r_coast, "vision + coast"), (r_sinb, "VIO sin sesgo"),
                       (r_vio, "VIO completo")):
            err = np.linalg.norm(r["poses"][:, :2] - gt[:, :2], axis=1)
            ax2.plot(100 * err, lw=1.1, label=lab)
        for i in apagados:
            ax2.axvspan(i - 0.5, i + 0.5, color="0.85", zorder=0)
        ax2.set_xlabel("keyframe"), ax2.set_ylabel("error [cm]")
        ax2.legend(fontsize=9), ax2.grid(alpha=0.3)
        ax2.set_title("error por keyframe (gris: apagon)")
        salida = AQUI / "salida"
        salida.mkdir(exist_ok=True)
        fig.savefig(salida / "vio.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\nGraficas: {salida / 'vio.png'}")
    except ImportError:
        pass

    print("Ahora corre `python verificacion.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
