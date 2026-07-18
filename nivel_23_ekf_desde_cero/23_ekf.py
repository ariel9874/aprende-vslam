#!/usr/bin/env python3
"""Nivel 23 (bonus) — El EKF desde cero (y el EKF de estado de error).

Cuatro actos, cada uno agrega UNA sola cosa — y cada uno es un script que
puedes correr (y leer) por separado:

    fusion_1d.py         1. fusionar dos numeros (K es una media ponderada)
    kalman_lineal.py     2. el estado se mueve (predecir + corregir)
    ekf_localizacion.py  3. el mundo no es lineal (jacobianos; el bug de ±π)
    ekf_de_error.py      4. el EKF de error (el filtro de los VIO reales)

Este driver los corre seguidos, arma la tabla del nivel y las gráficas.

Uso:
    python 23_ekf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import ekf_de_error
import ekf_localizacion
import fusion_1d
import kalman_lineal
import mundo_imu


def main() -> int:
    raya = "=" * 72
    for acto in (fusion_1d, kalman_lineal, ekf_localizacion, ekf_de_error):
        print(raya)
        acto.main()
        print()

    # ── la tabla del nivel ───────────────────────────────────────────────────
    print(raya)
    print("La tabla del nivel (los numeros que hay que poder contar):\n")
    print("  acto  leccion                                        numero")
    print("  1     recursivo == batch                             ~1e-15")
    print("  1     sigma cae como 1/sqrt(N)                       0.0158 == 0.0158")
    print("  2     el filtro ES el grafo lineal                   ~1e-11")
    print("  2     velocidad estimada sin sensor                  17x vs derivar")
    print("  3     EKF vs dead reckoning                          2.0 vs 41 cm")
    print("  3     el bug del angulo (sin envolver)               80 cm (39x peor)")
    print("  4     ESKF: total / en el apagon                     7.2 / 17.2 cm")
    print("  4     sesgo descubierto EN VIVO                      0.0294 (real 0.03)")
    print("  4     filtro vs smoother (nivel 22, mismo mundo)     7.2 vs 4.7 cm")

    # ── las graficas ─────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(13, 10))

    # (a) acto 2: la respiracion de sigma
    sim = kalman_lineal.simular()
    kf = kalman_lineal.filtrar(sim["z"])
    resp = kf["respiracion"][:40]
    ax = axs[0, 0]
    t = np.arange(len(resp))
    ax.plot(np.repeat(t, 2), 100 * resp.ravel(), "-", color="tab:blue", lw=1)
    ax.plot(t, 100 * resp[:, 0], "^", color="tab:red", ms=4,
            label="tras PREDECIR (crece)")
    ax.plot(t, 100 * resp[:, 1], "v", color="tab:green", ms=4,
            label="tras CORREGIR (baja)")
    ax.set_title("Acto 2: la respiracion de sigma (el carrito)")
    ax.set_xlabel("paso")
    ax.set_ylabel("sigma de la posicion [cm]")
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) acto 3: el bug del angulo
    m3 = ekf_localizacion.generar()
    ekf3 = ekf_localizacion.correr_ekf(m3)
    mal3 = ekf_localizacion.correr_ekf(m3, con_envolver=False)
    ax = axs[0, 1]
    ax.plot(m3["gt"][:, 0], m3["gt"][:, 1], "k-", lw=2, label="verdad")
    ax.plot(ekf3[:, 0], ekf3[:, 1], "-", color="tab:green", lw=1.2,
            label="EKF (2.0 cm)")
    ax.plot(mal3[:, 0], mal3[:, 1], "-", color="tab:red", lw=1, alpha=0.8,
            label="sin envolver (80 cm)")
    ax.plot(*ekf_localizacion.LANDMARKS.T, "b*", ms=9)
    ax.set_title("Acto 3: el EKF -- y el bug del angulo")
    ax.axis("equal")
    ax.legend()
    ax.grid(alpha=0.3)

    # (c) acto 4: el ESKF cruza el apagon
    m4 = mundo_imu.generar()
    r4 = ekf_de_error.correr(m4)
    apag = np.array(m4["apagados"])
    ax = axs[1, 0]
    ax.plot(m4["gt"][:, 0], m4["gt"][:, 1], "k-", lw=2, label="verdad")
    ax.plot(r4["poses"][:, 0], r4["poses"][:, 1], "-", color="tab:green",
            lw=1.2, label="ESKF online (7.2 cm)")
    ax.plot(m4["gt"][apag, 0], m4["gt"][apag, 1], "r-", lw=5, alpha=0.35,
            label="APAGON (5 s)")
    ax.plot(*m4["landmarks"].T, "b*", ms=9, label="landmarks (estimados)")
    ax.set_title("Acto 4: el EKF de error (el mundo del nivel 22)")
    ax.axis("equal")
    ax.legend()
    ax.grid(alpha=0.3)

    # (d) acto 4: el sesgo, descubierto en vivo
    ax = axs[1, 1]
    t_kf = np.arange(len(r4["biases"])) * m4["dt_kf"]
    ax.plot(t_kf, r4["biases"], "-", color="tab:green", lw=1.5,
            label="sesgo estimado")
    ax.axhline(mundo_imu.BIAS_GYRO_REAL, color="k", ls="--",
               label=f"real ({mundo_imu.BIAS_GYRO_REAL})")
    ax.axvspan(t_kf[apag[0]], t_kf[apag[-1]], color="r", alpha=0.15,
               label="apagon")
    ax.set_title("Acto 4: el sesgo del giroscopo, descubierto EN VIVO")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("b_g [rad/s]")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    salida = AQUI / "salida"
    salida.mkdir(exist_ok=True)
    fig.savefig(salida / "ekf.png", dpi=110)
    print(f"\nGraficas: {salida / 'ekf.png'}")
    print("Ahora corre `python verificacion.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
