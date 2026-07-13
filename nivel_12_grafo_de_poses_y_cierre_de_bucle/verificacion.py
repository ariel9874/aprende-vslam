#!/usr/bin/env python3
"""Examen del nivel 12: grafo de poses, Sim(3) y cierre de bucle.

Sin dataset. Numeros medidos: Lie vs serie < 1e-12; bucle 0.71 -> 0.095 m;
Strasdat SE(3) 0.80 (empeora) vs Sim(3) 0.00; falso positivo 3.54 sin Huber.
Si todo pasa: NIVEL 12: VERIFICADO.

Uso:
    python verificacion.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

from lie import se3_exp, se3_inv, se3_log, sim3_exp, sim3_inv, sim3_log
from pose_graph import GrafoDePoses

spec = importlib.util.spec_from_file_location("n12", AQUI / "12_grafo_de_poses.py")
n12 = importlib.util.module_from_spec(spec)
sys.modules["n12"] = n12
spec.loader.exec_module(n12)

fallos = []


def check(nombre: str, ok: bool, detalle: str = "") -> None:
    estado = "OK " if ok else "FALLO"
    print(f"  [{estado}] {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def main() -> int:
    print("Verificando el grafo de poses (geometria simulada exacta)\n")

    # 1. El algebra de Lie es correcta CONTRA LA SERIE de matrices (la verdad
    #    numerica: si las formulas cerradas estan mal, esto lo caza).
    e_se3, e_sim3 = n12.exp1_lie()
    print()
    check("SE(3):  Exp/Log == exponencial por serie", e_se3 < 1e-9,
          f"err max {e_se3:.1e}")
    check("Sim(3): Exp/Log == exponencial por serie", e_sim3 < 1e-9,
          f"err max {e_sim3:.1e}")

    # Los casos limite por separado (theta=0 y lambda=0 son donde las formulas
    # ingenuas dividen por cero).
    err = 0.0
    for xi in [np.zeros(7), np.r_[0.3, -0.2, 0.1, 0, 0, 0, 0.5],
               np.r_[0.3, -0.2, 0.1, 0.4, 0.1, -0.2, 0.0]]:
        S = sim3_exp(xi)
        err = max(err, float(np.abs(sim3_log(S) - xi).max()))
        err = max(err, float(np.abs(sim3_inv(S) @ S - np.eye(4)).max()))
    check("Sim(3) en los casos limite (theta=0, lambda=0)", err < 1e-9,
          f"err max {err:.1e}")

    # 2. El cierre de bucle deshace la deriva.
    antes, despues = n12.exp2_cierre_de_bucle(AQUI / "salida")
    print()
    check("el bucle reduce el ATE >=5x", antes / despues >= 5.0,
          f"{antes:.2f} m -> {despues:.3f} m")
    check("ATE final < 20 cm", despues < 0.20, f"{despues*100:.1f} cm")

    # 3. EL EXPERIMENTO DEL NIVEL (Strasdat): con deriva de ESCALA, el grafo
    #    SE(3) empeora y el Sim(3) arregla.
    odom, se3, sim3 = n12.exp3_strasdat()
    print()
    check("el grafo SE(3) NO mejora la odometria (o la empeora)", se3 >= odom * 0.95,
          f"odometria {odom:.2f} m -> SE(3) {se3:.2f} m")
    check("el grafo Sim(3) SI la arregla (<10 cm)", sim3 < 0.10,
          f"{sim3:.3f} m (medido: 0.00)")
    check("Sim(3) es >=5x mejor que SE(3)", se3 / max(sim3, 1e-6) >= 5.0,
          f"{se3:.2f} vs {sim3:.3f} m")

    # 4. El falso positivo: sin robustez destroza el grafo; rechazarlo lo salva.
    r = n12.exp4_falso_positivo()
    print()
    check("el bucle falso destroza el grafo sin robustez",
          r["cuadratico (sin Huber)"] > 5 * r["rechazado (verificacion)"],
          f"{r['cuadratico (sin Huber)']:.2f} m vs "
          f"{r['rechazado (verificacion)']:.2f} m si se rechaza")
    check("Huber con umbral razonable NO basta (la leccion incomoda)",
          r["Huber delta=1.0"] > 2.0,
          f"{r['Huber delta=1.0']:.2f} m: sigue siendo un desastre")
    check("rechazar la arista SI funciona", r["rechazado (verificacion)"] < 0.25,
          f"{r['rechazado (verificacion)']:.2f} m")

    print()
    if fallos:
        print(f"NIVEL 12: {len(fallos)} fallo(s): {', '.join(fallos)}")
        return 1
    print("NIVEL 12: VERIFICADO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
