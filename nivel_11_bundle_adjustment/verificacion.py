#!/usr/bin/env python3
"""Examen del nivel 11: el BA y sus cinco lecciones.

Sin dataset: geometria sintetica exacta. Numeros medidos: reproyeccion
22.6 -> 0.47 px, gauge 1 ancla = 1.1494 / 2 anclas = 1.0000, Huber mejora
84%, min_obs 18.8 -> 6.4 cm.
Si todo pasa: NIVEL 11: VERIFICADO.

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

from bundle_adjustment import (bundle_adjustment, invert_se3,
                               residual_and_jacobians, se3_exp)

spec = importlib.util.spec_from_file_location("n11", AQUI / "11_bundle_adjustment.py")
n11 = importlib.util.module_from_spec(spec)
sys.modules["n11"] = n11
spec.loader.exec_module(n11)

fallos = []


def check(nombre: str, ok: bool, detalle: str = "") -> None:
    estado = "OK " if ok else "FALLO"
    print(f"  [{estado}] {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def main() -> int:
    K = n11.K
    print("Verificando el bundle adjustment (geometria sintetica exacta)\n")

    # 1. Los JACOBIANOS son correctos (contra diferencias finitas). Si esto
    #    falla, todo lo demas es ruido: es el primer test que hay que escribir.
    gt_poses, gt_pts, obs = n11.escena()
    T, X, uv = gt_poses[2], gt_pts[10], np.array([300.0, 250.0])
    r0, Jp, Jx = residual_and_jacobians(K, invert_se3(T), X, uv)
    eps = 1e-7
    Jp_fd = np.zeros((2, 6))
    for i in range(6):
        d = np.zeros(6)
        d[i] = eps
        r2, _, _ = residual_and_jacobians(K, invert_se3(T @ se3_exp(d)), X, uv)
        Jp_fd[:, i] = (r2 - r0) / eps
    Jx_fd = np.zeros((2, 3))
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps
        r2, _, _ = residual_and_jacobians(K, invert_se3(T), X + d, uv)
        Jx_fd[:, i] = (r2 - r0) / eps
    e_jp = float(np.abs(Jp - Jp_fd).max())
    e_jx = float(np.abs(Jx - Jx_fd).max())
    check("J_pose  == diferencias finitas", e_jp < 1e-3, f"err max {e_jp:.1e}")
    check("J_punto == diferencias finitas", e_jx < 1e-3, f"err max {e_jx:.1e}")

    # 2. El BA converge: baja la reproyeccion Y se acerca a la VERDAD (que es
    #    lo que de verdad importa: bajar el costo es facil, acertar no).
    px, ep, ec = n11.exp1_converge(AQUI / "salida")
    print()
    check("reproyeccion final < 1 px", px < 1.0, f"{px:.2f} px (medido: 0.47)")
    check("error de los puntos < 10 cm", ep < 10.0, f"{ep:.1f} cm (medido: 5.4)")
    check("error de las camaras < 5 cm", ec < 5.0, f"{ec:.1f} cm (medido: 1.8)")

    # 3. EL EXPERIMENTO DEL NIVEL: el gauge tiene 7 gdl.
    esc1, esc2 = n11.exp2_gauge()
    print()
    check("con 1 ancla la escala queda LIBRE (sigue en ~1.15)",
          abs(esc1 - 1.15) < 0.02, f"{esc1:.4f} (medido: 1.1494)")
    check("con 2 anclas la escala se RECUPERA (~1.00)",
          abs(esc2 - 1.00) < 0.02, f"{esc2:.4f} (medido: 1.0000)")

    # 4. El agujero de costo: sin penalizacion, el optimizador esconde puntos
    #    detras de la camara; con penalizacion, ninguno acaba ahi.
    _, detras_mal, _, detras_bien = n11.exp3_agujero_de_costo()
    print()
    check("con el agujero, hay puntos escondidos DETRAS", detras_mal > 0,
          f"{detras_mal} puntos (medido: 2)")
    check("con penalizacion, NINGUNO acaba detras", detras_bien == 0,
          f"{detras_bien} puntos")

    # 5. Huber amansa a los outliers, pero no los elimina.
    e_cuad, e_huber = n11.exp4_huber()
    print()
    check("Huber mejora >=50% sobre el cuadratico puro",
          e_huber < 0.5 * e_cuad, f"{e_cuad:.1f} -> {e_huber:.1f} cm")
    check("...pero NO alcanza el error sin outliers (~5 cm)", e_huber > 8.0,
          f"{e_huber:.1f} cm: queda sesgo, Huber no rechaza")

    # 6. Un punto con una sola observacion se desliza por su rayo.
    e_sin, e_con = n11.exp5_una_observacion()
    print()
    check("min_obs=2 protege a los puntos huerfanos", e_con < 0.6 * e_sin,
          f"{e_sin:.1f} -> {e_con:.1f} cm")

    print()
    if fallos:
        print(f"NIVEL 11: {len(fallos)} fallo(s): {', '.join(fallos)}")
        return 1
    print("NIVEL 11: VERIFICADO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
