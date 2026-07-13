#!/usr/bin/env python3
"""Examen del nivel 07: la pose recuperada contra ground truth exacto.

Genera los datos si faltan y verifica los tres experimentos del nivel.
Si todo pasa: NIVEL 07: VERIFICADO.

Uso:
    python verificacion.py
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
DATOS = AQUI / "data" / "secuencia"

spec = importlib.util.spec_from_file_location("n7", AQUI / "07_geometria_epipolar.py")
n7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n7)

fallos = []


def check(nombre: str, ok: bool, detalle: str = "") -> None:
    estado = "OK " if ok else "FALLO"
    print(f"  [{estado}] {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def main() -> int:
    if not (DATOS / "images").is_dir():
        print("Generando la secuencia sintetica...")
        r = subprocess.run([sys.executable, str(AQUI / "genera_datos.py")])
        if r.returncode != 0:
            raise SystemExit("genera_datos.py fallo")

    K = n7.leer_calibracion()
    gt = n7.leer_gt()
    print("Verificando sobre la secuencia sintetica (GT exacto)\n")

    # 1. Recuperacion de pose en el par 0 -> 6.
    gray_a, gray_b = n7.cargar_frame(0), n7.cargar_frame(6)
    pts_a, pts_b = n7.emparejar(gray_a, gray_b)
    r = n7.estimar_pose(pts_a, pts_b, K)
    check("hay inliers de sobra (>=100)", r["n_ransac"] >= 100,
          str(r["n_ransac"]))

    T_b_a = n7.invert_se3(gt[6]) @ gt[0]
    err_R = n7.angulo_entre_rotaciones(r["R"], T_b_a[:3, :3])
    err_t = n7.angulo_entre_vectores(r["t"], T_b_a[:3, 3])
    check("error de rotacion < 1 grado", err_R < 1.0, f"{err_R:.3f}")
    check("error de direccion de t < 5 grados", err_t < 5.0, f"{err_t:.3f}")

    # La escala NO es un numero libre: recoverPose la fija a 1 por convencion.
    check("||t|| == 1 (la convencion monocular)",
          abs(np.linalg.norm(r["t"]) - 1.0) < 1e-6,
          f"{np.linalg.norm(r['t']):.6f}")

    # 2. La E estimada explica los puntos: distancia mediana a la epipolar.
    inl_a, inl_b = pts_a[r["mask"]], pts_b[r["mask"]]
    if len(inl_a) < 12:
        inl_a, inl_b = pts_a, pts_b
    d_med = n7.distancia_a_epipolares(inl_a, inl_b, r["E"], K)
    check("distancia mediana a epipolares < 1.5 px", d_med < 1.5,
          f"{d_med:.2f} px")

    # 3. La trampa de recoverPose (frames consecutivos, prof/baseline ~110-280):
    #    la sobrecarga basica debe perder MUCHOS inliers frente a distanceThresh.
    qa, qb = n7.emparejar(n7.cargar_frame(0), n7.cargar_frame(1))
    con = n7.estimar_pose(qa, qb, K, usar_dist_thresh=True)
    sin = n7.estimar_pose(qa, qb, K, usar_dist_thresh=False)
    check("la trampa se reproduce (basico < 50% de con-thresh)",
          sin["n_inliers"] < 0.5 * con["n_inliers"],
          f"{sin['n_inliers']} vs {con['n_inliers']}")

    print()
    if fallos:
        print(f"NIVEL 07: {len(fallos)} fallo(s): {', '.join(fallos)}")
        return 1
    print("NIVEL 07: VERIFICADO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
