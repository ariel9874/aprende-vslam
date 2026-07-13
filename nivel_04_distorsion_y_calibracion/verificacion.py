#!/usr/bin/env python3
"""Examen del nivel 04: la calibracion, contra la verdad.

Genera las vistas si faltan, calibra y compara con los parametros reales de
la camara simulada. Numeros medidos: RMS 0.24 px, intrinsecos al <0.5%,
campo de distorsion a 0.33 px de media, recta del borde 1.92 -> 0.08 px.
Si todo pasa: NIVEL 04: VERIFICADO.

Uso:
    python verificacion.py
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent
DATOS = AQUI / "data" / "tablero"

spec = importlib.util.spec_from_file_location("n4", AQUI / "04_calibracion.py")
n4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n4)

fallos = []


def check(nombre: str, ok: bool, detalle: str = "") -> None:
    estado = "OK " if ok else "FALLO"
    print(f"  [{estado}] {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def main() -> int:
    if not list(DATOS.glob("vista_*.png")):
        print("Generando las vistas del tablero...")
        r = subprocess.run([sys.executable, str(AQUI / "genera_tablero.py")])
        if r.returncode != 0:
            raise SystemExit("genera_tablero.py fallo")

    rutas = sorted(DATOS.glob("vista_*.png"))
    K_gt, dist_gt, _ = n4.leer_gt()
    print(f"Verificando con {len(rutas)} vistas\n")

    # 1. El tablero se detecta en TODAS las vistas (si no, el generador o el
    #    detector estan mal — con vistas sinteticas no hay excusa).
    objs, imgs, shape = n4.detectar_esquinas(rutas)
    check("tablero detectado en todas las vistas", len(imgs) == len(rutas),
          f"{len(imgs)}/{len(rutas)}")

    # 2. La calibracion converge con error de reproyeccion sano.
    rms, K_est, dist_est, _, _ = cv2.calibrateCamera(objs, imgs, shape, None, None)
    dist_est = dist_est.ravel()
    check("error de reproyeccion < 0.5 px", rms < 0.5, f"{rms:.3f} px (medido: 0.242)")

    # 3. Los intrinsecos recuperan la verdad (<1.5%: el margen del ruido de
    #    deteccion sub-pixel; medido 0.1-0.5%).
    for nombre, est, gt in [("fx", K_est[0, 0], K_gt[0, 0]),
                            ("fy", K_est[1, 1], K_gt[1, 1]),
                            ("cx", K_est[0, 2], K_gt[0, 2]),
                            ("cy", K_est[1, 2], K_gt[1, 2])]:
        err_pct = 100 * abs(est - gt) / gt
        check(f"{nombre} dentro del 1.5% de la verdad", err_pct < 1.5,
              f"{est:.2f} vs {gt:.2f} ({err_pct:.2f}%)")

    # 4. EL criterio honesto: el CAMPO de distorsion, no los coeficientes
    #    sueltos (que estan correlacionados y pueden diferir describiendo la
    #    misma curva — en esta corrida k2 difiere un 4% y da igual).
    dif = np.linalg.norm(n4.campo_de_distorsion(K_gt, dist_gt)
                         - n4.campo_de_distorsion(K_est, dist_est), axis=1)
    check("campo de distorsion: error medio < 1 px", dif.mean() < 1.0,
          f"{dif.mean():.3f} px (medido: 0.328)")

    # 5. Rectificar ENDEREZA de verdad: la recta que roza el borde izquierdo
    #    (donde la lente muerde) pasa de combarse ~2 px a ser recta.
    def residuo(pts):
        c = pts - pts.mean(0)
        _, _, Vt = np.linalg.svd(c)
        return float(np.sqrt((np.abs(c @ Vt[1]) ** 2).mean()))

    ideal = np.stack([np.full(60, 15.0), np.linspace(10, 470, 60)], axis=1)
    xy = (ideal - [K_gt[0, 2], K_gt[1, 2]]) / [K_gt[0, 0], K_gt[1, 1]]
    real, _ = cv2.projectPoints(np.hstack([xy, np.ones((60, 1))]),
                                np.zeros(3), np.zeros(3), K_gt, dist_gt)
    real = real.reshape(-1, 2)
    rect = cv2.undistortPoints(real.reshape(-1, 1, 2), K_est, dist_est,
                               P=K_est).reshape(-1, 2)
    antes, despues = residuo(real), residuo(rect)
    check("la recta del borde se endereza (>=5x)", antes / max(despues, 1e-9) >= 5.0,
          f"{antes:.2f} -> {despues:.2f} px (medido: 1.92 -> 0.08)")

    print()
    if fallos:
        print(f"NIVEL 04: {len(fallos)} fallo(s): {', '.join(fallos)}")
        return 1
    print("NIVEL 04: VERIFICADO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
