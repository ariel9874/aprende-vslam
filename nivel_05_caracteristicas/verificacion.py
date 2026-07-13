#!/usr/bin/env python3
"""Examen del nivel 05: Harris propio, detectores e invarianza medida.

Umbrales calibrados sobre el primer frame de fr1_xyz (medidos: Harris 100%,
ORB rot 86% / esc 0.5x 65%, GFTT esc 37%); se dejan margenes generosos por
si corres sobre otra imagen. Si todo pasa: NIVEL 05: VERIFICADO.

Uso:
    python verificacion.py [--root <secuencia_TUM>] [--imagen foto.png]
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent
DATASET_DEFAULT = AQUI / "data" / "rgbd_dataset_freiburg1_xyz"

# El script principal empieza por digito: se carga por ruta.
spec = importlib.util.spec_from_file_location("n5", AQUI / "05_caracteristicas.py")
n5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n5)

fallos = []


def check(nombre: str, ok: bool, detalle: str = "") -> None:
    estado = "OK " if ok else "FALLO"
    print(f"  [{estado}] {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagen")
    parser.add_argument("--root")
    args = parser.parse_args()

    ruta = n5.encontrar_imagen(args)
    gray = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
    print(f"Verificando con {ruta.name}\n")

    # 1. El Harris propio reproduce al de OpenCV (top-50, radio 3 px).
    top_mano = n5.top_esquinas(n5.harris_a_mano(gray), 50)
    top_cv2 = n5.top_esquinas(
        cv2.cornerHarris(gray.astype(np.float32), blockSize=3, ksize=3, k=0.04), 50)
    coinc = n5.coincidencia(top_mano, top_cv2)
    check("Harris propio == cv2 (>=80% del top-50)", coinc >= 0.80,
          f"{100*coinc:.0f}% (medido en fr1: 100%)")

    # 2. Los detectores encuentran material de sobra en una imagen real.
    orb = cv2.ORB_create(nfeatures=2000)
    n_orb = len(orb.detect(gray, None))
    check("ORB detecta >=1000 kps", n_orb >= 1000, str(n_orb))

    # 3. Invarianza medida: rotar 30 grados no destruye la deteccion.
    h, w = gray.shape
    M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), 30.0, 1.0)
    r_rot = n5.repetibilidad(gray, n5.como_detector(orb), M_rot)
    check("repetibilidad ORB con rot 30 >= 60%", r_rot >= 0.60,
          f"{100*r_rot:.0f}% (medido en fr1: 86%)")

    # 4. La piramide de escalas se nota: ORB aguanta 0.5x mejor que GFTT
    #    (detector de un solo nivel). Este es EL numero de la leccion.
    M_esc = cv2.getRotationMatrix2D((w / 2, h / 2), 0.0, 0.5)
    gftt = cv2.GFTTDetector_create(maxCorners=1000, qualityLevel=0.01)
    r_orb = n5.repetibilidad(gray, n5.como_detector(orb), M_esc)
    r_gftt = n5.repetibilidad(gray, n5.como_detector(gftt), M_esc)
    check("a escala 0.5x, ORB > GFTT", r_orb > r_gftt,
          f"ORB {100*r_orb:.0f}% vs GFTT {100*r_gftt:.0f}% (medido: 65 vs 37)")

    print()
    if fallos:
        print(f"NIVEL 05: {len(fallos)} fallo(s): {', '.join(fallos)}")
        return 1
    print("NIVEL 05: VERIFICADO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
