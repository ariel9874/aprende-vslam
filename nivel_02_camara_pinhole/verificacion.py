#!/usr/bin/env python3
"""Examen del nivel 02: la proyeccion pinhole, exacta.

Si todo pasa: NIVEL 02: VERIFICADO.

Uso:
    python verificacion.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("n2", AQUI / "02_pinhole.py")
n2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n2)

fallos = []


def check(nombre: str, ok: bool, detalle: str = "") -> None:
    estado = "OK " if ok else "FALLO"
    print(f"  [{estado}] {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def main() -> int:
    cam = n2.PinholeCamera(n2.FX, n2.FY, n2.CX, n2.CY)
    print("Verificando la camara pinhole (K de TUM fr1)\n")

    # 1. El eje optico cae exactamente en el punto principal.
    uv = cam.project(np.array([[0.0, 0.0, 2.0]]))[0]
    check("eje optico -> (cx, cy)",
          abs(uv[0] - n2.CX) < 1e-9 and abs(uv[1] - n2.CY) < 1e-9,
          f"({uv[0]:.3f}, {uv[1]:.3f})")

    # 2. u - cx = fx*X/Z, comprobado con numeros redondos.
    uv = cam.project(np.array([[1.0, 0.0, 2.0]]))[0]
    check("1 m a la derecha a 2 m -> cx + fx/2",
          abs(uv[0] - (n2.CX + n2.FX / 2)) < 1e-9, f"u = {uv[0]:.3f}")

    # 3. Ida y vuelta exacta cuando se conoce Z.
    X = np.array([[0.3, -0.2, 1.7], [-0.8, 0.5, 4.2], [0.0, 0.0, 0.9]])
    rec = cam.backproject(cam.project(X), X[:, 2])
    check("backproject(project(X), Z) == X", np.abs(rec - X).max() < 1e-9,
          f"err max {np.abs(rec - X).max():.2e}")

    # 4. La profundidad NO es observable: mismo rayo -> mismo pixel.
    rayo = np.array([[0.3, 0.2, 1.5], [0.6, 0.4, 3.0]])
    duv = np.abs(np.diff(cam.project(rayo), axis=0)).max()
    check("dos puntos del mismo rayo -> mismo pixel", duv < 1e-9,
          f"dif {duv:.2e} px")

    # 5. El tamano proyectado es lineal en fx (y en 1/Z).
    c2 = n2.PinholeCamera(2 * n2.FX, 2 * n2.FY, n2.CX, n2.CY)
    seg = np.array([[0.0, 0.0, 3.0], [0.4, 0.0, 3.0]])
    w1 = cam.project(seg)[1, 0] - cam.project(seg)[0, 0]
    w2 = c2.project(seg)[1, 0] - c2.project(seg)[0, 0]
    check("doblar fx dobla el tamano en px", abs(w2 / w1 - 2.0) < 1e-9,
          f"ratio {w2/w1:.6f}")
    lejos = seg * np.array([1.0, 1.0, 2.0])
    w3 = cam.project(lejos)[1, 0] - cam.project(lejos)[0, 0]
    check("doblar Z parte el tamano a la mitad", abs(w1 / w3 - 2.0) < 1e-9,
          f"ratio {w1/w3:.6f}")

    print()
    if fallos:
        print(f"NIVEL 02: {len(fallos)} fallo(s): {', '.join(fallos)}")
        return 1
    print("NIVEL 02: VERIFICADO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
