#!/usr/bin/env python3
"""Examen del nivel 08: corre la odometría completa y verifica el ATE.

Genera los datos si faltan, ejecuta el pipeline entero como lo harías tú
(subproceso, sin trucos) y comprueba el número contra la referencia medida
del repo padre. Si todo pasa, imprime NIVEL 08: VERIFICADO.

Uso:
    python verificacion.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from evaluacion import ate, load_tum_positions

AQUI = Path(__file__).resolve().parent
DATOS = AQUI / "data" / "secuencia"
SALIDA = AQUI / "salida"

fallos = []


def check(nombre: str, ok: bool, detalle: str = "") -> None:
    estado = "OK " if ok else "FALLO"
    print(f"  [{estado}] {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def main() -> int:
    # 1. Datos: se generan si faltan (idempotente).
    if not (DATOS / "images").is_dir():
        print("Generando la secuencia sintetica...")
        r = subprocess.run([sys.executable, str(AQUI / "genera_datos.py")])
        if r.returncode != 0:
            raise SystemExit("genera_datos.py fallo")

    n_imgs = len(list((DATOS / "images").glob("*.png")))
    print(f"Datos: {n_imgs} imagenes en {DATOS / 'images'}\n")
    check("la secuencia tiene 80 frames", n_imgs == 80, str(n_imgs))

    # 2. Correr el pipeline ENTERO, como lo correria el alumno.
    print("\nCorriendo la odometria (unos segundos)...")
    r = subprocess.run([sys.executable, str(AQUI / "08_odometria_visual.py")],
                       capture_output=True, text=True)
    print()
    check("el pipeline termina sin errores", r.returncode == 0,
          "" if r.returncode == 0 else r.stderr.strip().splitlines()[-1])
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        return 1

    # 3. La trayectoria existe y esta completa (una pose por frame).
    tray = SALIDA / "trayectoria.txt"
    check("salida/trayectoria.txt existe", tray.exists())
    est = load_tum_positions(tray)
    gt = load_tum_positions(DATOS / "groundtruth.txt")
    check("una pose por frame", len(est) == len(gt), f"{len(est)} vs {len(gt)}")

    # 4. EL numero del nivel: ATE con alineacion de similitud.
    #    Referencia: ~13 cm (v0.1 del repo padre, mismo pipeline y escena).
    #    Toleramos hasta 20 cm: RANSAC es aleatorio y +-20% es normal.
    m = ate(est, gt)
    print(f"\n  ATE rmse = {m['rmse']*100:.1f} cm  "
          f"(media {m['mean']*100:.1f}, max {m['max']*100:.1f}, "
          f"{m['rmse_pct']:.1f}% del recorrido)")
    check("ATE < 20 cm (referencia ~13)", m["rmse"] < 0.20,
          f"{m['rmse']*100:.1f} cm")

    # 5. La deriva es REAL (la leccion del nivel): el error del ultimo tercio
    #    debe superar al del primero — sin optimizacion, el error se acumula.
    #    (Se mide sobre la trayectoria alineada globalmente.)
    from evaluacion import umeyama_alignment
    s, R, t = umeyama_alignment(est, gt)
    aligned = (s * (R @ est.T)).T + t
    err = np.linalg.norm(aligned - gt, axis=1)
    n = len(err) // 3
    check("la deriva crece (err fin > err inicio)", err[-n:].mean() > err[:n].mean(),
          f"inicio {err[:n].mean()*100:.1f} cm vs fin {err[-n:].mean()*100:.1f} cm")

    print()
    if fallos:
        print(f"NIVEL 08: {len(fallos)} fallo(s): {', '.join(fallos)}")
        return 1
    print("NIVEL 08: VERIFICADO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
