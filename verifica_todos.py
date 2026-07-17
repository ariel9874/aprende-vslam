#!/usr/bin/env python3
"""Corre el verificacion.py de TODOS los niveles construidos.

No es parte del curso (el alumno corre cada nivel por separado): es la red
de seguridad de quien MANTIENE el curso — un solo comando que dice si algo
se rompio. Cada nivel corre en su propia carpeta, como haria el alumno.

Uso:
    python verifica_todos.py
    python verifica_todos.py --root <ruta_a_TUM_fr1_xyz>   # reusa un dataset ya bajado
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent

# Los niveles que usan el dataset TUM (aceptan --root); el resto es autocontenido.
USAN_TUM = {"nivel_00_entorno_y_primeros_pixeles",
            "nivel_05_caracteristicas",
            "nivel_06_matching"}
# Los niveles 14-18 usan OTRAS secuencias: cada una con su flag. (Los
# examenes de 16, 19 y 20 no necesitan dataset.)
USAN_TUM_FR2 = {"nivel_14_datos_reales_tum",
                "nivel_18_ingenieria_de_tiempo_real"}
USAN_TUM_FR1DESK = {"nivel_15_rgbd_escala_metrica",
                    "nivel_17_features_aprendidas"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="secuencia TUM fr1_xyz ya descargada")
    parser.add_argument("--root-fr2", help="secuencia TUM fr2_xyz ya descargada")
    parser.add_argument("--root-fr1desk",
                        help="secuencia TUM fr1_desk ya descargada")
    args = parser.parse_args()

    niveles = sorted(d for d in AQUI.iterdir()
                     if d.is_dir() and d.name.startswith("nivel_")
                     and (d / "verificacion.py").exists())

    print(f"Verificando {len(niveles)} niveles construidos\n")
    resultados = []
    for d in niveles:
        cmd = [sys.executable, "verificacion.py"]
        if args.root and d.name in USAN_TUM:
            cmd += ["--root", args.root]
        if args.root_fr2 and d.name in USAN_TUM_FR2:
            cmd += ["--root", args.root_fr2]
        if args.root_fr1desk and d.name in USAN_TUM_FR1DESK:
            cmd += ["--root", args.root_fr1desk]
        t0 = time.perf_counter()
        r = subprocess.run(cmd, cwd=d, capture_output=True, text=True)
        dt = time.perf_counter() - t0
        ok = r.returncode == 0
        resultados.append((d.name, ok, dt))
        print(f"  [{'OK ' if ok else 'FALLO'}] {d.name:38s} {dt:5.1f} s")
        if not ok:
            print("  ---- salida del fallo ----")
            print("  " + "\n  ".join((r.stdout + r.stderr).strip().splitlines()[-15:]))

    n_ok = sum(1 for _, ok, _ in resultados if ok)
    print(f"\n{n_ok}/{len(resultados)} niveles VERIFICADOS")
    return 0 if n_ok == len(resultados) else 1


if __name__ == "__main__":
    raise SystemExit(main())
