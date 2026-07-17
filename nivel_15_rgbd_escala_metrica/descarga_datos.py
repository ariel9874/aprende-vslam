#!/usr/bin/env python3
"""Descarga el dataset del nivel: TUM RGB-D freiburg1_desk (~330 MB).

La misma secuencia handheld que DERROTÓ al nivel 14 (562 frames perdidos):
aquí, con el sensor de profundidad, se cruza entera. Incluye los mapas de
profundidad (depth/ y depth.txt) — este nivel los necesita.

Es idempotente: si el dataset ya está extraído en data/, no hace nada. Si ya
lo tienes en otra ruta (p. ej. lo bajaste para el ejercicio 1 del nivel 14),
no corras esto: pasa `--root <ruta>` a los scripts.

Uso:
    python descarga_datos.py
"""

from __future__ import annotations

import sys
import tarfile
import urllib.request
from pathlib import Path

URL = ("https://cvg.cit.tum.de/rgbd/dataset/freiburg1/"
       "rgbd_dataset_freiburg1_desk.tgz")
DATA = Path(__file__).resolve().parent / "data"
DEST = DATA / "rgbd_dataset_freiburg1_desk"


def main() -> int:
    if (DEST / "rgb").is_dir() and (DEST / "depth").is_dir():
        n = len(list((DEST / "rgb").glob("*.png")))
        print(f"Ya esta: {DEST} ({n} imagenes rgb). Nada que hacer.")
        return 0

    DATA.mkdir(parents=True, exist_ok=True)
    tgz = DATA / "rgbd_dataset_freiburg1_desk.tgz"

    if not tgz.exists():
        print(f"Descargando {URL}")
        print("(~330 MB; puede tardar varios minutos)")

        def progreso(bloques, tam_bloque, total):
            if total > 0 and bloques % 512 == 0:
                pct = min(100.0, 100.0 * bloques * tam_bloque / total)
                print(f"  {pct:5.1f}%", end="\r")

        urllib.request.urlretrieve(URL, tgz, reporthook=progreso)
        print("  100.0%")

    print(f"Extrayendo {tgz.name} ...")
    with tarfile.open(tgz) as tar:
        tar.extractall(DATA)
    print(f"OK: dataset en {DEST}")
    print("Puedes borrar el .tgz si quieres recuperar el espacio.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
