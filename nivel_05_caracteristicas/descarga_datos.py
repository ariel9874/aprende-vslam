#!/usr/bin/env python3
"""Descarga el dataset del nivel: TUM RGB-D freiburg1_xyz (~450 MB).

Es idempotente: si el dataset ya esta extraido en data/, no hace nada.
Si ya lo tienes en otra ruta (p. ej. lo bajaste para otro nivel), no hace
falta correr esto: pasa `--root <ruta>` a los scripts del nivel.

Uso:
    python descarga_datos.py
"""

from __future__ import annotations

import sys
import tarfile
import urllib.request
from pathlib import Path

URL = ("https://cvg.cit.tum.de/rgbd/dataset/freiburg1/"
       "rgbd_dataset_freiburg1_xyz.tgz")
DATA = Path(__file__).resolve().parent / "data"
DEST = DATA / "rgbd_dataset_freiburg1_xyz"


def main() -> int:
    if (DEST / "rgb").is_dir():
        n = len(list((DEST / "rgb").glob("*.png")))
        print(f"Ya esta: {DEST} ({n} imagenes rgb). Nada que hacer.")
        return 0

    DATA.mkdir(parents=True, exist_ok=True)
    tgz = DATA / "rgbd_dataset_freiburg1_xyz.tgz"

    if not tgz.exists():
        print(f"Descargando {URL}")
        print("(~450 MB; puede tardar varios minutos)")

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
