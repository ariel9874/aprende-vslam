#!/usr/bin/env python3
"""Descarga el dataset del nivel: TUM RGB-D freiburg2_xyz (~2.1 GB).

OJO al tamaño: fr2_xyz es la secuencia más larga del curso (3669 frames).
Es idempotente: si el dataset ya está extraído en data/, no hace nada. Si ya
lo tienes en otra ruta, no corras esto: pasa `--root <ruta>` a los scripts.

Con `--fr1-desk` baja además freiburg1_desk (~330 MB): la secuencia que este
sistema NO puede con ella — el ejercicio 1 del nivel.

Uso:
    python descarga_datos.py
    python descarga_datos.py --fr1-desk
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

SECUENCIAS = {
    "fr2_xyz": ("https://cvg.cit.tum.de/rgbd/dataset/freiburg2/"
                "rgbd_dataset_freiburg2_xyz.tgz",
                "rgbd_dataset_freiburg2_xyz", "~2.1 GB"),
    "fr1_desk": ("https://cvg.cit.tum.de/rgbd/dataset/freiburg1/"
                 "rgbd_dataset_freiburg1_desk.tgz",
                 "rgbd_dataset_freiburg1_desk", "~330 MB"),
}


def descargar(nombre: str) -> None:
    url, carpeta, tam = SECUENCIAS[nombre]
    dest = DATA / carpeta
    if (dest / "rgb").is_dir():
        n = len(list((dest / "rgb").glob("*.png")))
        print(f"Ya esta: {dest} ({n} imagenes rgb). Nada que hacer.")
        return

    DATA.mkdir(parents=True, exist_ok=True)
    tgz = DATA / f"{carpeta}.tgz"
    if not tgz.exists():
        print(f"Descargando {url}")
        print(f"({tam}; puede tardar un buen rato)")

        def progreso(bloques, tam_bloque, total):
            if total > 0 and bloques % 512 == 0:
                pct = min(100.0, 100.0 * bloques * tam_bloque / total)
                print(f"  {pct:5.1f}%", end="\r")

        urllib.request.urlretrieve(url, tgz, reporthook=progreso)
        print("  100.0%")

    print(f"Extrayendo {tgz.name} ...")
    with tarfile.open(tgz) as tar:
        tar.extractall(DATA)
    print(f"OK: dataset en {dest}")
    print("Puedes borrar el .tgz si quieres recuperar el espacio.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fr1-desk", action="store_true",
                        help="bajar tambien freiburg1_desk (ejercicio 1)")
    args = parser.parse_args()

    descargar("fr2_xyz")
    if args.fr1_desk:
        descargar("fr1_desk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
