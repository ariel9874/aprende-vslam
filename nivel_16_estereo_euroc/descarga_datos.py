#!/usr/bin/env python3
"""Descarga el dataset del nivel: EuRoC V1_01_easy (~1.1 GB).

OJO: el host oficial de EuRoC (robotics.ethz.ch) lleva temporadas caído.
Se usa el mirror publicado en HuggingFace (`pepijn223/euroc-mirror`,
formato .zip idéntico al original) — la misma solución que documentó el
repo padre.

Es idempotente: si el dataset ya está extraído en data/, no hace nada.
El EXAMEN del nivel no necesita este dataset (fabrica su propio rig);
esta secuencia es para el driver (16_estereo.py).

Uso:
    python descarga_datos.py
"""

from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://huggingface.co/datasets/pepijn223/euroc-mirror/resolve/"
       "main/V1_01_easy.zip")
DATA = Path(__file__).resolve().parent / "data"
DEST = DATA / "V1_01_easy"


def main() -> int:
    if (DEST / "mav0" / "cam1" / "data.csv").is_file():
        print(f"Ya esta: {DEST}. Nada que hacer.")
        return 0

    DATA.mkdir(parents=True, exist_ok=True)
    zip_path = DATA / "V1_01_easy.zip"

    if not zip_path.exists():
        print(f"Descargando {URL}")
        print("(~1.1 GB; puede tardar un buen rato)")

        def progreso(bloques, tam_bloque, total):
            if total > 0 and bloques % 512 == 0:
                pct = min(100.0, 100.0 * bloques * tam_bloque / total)
                print(f"  {pct:5.1f}%", end="\r")

        urllib.request.urlretrieve(URL, zip_path, reporthook=progreso)
        print("  100.0%")

    print(f"Extrayendo {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as z:
        # El zip trae mav0/ en la raiz: extraer dentro de V1_01_easy/.
        primero = z.namelist()[0]
        destino = DATA if primero.startswith("V1_01_easy") else DEST
        z.extractall(destino)
    print(f"OK: dataset en {DEST}")
    print("Puedes borrar el .zip si quieres recuperar el espacio.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
