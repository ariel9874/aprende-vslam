#!/usr/bin/env python3
"""Examen del nivel 00: comprueba que dominas la imagen-como-matriz.

Corre las mismas operaciones del script principal y las verifica con
tolerancias. Si todo pasa, imprime NIVEL 00: VERIFICADO.

Uso:
    python verificacion.py                     # dataset del nivel
    python verificacion.py --root <secuencia>  # dataset en otra ruta
    python verificacion.py --imagen foto.png   # (relaja el check de tamano)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent
DATASET_DEFAULT = AQUI / "data" / "rgbd_dataset_freiburg1_xyz"

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

    es_tum = args.imagen is None
    if args.imagen:
        ruta = Path(args.imagen)
    else:
        root = Path(args.root) if args.root else DATASET_DEFAULT
        rgb = sorted((root / "rgb").glob("*.png"))
        if not rgb:
            raise SystemExit(f"No hay dataset en {root}. "
                             "Corre `python descarga_datos.py` o pasa --root.")
        ruta = rgb[0]

    print(f"Verificando con {ruta.name}\n")
    bgr = cv2.imread(str(ruta), cv2.IMREAD_COLOR)

    # 1. La imagen es la matriz que esperamos.
    check("la imagen carga", bgr is not None)
    if bgr is None:
        return 1
    check("dtype es uint8", bgr.dtype == np.uint8, str(bgr.dtype))
    check("tiene 3 canales", bgr.ndim == 3 and bgr.shape[2] == 3, str(bgr.shape))
    if es_tum:
        check("resolucion TUM 640x480", bgr.shape[:2] == (480, 640), str(bgr.shape[:2]))

    # 2. El gris a mano coincide con la referencia a +-1 (solo redondeo).
    pesos = np.array([0.114, 0.587, 0.299])
    gris_mano = np.clip(np.round(bgr.astype(np.float64) @ pesos), 0, 255).astype(np.uint8)
    gris_cv2 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dif_max = int(np.abs(gris_mano.astype(int) - gris_cv2.astype(int)).max())
    check("gris a mano == cv2 (+-1)", dif_max <= 1, f"dif max {dif_max}")

    # 3. La imagen esta expuesta con sentido (ni negra ni quemada):
    #    una foto real de interior tiene la media lejos de ambos extremos.
    media = float(gris_mano.mean())
    check("exposicion razonable (media en 20..235)", 20.0 <= media <= 235.0,
          f"media {media:.1f}")

    # 4. El negativo invierte de verdad: negativo + original == 255 exacto.
    negativo = 255 - gris_mano
    check("negativo + original == 255", bool(np.all(negativo + gris_mano == 255)))

    print()
    if fallos:
        print(f"NIVEL 00: {len(fallos)} fallo(s): {', '.join(fallos)}")
        return 1
    print("NIVEL 00: VERIFICADO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
