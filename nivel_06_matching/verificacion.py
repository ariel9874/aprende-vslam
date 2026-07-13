#!/usr/bin/env python3
"""Examen del nivel 06: el ratio test hace su trabajo, con numeros.

Umbrales calibrados sobre fr1_xyz gap 30 (medidos: 2000 matches crudos,
418 tras ratio 0.75, sospechosos 30% -> 8%); margenes generosos por si
corres con otro par. Si todo pasa: NIVEL 06: VERIFICADO.

Uso:
    python verificacion.py [--root <secuencia_TUM>] [--gap 30]
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import cv2

AQUI = Path(__file__).resolve().parent
DATASET_DEFAULT = AQUI / "data" / "rgbd_dataset_freiburg1_xyz"

spec = importlib.util.spec_from_file_location("n6", AQUI / "06_matching.py")
n6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n6)

fallos = []


def check(nombre: str, ok: bool, detalle: str = "") -> None:
    estado = "OK " if ok else "FALLO"
    print(f"  [{estado}] {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DATASET_DEFAULT))
    parser.add_argument("--gap", type=int, default=30)
    args = parser.parse_args()

    gray_a, gray_b, na, nb = n6.cargar_par(Path(args.root), args.gap)
    print(f"Verificando con el par {na} -> {nb}\n")

    orb = cv2.ORB_create(nfeatures=2000)
    kps_a, desc_a = orb.detectAndCompute(gray_a, None)
    kps_b, desc_b = orb.detectAndCompute(gray_b, None)

    # 1. Hay material y el ratio filtra de verdad.
    todos, buenos = n6.emparejar(desc_a, desc_b, 0.75)
    check("matches crudos >= 1000", len(todos) >= 1000, str(len(todos)))
    check("ratio 0.75 deja 150..900 matches", 150 <= len(buenos) <= 900,
          f"{len(buenos)} (medido en fr1: 418)")
    check("el ratio elimina la mayoria", len(buenos) < 0.5 * len(todos),
          f"sobreviven {100*len(buenos)/len(todos):.0f}%")

    # 2. La supervivencia crece con el umbral (sanity de la implementacion).
    n_060 = len(n6.emparejar(desc_a, desc_b, 0.60)[1])
    n_090 = len(n6.emparejar(desc_a, desc_b, 0.90)[1])
    check("supervivencia monotona (0.60 < 0.75 < 0.90)",
          n_060 < len(buenos) < n_090, f"{n_060} < {len(buenos)} < {n_090}")

    # 3. EL numero de la leccion: el ratio limpia matches incoherentes.
    f_antes = n6.fraccion_sospechosos(n6.desplazamientos(todos, kps_a, kps_b))
    f_despues = n6.fraccion_sospechosos(n6.desplazamientos(buenos, kps_a, kps_b))
    check("sospechosos despues < antes", f_despues < f_antes,
          f"{100*f_antes:.0f}% -> {100*f_despues:.0f}% (medido: 30 -> 8)")

    # 4. Pasado el punto dulce, la basura vuelve (el brazo derecho de la U).
    lax = n6.emparejar(desc_a, desc_b, 0.95)[1]
    f_lax = n6.fraccion_sospechosos(n6.desplazamientos(lax, kps_a, kps_b))
    check("con ratio 0.95 hay mas basura que con 0.75", f_lax > f_despues,
          f"{100*f_lax:.0f}% vs {100*f_despues:.0f}% (medido: 24 vs 8)")

    print()
    if fallos:
        print(f"NIVEL 06: {len(fallos)} fallo(s): {', '.join(fallos)}")
        return 1
    print("NIVEL 06: VERIFICADO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
