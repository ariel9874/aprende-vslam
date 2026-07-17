#!/usr/bin/env python3
"""Examen del nivel 15: SLAM RGB-D en METROS, sobre fr1_desk.

fr1_desk es la secuencia handheld que DERROTÓ al nivel 14 (562 de 613 frames
perdidos, mapa inservible). Con profundidad, el mismo esqueleto la convierte
en un mapa MÉTRICO a ~2 cm. Dos actos, con los números medidos al construir
el nivel:

  1. El sistema completo: init instantánea (el primer frame YA es mapa),
     trackea la mayor parte de la secuencia (medido: 203 perdidos de 596 —
     los episodios de blur, sin relocalización, siguen doliendo), y el ATE
     final de keyframes con alineación RÍGIDA queda en pocos cm con escala
     de similitud ≈ 1 (medido: 2.3 cm, escala 1.012 — metros de verdad).
  2. La ablación (bf = 0): sin el residuo de profundidad el mapa pre-GBA se
     degrada de forma medible (medido: 2.6 -> 5.8 cm, 2.2x peor).

Necesita el dataset (python descarga_datos.py, ~330 MB) o `--root <ruta>`.
Dura ~5 min (dos pasadas de 613 frames + dos BA globales).

Uso:
    python verificacion.py [--root <ruta_fr1_desk>]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

from dataset import leer_trayectoria_tum

spec = importlib.util.spec_from_file_location("n15", AQUI / "15_rgbd.py")
n15 = importlib.util.module_from_spec(spec)
sys.modules["n15"] = n15
spec.loader.exec_module(n15)

fallos = []


def check(nombre: str, ok: bool, detalle: str = "") -> None:
    estado = "OK " if ok else "FALLO"
    print(f"  [{estado}] {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="secuencia fr1_desk ya descargada")
    args = parser.parse_args()

    root = Path(args.root) if args.root else n15.DATASET_DEFAULT
    if not (root / "depth.txt").is_file():
        raise SystemExit(f"No hay dataset RGB-D en {root}.\n"
                         "Corre `python descarga_datos.py` (~330 MB) o pasa "
                         "--root <ruta a rgbd_dataset_freiburg1_desk>.")

    gt_ts, gt_pos = leer_trayectoria_tum(root / "groundtruth.txt")
    print(f"Verificando sobre {root.name} (la secuencia que derroto "
          "al nivel 14)\n")

    # ── Acto 1: el sistema completo ──────────────────────────────────────────
    print("[1/2] Sistema completo (residuo de profundidad ON)...")
    ts, pos, estados, s = n15.correr(root)

    check("la init es INSTANTANEA (el primer frame procesado ya es mapa)",
          estados and estados[0] == "INIT-OK",
          f"estado del frame 0: {estados[0] if estados else '?'}")
    check("el mapa nace en METROS (mediana de profundidad 0.3-8 m)",
          0.3 < float(np.median([X[2] for X in s.mapa.puntos.values()])) < 8.0,
          "retro-proyeccion directa del sensor")
    check("se insertaron >=20 keyframes", len(s.kf_poses) >= 20,
          f"{len(s.kf_poses)} (medido: 30)")
    check("sobrevive lo que el nivel 14 no pudo (perdidos < 350 de ~600)",
          s.n_perdidos < 350,
          f"{s.n_perdidos} perdidos (nivel 14: 562; el resto del hueco es "
          "la reloc, que vive en el padre)")

    m0 = n15.evaluar_kfs(s, ts, gt_ts, gt_pos)
    print(f"\n      BA global offline ({s.GBA_ITERS} iteraciones)...")
    s.global_bundle_adjustment()
    m1 = n15.evaluar_kfs(s, ts, gt_ts, gt_pos)

    print(f"\n  ATE final-KF RIGIDO: {100*m0['rmse']:5.1f} -> "
          f"{100*m1['rmse']:.1f} cm tras GBA | escala {m1['scale_sim']:.3f}\n")

    check("ATE final de keyframes RIGIDO < 5 cm", m1["rmse"] < 0.05,
          f"{100*m1['rmse']:.1f} cm (medido: 2.3; el padre: 2.8)")
    check("la escala de similitud es ~1 (|escala - 1| < 0.05)",
          abs(m1["scale_sim"] - 1.0) < 0.05,
          f"{m1['scale_sim']:.3f} (medido: 1.012; el padre: 1.005). Este "
          "es EL numero del nivel: metros de verdad, no gauge")

    # ── Acto 2: la ablacion del residuo de profundidad ───────────────────────
    # bf = 0: la init sigue siendo metrica, pero el BA ya no MIDE metros en
    # cada observacion — solo re-teje reproyecciones 2D. La estructura del
    # mapa pierde su ancla y el error pre-GBA crece de forma medible. (En el
    # padre, con reloc, la ablacion ademas cruzaba peor el episodio de blur:
    # 12.8 cm vs 2.8. Aqui ambos mueren en el mismo blur — el residuo se nota
    # en la CALIDAD del mapa, no en la supervivencia del frontend.)
    print("\n[2/2] Ablacion: residuo de profundidad OFF (bf = 0)...")
    ts2, pos2, estados2, s2 = n15.correr(root, usar_residuo=False)
    m2 = n15.evaluar_kfs(s2, ts2, gt_ts, gt_pos)
    print(f"  ATE final-KF RIGIDO pre-GBA: {100*m2['rmse']:.1f} cm "
          f"(con residuo: {100*m0['rmse']:.1f}) | escala {m2['scale_sim']:.3f}")
    check("sin residuo, el mapa pre-GBA es medible-mente peor (>=1.3x)",
          m2["rmse"] >= 1.3 * m0["rmse"],
          f"{100*m2['rmse']:.1f} vs {100*m0['rmse']:.1f} cm "
          f"({m2['rmse']/max(m0['rmse'],1e-9):.1f}x; medido: 2.3x)")

    print()
    if fallos:
        print(f"NIVEL 15: {len(fallos)} fallo(s): {', '.join(fallos)}")
        return 1
    print("NIVEL 15: VERIFICADO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
