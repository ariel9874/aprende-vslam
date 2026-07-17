#!/usr/bin/env python3
"""Examen del nivel 17: cuándo gana el deep — MEDIDO, no creído.

Tres actos sobre fr1_desk (la secuencia del motion blur), con los números
medidos al construir el nivel:

  1. DONDE GANA: matching a través del blur, par a par. Un frame nítido
     contra uno borroso de la misma ráfaga: ORB colapsa (16 matches / 8
     E-inliers en el par más duro), SuperPoint+LightGlue no (210 / 109 —
     13x más inliers). El descriptor con contexto sobrevive al blur que
     borra los gradientes locales de ORB.
  2. LA LECCIÓN INCÓMODA (la 29 del padre): en ESTE tracker, la ráfaga dura
     no la cruza NINGÚN frontend (ORB ~546 perdidos, SuperPoint ~557 —
     empatados en la muerte). El episodio es ESTRUCTURAL, no de umbrales.
     La cura real ya la mediste: el residuo de profundidad del nivel 15
     cruza esta MISMA secuencia a 2.3 cm. El sensor, no el frontend.
  3. EL COSTO: SuperPoint+LightGlue corre a ~9 fps en GPU contra ~26 de
     ORB en CPU. El deep no es gratis.

Necesita fr1_desk (python descarga_datos.py, ~330 MB) y el frontend deep
instalado (ver requirements.txt). Con GPU dura ~4 min; en CPU, MUCHO más.

Uso:
    python verificacion.py [--root <ruta_fr1_desk>]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

from dataset import SecuenciaTUM, camara_tum, leer_trayectoria_tum

spec = importlib.util.spec_from_file_location("n17", AQUI / "17_aprendidas.py")
n17 = importlib.util.module_from_spec(spec)
sys.modules["n17"] = n17
spec.loader.exec_module(n17)

fallos = []


def check(nombre: str, ok: bool, detalle: str = "") -> None:
    estado = "OK " if ok else "FALLO"
    print(f"  [{estado}] {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def inliers_epipolares(K, kA, dA, kB, dB, matcher, shape):
    """matches y E-inliers de un par de imágenes (el árbitro: RANSAC)."""
    ms = matcher.match(dA, dB, kA, kB, shape)
    if len(ms) < 8:
        return len(ms), 0
    p0 = np.float64([kA[m.queryIdx].pt for m in ms])
    p1 = np.float64([kB[m.trainIdx].pt for m in ms])
    E, mask = cv2.findEssentialMat(p0, p1, K, method=cv2.RANSAC,
                                   prob=0.999, threshold=1.0)
    return len(ms), (0 if E is None else int(mask.sum()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="secuencia fr1_desk ya descargada")
    args = parser.parse_args()

    root = Path(args.root) if args.root else n17.DATASET_DEFAULT
    if not (root / "rgb.txt").is_file():
        raise SystemExit(f"No hay dataset en {root}.\n"
                         "Corre `python descarga_datos.py` (~330 MB) o pasa "
                         "--root <ruta a rgbd_dataset_freiburg1_desk>.")
    try:
        from features import (ExtractorORB, ExtractorSuperPoint,
                              MatcherLightGlue, MatcherRatio)
        sp = ExtractorSuperPoint()
    except ImportError as exc:
        raise SystemExit(str(exc))

    print(f"Verificando sobre {root.name} | frontend deep en: {sp.device}\n")
    cv2.setRNGSeed(7)
    K, dist = camara_tum(root.name)
    maps = cv2.initUndistortRectifyMap(K, dist, None, K, (640, 480),
                                       cv2.CV_32FC1)

    # ── Acto 1: el experimento del blur (par a par) ──────────────────────────
    print("[1/3] Matching a traves del blur (frame 40 nitido vs la rafaga)...")
    frames = {}
    for i, (ts, g) in enumerate(SecuenciaTUM(root)):
        if i in (40, 52, 56, 60):
            frames[i] = cv2.remap(g, maps[0], maps[1], cv2.INTER_LINEAR)
        if i > 60:
            break
    orb, ratio = ExtractorORB(), MatcherRatio()
    lg = MatcherLightGlue(device=sp.device)
    kA_o, dA_o = orb.detectar(frames[40])
    kA_s, dA_s = sp.detectar(frames[40])
    peor_factor = np.inf
    for b in (52, 56, 60):
        kB_o, dB_o = orb.detectar(frames[b])
        kB_s, dB_s = sp.detectar(frames[b])
        _, io = inliers_epipolares(K, kA_o, dA_o, kB_o, dB_o, ratio,
                                   frames[40].shape)
        _, isp = inliers_epipolares(K, kA_s, dA_s, kB_s, dB_s, lg,
                                    frames[40].shape)
        factor = isp / max(io, 1)
        peor_factor = min(peor_factor, factor)
        print(f"  par (40,{b}): ORB {io:4d} E-inliers | "
              f"SuperPoint+LightGlue {isp:4d}  ({factor:.1f}x)")
    check("el deep atraviesa el blur >= 3x mejor (en TODOS los pares)",
          peor_factor >= 3.0,
          f"peor factor {peor_factor:.1f}x (medido: 6.5x, 6.9x, 13.6x)")

    # ── Acto 2: la leccion incomoda (el sistema completo) ────────────────────
    print("\n[2/3] El sistema completo con cada frontend (la leccion 29)...")
    gt_ts, gt_pos = leer_trayectoria_tum(root / "groundtruth.txt")
    _, _, _, s_orb, fps_orb = n17.correr(root, frontend="orb",
                                         usar_gba=False)
    _, _, _, s_sp, fps_sp = n17.correr(root, frontend="superpoint",
                                       usar_gba=False)
    print(f"  ORB:        {s_orb.n_perdidos} perdidos | {fps_orb:.1f} fps")
    print(f"  SuperPoint: {s_sp.n_perdidos} perdidos | {fps_sp:.1f} fps")
    check("la rafaga dura NO la cruza ningun frontend en este tracker",
          s_orb.n_perdidos > 400 and s_sp.n_perdidos > 400,
          f"ORB {s_orb.n_perdidos}, SP {s_sp.n_perdidos} (medido: ~546/~557; "
          "el episodio es ESTRUCTURAL — el residuo de profundidad del nivel "
          "15 SI lo cruza: 2.3 cm)")

    # ── Acto 3: el costo ─────────────────────────────────────────────────────
    print("\n[3/3] El costo computacional...")
    check("el deep NO es gratis (fps ORB > fps deep)",
          fps_orb > fps_sp,
          f"ORB {fps_orb:.1f} vs SP {fps_sp:.1f} fps (medido: ~26 vs ~9 "
          "en GPU; en CPU la brecha es mucho mayor)")

    print()
    if fallos:
        print(f"NIVEL 17: {len(fallos)} fallo(s): {', '.join(fallos)}")
        return 1
    print("NIVEL 17: VERIFICADO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
