#!/usr/bin/env python3
"""
Nivel 17 — Features aprendidas: cuándo el deep gana (y cómo medirlo)
====================================================================

El sistema del nivel 14, con el frontend intercambiable. La pregunta se
responde con UNA secuencia: fr1_desk — la cámara en mano con motion blur que
ORB no puede cruzar (562 de 613 frames perdidos en el nivel 14).

    python 17_aprendidas.py                    # la comparativa: ORB vs deep
    python 17_aprendidas.py --frontend superpoint
    python 17_aprendidas.py --matcher ratio    # superpoint SIN lightglue

Con GPU NVIDIA el frontend deep corre a unos pocos fps; en CPU funciona pero
tarda MUCHO mas (documentado, no prohibido: paciencia).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from dataset import (SecuenciaTUM, asociar_por_timestamp, camara_tum,
                     leer_trayectoria_tum)
from evaluacion import ate
from slam import SLAM

AQUI = Path(__file__).resolve().parent
DATASET_DEFAULT = AQUI / "data" / "rgbd_dataset_freiburg1_desk"


def construir_frontend(nombre: str, matcher: str):
    """(extractor, matcher_pares) — los enchufes de features.py."""
    from features import (ExtractorORB, ExtractorSuperPoint, MatcherLightGlue,
                          MatcherRatio)
    if nombre == "orb":
        return ExtractorORB(), MatcherRatio()
    extractor = ExtractorSuperPoint()
    if matcher == "lightglue":
        return extractor, MatcherLightGlue(features="superpoint",
                                           device=extractor.device)
    return extractor, MatcherRatio()


def correr(root: Path, frontend: str = "orb", matcher: str = "lightglue",
           max_frames: int = 0, usar_gba: bool = True, verbose: bool = False):
    """Corre el SLAM monocular del nivel 14 con el frontend elegido."""
    cv2.setRNGSeed(7)
    K, dist = camara_tum(root.name)
    loader = SecuenciaTUM(root)
    maps = None
    if np.any(dist != 0.0):
        maps = cv2.initUndistortRectifyMap(K, dist, None, K, (640, 480),
                                           cv2.CV_32FC1)
    extractor, matcher_pares = construir_frontend(frontend, matcher)
    if verbose and frontend == "superpoint":
        print(f"  superpoint en: {extractor.device}"
              + (" (CPU: tardara MUCHO mas)" if extractor.device == "cpu"
                 else ""))
    s = SLAM(K, extractor=extractor, matcher_pares=matcher_pares)

    ts_list, pos, estados = [], [], []
    t0 = time.perf_counter()
    for i, (ts, gray) in enumerate(loader):
        if max_frames and i >= max_frames:
            break
        if maps is not None:
            gray = cv2.remap(gray, maps[0], maps[1], cv2.INTER_LINEAR)
        T, info = s.procesar(gray)
        ts_list.append(ts)
        pos.append(T[:3, 3].copy())
        estados.append(info["estado"])
        if verbose and (i % 100 == 0 or info["loop"]):
            extra = f"  <<< BUCLE {info['loop']}" if info["loop"] else ""
            print(f"  frame {i:4d} | {info['estado']:7s} | inliers "
                  f"{info['n_inliers']:3d} | mapa {info['n_mapa']:6d} | "
                  f"KFs {len(s.kf_poses):3d}{extra}")
    dt = time.perf_counter() - t0
    fps = len(ts_list) / dt if dt > 0 else 0.0
    if verbose:
        print(f"  ({len(ts_list)} frames en {dt:.0f} s = {fps:.1f} fps)")
    if usar_gba and len(s.kf_poses) >= 3:
        s.global_bundle_adjustment()
    return np.array(ts_list), np.array(pos), estados, s, fps


def evaluar_kfs(tracker, ts, gt_ts, gt_pos) -> dict:
    """ATE de SIMILITUD de la trayectoria final de KFs (monocular)."""
    frames, pos = tracker.trayectoria_kfs()
    if len(frames) < 3:
        return {"rmse": float("nan"), "n": 0}
    assoc = asociar_por_timestamp(ts[frames], gt_ts, max_dt=0.05)
    ok = assoc >= 0
    if ok.sum() < 3:
        return {"rmse": float("nan"), "n": 0}
    m = ate(pos[ok], gt_pos[assoc[ok]])
    m["n"] = int(ok.sum())
    return m


def resumen(nombre: str, s, m_kf, fps) -> dict:
    return {"frontend": nombre, "perdidos": s.n_perdidos,
            "kfs": len(s.kf_poses), "bucles": len(s.eventos_bucle),
            "relocs": len(s.eventos_reloc),
            "ate_kf": m_kf["rmse"], "fps": fps}


def main() -> int:
    parser = argparse.ArgumentParser(description="Nivel 17: deep vs clasico")
    parser.add_argument("--root", default=None)
    parser.add_argument("--frontend", default="ambos",
                        choices=["ambos", "orb", "superpoint"])
    parser.add_argument("--matcher", default="lightglue",
                        choices=["lightglue", "ratio"],
                        help="matcher de pares para superpoint")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--sin-gba", action="store_true")
    args = parser.parse_args()

    root = Path(args.root) if args.root else DATASET_DEFAULT
    if not (root / "rgb.txt").is_file():
        raise SystemExit(f"No hay dataset en {root}.\n"
                         "Corre `python descarga_datos.py` o pasa --root.")

    gt_ts, gt_pos = leer_trayectoria_tum(root / "groundtruth.txt")
    frontends = (["orb", "superpoint"] if args.frontend == "ambos"
                 else [args.frontend])
    filas = []
    for f in frontends:
        print(f"\n=== {f.upper()}"
              + (f" + {args.matcher}" if f == "superpoint" else " + ratio")
              + f" sobre {root.name} ===")
        ts, pos, estados, s, fps = correr(root, frontend=f,
                                          matcher=args.matcher,
                                          max_frames=args.max_frames,
                                          usar_gba=not args.sin_gba,
                                          verbose=True)
        m_kf = evaluar_kfs(s, ts, gt_ts, gt_pos)
        print(f"  perdidos {s.n_perdidos} | KFs {len(s.kf_poses)} | "
              f"bucles {len(s.eventos_bucle)} | relocs "
              f"{len(s.eventos_reloc)} | "
              f"ATE final-KF {100*m_kf['rmse']:.1f} cm (similitud)")
        filas.append(resumen(f, s, m_kf, fps))

    if len(filas) == 2:
        print("\n" + "=" * 64)
        print("LA TABLA DEL NIVEL (fr1_desk, la secuencia del motion blur):")
        print(f"  {'frontend':12s} {'perdidos':>9s} {'relocs':>7s} "
              f"{'KFs':>5s} {'ATE-KF':>8s} {'fps':>6s}")
        for r in filas:
            print(f"  {r['frontend']:12s} {r['perdidos']:9d} "
                  f"{r['relocs']:7d} {r['kfs']:5d} "
                  f"{100*r['ate_kf']:6.1f}cm {r['fps']:6.1f}")
        print("=" * 64)
        print("La leccion incomoda (la 29 del padre): en este tracker, la")
        print("rafaga dura no la cruza NINGUN frontend — el episodio es")
        print("estructural. Donde el deep SI gana se mide par a par (el")
        print("examen, acto 1: 13x mas inliers a traves del blur). Y la")
        print("cura real de fr1_desk ya la mediste en el nivel 15: el")
        print("residuo de profundidad la cruza entera a 2.3 cm.")

    print("\nAhora corre `python verificacion.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
