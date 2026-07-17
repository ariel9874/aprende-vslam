#!/usr/bin/env python3
"""
Nivel 14 — Datos reales: TUM RGB-D
==================================

El SLAM del nivel 13, contra imágenes de una cámara DE VERDAD. Lo que el
mundo real añade y este script resuelve:

  - DISTORSIÓN de lente: se pre-rectifica cada imagen (nivel 04) con la
    calibración publicada de la cámara Freiburg.
  - TIMESTAMPS reales: el ground truth (mocap, ~100 Hz) se asocia a cada
    frame RGB (~30 Hz) por timestamp más cercano.
  - MATCHING ambiguo: contra un mapa real el matching por descriptor se
    degrada; el matching GUIADO por reproyección es la cura (ver slam.py).

Y el refinamiento final: UN bundle adjustment GLOBAL offline, cuyo efecto se
mide donde debe medirse — la trayectoria FINAL de keyframes.

Uso:
    python descarga_datos.py                          # una vez (~2.1 GB)
    python 14_datos_reales.py                         # fr2_xyz completa
    python 14_datos_reales.py --max-frames 1000       # mas rapido
    python 14_datos_reales.py --root <otra_secuencia> # p. ej. fr1_desk
    python 14_datos_reales.py --sin-guiado            # la ablacion
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
DATASET_DEFAULT = AQUI / "data" / "rgbd_dataset_freiburg2_xyz"


def correr(root: Path, max_frames: int = 0, usar_ba: bool = True,
           usar_bucle: bool = True, usar_guiado: bool = True,
           verbose: bool = False):
    """Corre el SLAM sobre una secuencia TUM.

    Devuelve (timestamps, posiciones_online, estados, tracker). Fija la
    semilla del RANSAC de OpenCV para que dos corridas (p. ej. la ablación
    guiado ON/OFF) sean comparables y el examen, reproducible.
    """
    cv2.setRNGSeed(7)
    K, dist = camara_tum(root.name)
    loader = SecuenciaTUM(root)

    # Pre-rectificación (nivel 04): la geometría del curso asume el modelo
    # pinhole IDEAL, así que la distorsión se quita ANTES, una vez por frame.
    # (La alternativa por-keypoint, undistortPoints, es más barata; el padre
    # la usa. Rectificar la imagen entera es más simple de leer.)
    maps = None
    if np.any(dist != 0.0):
        maps = cv2.initUndistortRectifyMap(K, dist, None, K, (640, 480),
                                           cv2.CV_32FC1)

    s = SLAM(K, usar_ba=usar_ba, usar_bucle=usar_bucle,
             usar_guiado=usar_guiado)
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
        if verbose and (i % 200 == 0 or info["loop"]):
            extra = f"  <<< BUCLE {info['loop']}" if info["loop"] else ""
            print(f"  frame {i:4d} | {info['estado']:7s} | inliers "
                  f"{info['n_inliers']:3d} | mapa {info['n_mapa']:6d} | "
                  f"KFs {len(s.kf_poses):3d}{extra}")
    if verbose:
        dt = time.perf_counter() - t0
        print(f"  ({len(ts_list)} frames en {dt:.0f} s = {len(ts_list)/dt:.1f} fps)")
    return np.array(ts_list), np.array(pos), estados, s


def evaluar_online(ts, pos, estados, gt_ts, gt_pos) -> dict:
    """ATE de las poses EMITIDAS, desde que el mapa existe (tras INIT)."""
    start = next((i for i, e in enumerate(estados)
                  if e in ("INIT-OK", "TRACK")), 0)
    assoc = asociar_por_timestamp(ts[start:], gt_ts, max_dt=0.02)
    ok = assoc >= 0
    if ok.sum() < 3:
        return {"rmse": float("nan"), "n": 0}
    m = ate(pos[start:][ok], gt_pos[assoc[ok]])
    m["n"] = int(ok.sum())
    return m


def evaluar_kfs(tracker, ts, gt_ts, gt_pos) -> dict:
    """ATE de la trayectoria FINAL de keyframes (la métrica honesta).

    Cada keyframe sabe en qué FRAME nació; ese frame tiene un TIMESTAMP real;
    ese timestamp se asocia al ground truth de la mocap. (En el nivel 13 el
    frame ERA el índice del GT; con timestamps reales hay un salto más.)
    """
    frames, pos = tracker.trayectoria_kfs()
    kf_ts = ts[frames]
    assoc = asociar_por_timestamp(kf_ts, gt_ts, max_dt=0.05)
    ok = assoc >= 0
    if ok.sum() < 3:
        return {"rmse": float("nan"), "n": 0}
    m = ate(pos[ok], gt_pos[assoc[ok]])
    m["n"] = int(ok.sum())
    return m


def main() -> int:
    parser = argparse.ArgumentParser(description="Nivel 14: datos reales (TUM)")
    parser.add_argument("--root", default=None,
                        help="carpeta de la secuencia TUM (default: fr2_xyz)")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--sin-guiado", action="store_true",
                        help="ablacion: matching global por descriptor")
    parser.add_argument("--sin-gba", action="store_true",
                        help="sin el BA global offline")
    parser.add_argument("--sin-bucle", action="store_true")
    args = parser.parse_args()

    root = Path(args.root) if args.root else DATASET_DEFAULT
    if not (root / "rgb.txt").is_file():
        raise SystemExit(f"No hay dataset en {root}.\n"
                         "Corre `python descarga_datos.py` o pasa --root.")

    print(f"Secuencia: {root.name}"
          + (f" (primeros {args.max_frames} frames)" if args.max_frames else "")
          + f" | guiado: {'OFF' if args.sin_guiado else 'ON'}\n")

    ts, pos, estados, s = correr(root, max_frames=args.max_frames,
                                 usar_bucle=not args.sin_bucle,
                                 usar_guiado=not args.sin_guiado,
                                 verbose=True)

    gt_ts, gt_pos = leer_trayectoria_tum(root / "groundtruth.txt")
    print(f"\nMapa: {len(s.mapa)} puntos | {len(s.kf_poses)} keyframes | "
          f"{len(s.eventos_bucle)} bucles | {s.n_perdidos} frames perdidos")
    if s.inliers_hist:
        print(f"Inliers de PnP: mediana {int(np.median(s.inliers_hist))} "
              f"(minimo {min(s.inliers_hist)})")

    m_on = evaluar_online(ts, pos, estados, gt_ts, gt_pos)
    m_kf0 = evaluar_kfs(s, ts, gt_ts, gt_pos)
    print(f"\nATE online (poses emitidas):     {100*m_on['rmse']:6.1f} cm "
          f"({m_on['n']} frames)")
    print(f"ATE final-KF ANTES del BA global:{100*m_kf0['rmse']:6.1f} cm "
          f"({m_kf0['n']} keyframes)")

    if not args.sin_gba:
        print(f"\nBA GLOBAL offline ({s.GBA_ITERS} iteraciones; "
              "leccion 27: con 10 no converge)...")
        t0 = time.perf_counter()
        s.global_bundle_adjustment()
        m_kf = evaluar_kfs(s, ts, gt_ts, gt_pos)
        print(f"ATE final-KF TRAS el BA global:  {100*m_kf['rmse']:6.1f} cm "
              f"({time.perf_counter()-t0:.0f} s)")
        print("\nLa historia completa, en una linea:")
        print(f"  online {100*m_on['rmse']:.1f} -> final-KF "
              f"{100*m_kf0['rmse']:.1f} -> + BA global {100*m_kf['rmse']:.1f} cm")
    else:
        m_kf = m_kf0

    # ── grafico: planta + error(t) de la trayectoria de keyframes ───────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from evaluacion import umeyama_alignment

        frames, kf_pos = s.trayectoria_kfs()
        assoc = asociar_por_timestamp(ts[frames], gt_ts, max_dt=0.05)
        ok = assoc >= 0
        sc, R, t = umeyama_alignment(kf_pos[ok], gt_pos[assoc[ok]])
        al = (sc * (R @ kf_pos[ok].T)).T + t
        gtm = gt_pos[assoc[ok]]
        err = np.linalg.norm(al - gtm, axis=1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        ax1.plot(gtm[:, 0], gtm[:, 1], "k--", lw=1.2, label="mocap (verdad)")
        ax1.plot(al[:, 0], al[:, 1], lw=1.2, label="keyframes (alineados)")
        ax1.set_xlabel("x [m]"), ax1.set_ylabel("y [m]")
        ax1.set_title(f"{root.name} en planta"), ax1.axis("equal")
        ax1.legend(), ax1.grid(alpha=0.3)
        ax2.plot(frames[ok], 100 * err, color="tab:red", lw=1.0)
        for _, kf_n in s.eventos_bucle:
            ax2.axvline(s.kf_frame.get(kf_n, 0), color="tab:blue",
                        ls=":", alpha=0.6)
        ax2.set_xlabel("frame"), ax2.set_ylabel("error [cm]")
        ax2.set_title("error por keyframe (azul: bucles)"), ax2.grid(alpha=0.3)
        salida = AQUI / "salida"
        salida.mkdir(exist_ok=True)
        fig.savefig(salida / f"{root.name}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\nGrafica: {salida / (root.name + '.png')}")
    except ImportError:
        pass

    print("Ahora corre `python verificacion.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
