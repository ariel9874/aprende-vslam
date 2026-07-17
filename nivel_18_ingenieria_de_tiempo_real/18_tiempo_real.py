#!/usr/bin/env python3
"""
Nivel 18 — Ingeniería de tiempo real: perfilar, sustituir, verificar
====================================================================

El MÉTODO de optimización de sistemas, en tres pasos que este script ejecuta
sobre fr2_xyz:

  1. PERFILAR: ¿dónde se va el tiempo? (la tabla por etapas). La intuición
     se REFUTA con datos: el repo padre apostaba por cv2 y el perfil dijo
     BA 57% + matching guiado 37%, cv2 8%.
  2. SUSTITUIR solo el punto caliente por una GEMELA: el BA vectorizado
     (ba_rapido.py) y el BoW sub-lineal (bow.py).
  3. VERIFICAR la equivalencia (verificacion.py): la gemela no "parece"
     funcionar — da lo mismo, a tolerancia numérica.

Uso:
    python descarga_datos.py                  # fr2_xyz (~2.1 GB, una vez)
    python 18_tiempo_real.py                  # perfil + escalera (1000 frames)
    python 18_tiempo_real.py --max-frames 600 # mas rapido
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from ba_rapido import bundle_adjustment_rapido
from bow import BolsaDePalabras
from dataset import (SecuenciaTUM, asociar_por_timestamp, camara_tum,
                     leer_trayectoria_tum)
from evaluacion import ate
from slam import SLAM

AQUI = Path(__file__).resolve().parent
DATASET_DEFAULT = AQUI / "data" / "rgbd_dataset_freiburg2_xyz"


def correr(root: Path, max_frames: int = 0, ba_fn=None, bow=None,
           verbose: bool = False):
    """Una corrida del tracker del nivel 14 con las gemelas enchufadas (o no).

    Devuelve (ts, estados, tracker, fps_tracking, t_gba).
    """
    cv2.setRNGSeed(7)
    K, dist = camara_tum(root.name)
    loader = SecuenciaTUM(root)
    maps = None
    if np.any(dist != 0.0):
        maps = cv2.initUndistortRectifyMap(K, dist, None, K, (640, 480),
                                           cv2.CV_32FC1)
    s = SLAM(K, ba_fn=ba_fn, bow=bow)
    ts_list = []
    t0 = time.perf_counter()
    for i, (ts, gray) in enumerate(loader):
        if max_frames and i >= max_frames:
            break
        if maps is not None:
            gray = cv2.remap(gray, maps[0], maps[1], cv2.INTER_LINEAR)
        s.procesar(gray)
        ts_list.append(ts)
    dt = time.perf_counter() - t0
    fps = len(ts_list) / dt if dt > 0 else 0.0
    t0 = time.perf_counter()
    s.global_bundle_adjustment()
    t_gba = time.perf_counter() - t0
    if verbose:
        print(f"  {len(ts_list)} frames | tracking {dt:.0f} s ({fps:.1f} fps)"
              f" | BA global {t_gba:.0f} s")
    return np.array(ts_list), s, fps, t_gba, dt


def ate_kfs(s, ts, gt_ts, gt_pos) -> float:
    frames, pos = s.trayectoria_kfs()
    assoc = asociar_por_timestamp(ts[frames], gt_ts, max_dt=0.05)
    ok = assoc >= 0
    if ok.sum() < 3:
        return float("nan")
    return ate(pos[ok], gt_pos[assoc[ok]])["rmse"]


def tabla_perfil(s, total: float) -> None:
    """La tabla del nivel: ¿DÓNDE se va el tiempo del tracking?"""
    filas = sorted(s.perfil.items(), key=lambda kv: -kv[1])
    resto = total - sum(s.perfil.values())
    print(f"\n  {'etapa':26s} {'segundos':>9s} {'%':>6s}")
    for etapa, seg in filas:
        print(f"  {etapa:26s} {seg:9.1f} {100*seg/total:5.1f}%")
    print(f"  {'(resto: E/S, numpy, ...)':26s} {resto:9.1f} "
          f"{100*resto/total:5.1f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description="Nivel 18: el metodo")
    parser.add_argument("--root", default=None)
    parser.add_argument("--max-frames", type=int, default=1000)
    args = parser.parse_args()

    root = Path(args.root) if args.root else DATASET_DEFAULT
    if not (root / "rgb.txt").is_file():
        raise SystemExit(f"No hay dataset en {root}.\n"
                         "Corre `python descarga_datos.py` o pasa --root.")
    gt_ts, gt_pos = leer_trayectoria_tum(root / "groundtruth.txt")
    n = args.max_frames

    # ── Paso 1: PERFILAR el sistema de referencia ────────────────────────────
    print(f"[1/2] PERFIL del sistema del nivel 14 ({n} frames de {root.name})")
    print("      Antes de mirar la tabla: ¿donde apostarias TU que se va")
    print("      el tiempo? Escribelo. La intuicion se refuta con datos.")
    ts, s0, fps0, gba0, dt0 = correr(root, max_frames=n, verbose=True)
    tabla_perfil(s0, dt0)
    ate0 = ate_kfs(s0, ts, gt_ts, gt_pos)

    # ── Paso 2: la ESCALERA (sustituir SOLO el punto caliente) ───────────────
    print("\n[2/2] LA ESCALERA: cada peldano enchufa UNA gemela verificada")
    print("\n  peldano 2: BA vectorizado (ba_rapido.py)...")
    ts, s1, fps1, gba1, _ = correr(root, max_frames=n,
                                   ba_fn=bundle_adjustment_rapido,
                                   verbose=True)
    ate1 = ate_kfs(s1, ts, gt_ts, gt_pos)

    print("\n  peldano 3: + BoW para el reconocimiento de lugar (bow.py)...")
    ts, s2, fps2, gba2, _ = correr(root, max_frames=n,
                                   ba_fn=bundle_adjustment_rapido,
                                   bow=BolsaDePalabras(),
                                   verbose=True)
    ate2 = ate_kfs(s2, ts, gt_ts, gt_pos)

    print("\n" + "=" * 70)
    print("LA ESCALERA (la del padre fue 4.3 -> 9.5 -> 18.7 -> 25.7 -> 46.7):")
    print(f"  {'configuracion':28s} {'fps':>6s} {'BA global':>10s} "
          f"{'ATE-KF':>8s} {'bucles':>7s}")
    for nombre, fps, gba, a, sx in [
            ("referencia (nivel 14)", fps0, gba0, ate0, s0),
            ("+ BA vectorizado", fps1, gba1, ate1, s1),
            ("+ BoW", fps2, gba2, ate2, s2)]:
        print(f"  {nombre:28s} {fps:6.1f} {gba:8.1f} s {100*a:6.1f}cm "
              f"{len(sx.eventos_bucle):7d}")
    print("=" * 70)
    print("Mismos resultados (mira el ATE y los bucles), menos tiempo: eso")
    print("es una gemela. Si el ATE cambiara, la 'optimizacion' seria un bug")
    print("con buena prensa. El siguiente orden de magnitud (GTSAM, C++)")
    print("exige toolchain: ver el README y los ejercicios.")
    print("\nAhora corre `python verificacion.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
