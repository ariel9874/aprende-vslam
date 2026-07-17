#!/usr/bin/env python3
"""
Nivel 15 — RGB-D y escala métrica
=================================

El SLAM del nivel 14, ahora con un sensor de profundidad: SLAM en METROS de
verdad. Tres cambios y un chequeo de honestidad:

  - INIT instantánea: el primer frame con profundidad YA es un mapa métrico.
  - El RESIDUO DE PROFUNDIDAD en el BA ([u, v, u_R]): el ancla métrica.
  - El bucle va en SE(3), no Sim(3): la escala es medición, no gauge.
  - El ATE se evalúa con alineación RÍGIDA, y la escala de similitud se
    reporta APARTE: ≈ 1.000 es la prueba de que el mapa está en metros.

Y el plato fuerte: fr1_desk — la secuencia que el nivel 14 NO pudo cruzar
(562 frames perdidos) — se cruza ENTERA. No por un umbral: porque crear mapa
ya no requiere paralaje, y el residuo de profundidad sostiene la escala.

Uso:
    python descarga_datos.py                # fr1_desk (~330 MB), una vez
    python 15_rgbd.py                       # la secuencia completa
    python 15_rgbd.py --sin-residuo         # la ablacion (bf = 0)
    python 15_rgbd.py --root <otra_secuencia_TUM_con_depth>
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


def correr(root: Path, max_frames: int = 0, usar_ba: bool = True,
           usar_bucle: bool = True, usar_guiado: bool = True,
           usar_residuo: bool = True, verbose: bool = False):
    """Corre el SLAM RGB-D sobre una secuencia TUM con depth.

    Devuelve (timestamps, posiciones_online, estados, tracker). Semilla fija
    para que las corridas sean comparables y el examen reproducible.
    """
    cv2.setRNGSeed(7)
    K, dist = camara_tum(root.name)
    loader = SecuenciaTUM(root, con_profundidad=True)

    maps = None
    if np.any(dist != 0.0):
        maps = cv2.initUndistortRectifyMap(K, dist, None, K, (640, 480),
                                           cv2.CV_32FC1)

    s = SLAM(K, usar_ba=usar_ba, usar_bucle=usar_bucle,
             usar_guiado=usar_guiado, usar_residuo=usar_residuo)
    ts_list, pos, estados = [], [], []
    saltados = 0
    t0 = time.perf_counter()
    for i, (ts, gray, prof) in enumerate(loader):
        if max_frames and i >= max_frames:
            break
        # El bug del mapa MIXTO (leccion 36 del padre): el stream de depth
        # puede arrancar tarde (fr1_desk: 6 frames sin pareja). Inicializar
        # sin profundidad crearia un mapa a escala gauge que luego se mezcla
        # con puntos en metros. Se ESPERA al primer frame con depth.
        if prof is None and s.estado == "INIT":
            saltados += 1
            continue
        if maps is not None:
            gray = cv2.remap(gray, maps[0], maps[1], cv2.INTER_LINEAR)
            if prof is not None:
                # La profundidad se rectifica con NEAREST: interpolar
                # bilinealmente A TRAVES de una discontinuidad inventa
                # profundidades que no existen en la escena.
                prof = cv2.remap(prof, maps[0], maps[1], cv2.INTER_NEAREST)
        T, info = s.procesar(gray, prof)
        ts_list.append(ts)
        pos.append(T[:3, 3].copy())
        estados.append(info["estado"])
        if verbose and (i % 100 == 0 or info["loop"]):
            extra = f"  <<< BUCLE {info['loop']}" if info["loop"] else ""
            print(f"  frame {i:4d} | {info['estado']:7s} | inliers "
                  f"{info['n_inliers']:3d} | mapa {info['n_mapa']:6d} | "
                  f"KFs {len(s.kf_poses):3d}{extra}")
    if verbose:
        dt = time.perf_counter() - t0
        print(f"  ({len(ts_list)} frames en {dt:.0f} s = "
              f"{len(ts_list)/dt:.1f} fps | {saltados} saltados sin depth)")
    return np.array(ts_list), np.array(pos), estados, s


def evaluar_online(ts, pos, estados, gt_ts, gt_pos) -> dict:
    """ATE RÍGIDO de las poses emitidas, desde que el mapa existe."""
    start = next((i for i, e in enumerate(estados)
                  if e in ("INIT-OK", "TRACK")), 0)
    assoc = asociar_por_timestamp(ts[start:], gt_ts, max_dt=0.02)
    ok = assoc >= 0
    if ok.sum() < 3:
        return {"rmse": float("nan"), "scale_sim": float("nan"), "n": 0}
    est, gt = pos[start:][ok], gt_pos[assoc[ok]]
    m = ate(est, gt, with_scale=False)          # RIGIDO: metrico de verdad
    m["scale_sim"] = ate(est, gt)["scale"]      # la escala, como CHEQUEO
    m["n"] = int(ok.sum())
    return m


def evaluar_kfs(tracker, ts, gt_ts, gt_pos) -> dict:
    """ATE RÍGIDO de la trayectoria final de keyframes + escala de chequeo."""
    frames, pos = tracker.trayectoria_kfs()
    assoc = asociar_por_timestamp(ts[frames], gt_ts, max_dt=0.05)
    ok = assoc >= 0
    if ok.sum() < 3:
        return {"rmse": float("nan"), "scale_sim": float("nan"), "n": 0}
    est, gt = pos[ok], gt_pos[assoc[ok]]
    m = ate(est, gt, with_scale=False)
    m["scale_sim"] = ate(est, gt)["scale"]
    m["n"] = int(ok.sum())
    return m


def main() -> int:
    parser = argparse.ArgumentParser(description="Nivel 15: RGB-D metrico")
    parser.add_argument("--root", default=None,
                        help="secuencia TUM con depth (default: fr1_desk)")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--sin-residuo", action="store_true",
                        help="ablacion: bf = 0 (init metrica, BA solo 2D)")
    parser.add_argument("--sin-guiado", action="store_true")
    parser.add_argument("--sin-bucle", action="store_true")
    parser.add_argument("--sin-gba", action="store_true")
    args = parser.parse_args()

    root = Path(args.root) if args.root else DATASET_DEFAULT
    if not (root / "depth.txt").is_file():
        raise SystemExit(f"No hay dataset RGB-D en {root}.\n"
                         "Corre `python descarga_datos.py` o pasa --root.")

    print(f"Secuencia: {root.name}"
          + (f" (primeros {args.max_frames} frames)" if args.max_frames else "")
          + f" | residuo de profundidad: {'OFF' if args.sin_residuo else 'ON'}\n")

    ts, pos, estados, s = correr(root, max_frames=args.max_frames,
                                 usar_bucle=not args.sin_bucle,
                                 usar_guiado=not args.sin_guiado,
                                 usar_residuo=not args.sin_residuo,
                                 verbose=True)

    gt_ts, gt_pos = leer_trayectoria_tum(root / "groundtruth.txt")
    print(f"\nMapa: {len(s.mapa)} puntos | {len(s.kf_poses)} keyframes | "
          f"{len(s.eventos_bucle)} bucles SE(3) | {s.n_perdidos} perdidos")
    if s.inliers_hist:
        print(f"Inliers de PnP: mediana {int(np.median(s.inliers_hist))}")

    m_on = evaluar_online(ts, pos, estados, gt_ts, gt_pos)
    m_kf0 = evaluar_kfs(s, ts, gt_ts, gt_pos)
    print(f"\nATE online  (RIGIDO): {100*m_on['rmse']:6.1f} cm | "
          f"escala similitud {m_on['scale_sim']:.3f}")
    print(f"ATE final-KF (RIGIDO): {100*m_kf0['rmse']:5.1f} cm | "
          f"escala similitud {m_kf0['scale_sim']:.3f}  (pre-GBA)")

    if not args.sin_gba:
        etiqueta = ("con residuo de profundidad" if s.residuo_bf > 0
                    else "residuo APAGADO (ablacion)")
        print(f"\nBA GLOBAL offline ({s.GBA_ITERS} iteraciones, {etiqueta})...")
        t0 = time.perf_counter()
        s.global_bundle_adjustment()
        m_kf = evaluar_kfs(s, ts, gt_ts, gt_pos)
        print(f"ATE final-KF (RIGIDO): {100*m_kf['rmse']:5.1f} cm | "
              f"escala similitud {m_kf['scale_sim']:.3f}  "
              f"({time.perf_counter()-t0:.0f} s)")
        print("\nLa linea completa (todo RIGIDO, sin regalar la escala):")
        print(f"  online {100*m_on['rmse']:.1f} -> final-KF "
              f"{100*m_kf0['rmse']:.1f} -> + BA global {100*m_kf['rmse']:.1f} cm"
              f" | escala {m_kf['scale_sim']:.3f} (metros de verdad si ~1.0)")
    else:
        m_kf = m_kf0

    # ── grafico: planta + error(t) ───────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from evaluacion import umeyama_alignment

        frames, kf_pos = s.trayectoria_kfs()
        assoc = asociar_por_timestamp(ts[frames], gt_ts, max_dt=0.05)
        ok = assoc >= 0
        _, R, t = umeyama_alignment(kf_pos[ok], gt_pos[assoc[ok]],
                                    with_scale=False)
        al = (R @ kf_pos[ok].T).T + t
        gtm = gt_pos[assoc[ok]]
        err = np.linalg.norm(al - gtm, axis=1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        ax1.plot(gtm[:, 0], gtm[:, 1], "k--", lw=1.2, label="mocap (verdad)")
        ax1.plot(al[:, 0], al[:, 1], lw=1.2,
                 label="keyframes (alineacion RIGIDA)")
        ax1.set_xlabel("x [m]"), ax1.set_ylabel("y [m]")
        ax1.set_title(f"{root.name} en planta (metros de verdad)")
        ax1.axis("equal"), ax1.legend(), ax1.grid(alpha=0.3)
        ax2.plot(frames[ok], 100 * err, color="tab:red", lw=1.0)
        for _, kf_n in s.eventos_bucle:
            ax2.axvline(s.kf_frame.get(kf_n, 0), color="tab:blue",
                        ls=":", alpha=0.6)
        ax2.set_xlabel("frame"), ax2.set_ylabel("error [cm]")
        ax2.set_title("error por keyframe (azul: bucles SE(3))")
        ax2.grid(alpha=0.3)
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
