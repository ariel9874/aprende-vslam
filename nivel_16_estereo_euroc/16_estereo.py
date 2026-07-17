#!/usr/bin/env python3
"""
Nivel 16 — Estéreo (EuRoC): la cámara derecha se vuelve real
============================================================

El SLAM métrico del nivel 15 sobre un DRON con dos cámaras calibradas
(EuRoC V1_01_easy). El pipeline es el mismo; lo nuevo vive en dataset.py:

  rig calibrado -> rectificar (epipolares = filas) -> disparidad (SGBM)
                -> z = bf/d -> el MISMO tracker métrico del nivel 15

Y las dos trampas de dataset que este driver resuelve:
  - los timestamps vienen en NANOsegundos (÷1e9);
  - el ground truth vive en el frame del CUERPO (IMU): hay que corregir el
    brazo de palanca de ~7 cm hacia la cámara (leer_gt_euroc), o el ATE
    queda contaminado por un error que ROTA con la pose.

Uso:
    python descarga_datos.py             # V1_01_easy (~1.1 GB), una vez
    python 16_estereo.py                 # la secuencia completa (dron, 6-DoF)
    python 16_estereo.py --max-frames 800
    python 16_estereo.py --sin-residuo   # ablacion: BA sin la fila estereo
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from dataset import CargadorEstereo, asociar_por_timestamp, leer_gt_euroc
from evaluacion import ate
from slam import SLAM

AQUI = Path(__file__).resolve().parent
DATASET_DEFAULT = AQUI / "data" / "V1_01_easy"


def correr(root: Path, max_frames: int = 0, usar_ba: bool = True,
           usar_bucle: bool = True, usar_guiado: bool = True,
           usar_residuo: bool = True, verbose: bool = False):
    """Corre el SLAM estéreo sobre una secuencia EuRoC. Semilla fija."""
    cv2.setRNGSeed(7)
    loader = CargadorEstereo(root)
    if verbose:
        print(f"  rig: baseline {100*loader.rig.baseline:.2f} cm | "
              f"bf {loader.stereo_bf:.1f} px·m (medidos por la calibracion)")
    # La camara del tracker es la IZQUIERDA RECTIFICADA (pinhole puro) y el
    # bf es el del RIG: el mismo numero que fabrico las profundidades.
    s = SLAM(loader.rig.K, bf=loader.stereo_bf, usar_ba=usar_ba,
             usar_bucle=usar_bucle, usar_guiado=usar_guiado,
             usar_residuo=usar_residuo)
    ts_list, pos, estados = [], [], []
    t0 = time.perf_counter()
    for i, (ts, gray, prof) in enumerate(loader):
        if max_frames and i >= max_frames:
            break
        T, info = s.procesar(gray, prof)
        ts_list.append(ts)
        pos.append(T[:3, 3].copy())
        estados.append(info["estado"])
        if verbose and (i % 200 == 0 or info["loop"]
                        or info["estado"] == "RELOC"):
            extra = f"  <<< BUCLE {info['loop']}" if info["loop"] else ""
            if info["estado"] == "RELOC":
                extra += f"  <<< RELOC vs KF {s.eventos_reloc[-1][1]}"
            print(f"  frame {i:4d} | {info['estado']:7s} | inliers "
                  f"{info['n_inliers']:3d} | mapa {info['n_mapa']:6d} | "
                  f"KFs {len(s.kf_poses):3d}{extra}")
    if verbose:
        dt = time.perf_counter() - t0
        print(f"  ({len(ts_list)} frames en {dt:.0f} s = "
              f"{len(ts_list)/dt:.1f} fps, SGBM incluido)")
    return np.array(ts_list), np.array(pos), estados, s


def evaluar_online(ts, pos, estados, gt_ts, gt_pos) -> dict:
    start = next((i for i, e in enumerate(estados)
                  if e in ("INIT-OK", "TRACK")), 0)
    assoc = asociar_por_timestamp(ts[start:], gt_ts, max_dt=0.02)
    ok = assoc >= 0
    if ok.sum() < 3:
        return {"rmse": float("nan"), "scale_sim": float("nan"), "n": 0}
    est, gt = pos[start:][ok], gt_pos[assoc[ok]]
    m = ate(est, gt, with_scale=False)
    m["scale_sim"] = ate(est, gt)["scale"]
    m["n"] = int(ok.sum())
    return m


def evaluar_kfs(tracker, ts, gt_ts, gt_pos) -> dict:
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
    parser = argparse.ArgumentParser(description="Nivel 16: estereo EuRoC")
    parser.add_argument("--root", default=None,
                        help="secuencia EuRoC (default: V1_01_easy)")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--sin-residuo", action="store_true")
    parser.add_argument("--sin-guiado", action="store_true")
    parser.add_argument("--sin-bucle", action="store_true")
    parser.add_argument("--sin-gba", action="store_true")
    args = parser.parse_args()

    root = Path(args.root) if args.root else DATASET_DEFAULT
    if not (root / "mav0" / "cam1" / "data.csv").is_file():
        raise SystemExit(f"No hay secuencia EuRoC estereo en {root}.\n"
                         "Corre `python descarga_datos.py` (~1.1 GB) o pasa "
                         "--root.")

    ts, pos, estados, s = correr(root, max_frames=args.max_frames,
                                 usar_bucle=not args.sin_bucle,
                                 usar_guiado=not args.sin_guiado,
                                 usar_residuo=not args.sin_residuo,
                                 verbose=True)

    gt_ts, gt_pos = leer_gt_euroc(root)
    print(f"\nMapa: {len(s.mapa)} puntos | {len(s.kf_poses)} keyframes | "
          f"{len(s.eventos_bucle)} bucles SE(3) | {s.n_perdidos} perdidos | "
          f"{len(s.eventos_reloc)} relocs")
    if s.inliers_hist:
        print(f"Inliers de PnP: mediana {int(np.median(s.inliers_hist))}")

    m_on = evaluar_online(ts, pos, estados, gt_ts, gt_pos)
    m_kf0 = evaluar_kfs(s, ts, gt_ts, gt_pos)
    print(f"\nATE online  (RIGIDO): {100*m_on['rmse']:6.1f} cm | "
          f"escala similitud {m_on['scale_sim']:.3f}")
    print(f"ATE final-KF (RIGIDO): {100*m_kf0['rmse']:5.1f} cm | "
          f"escala similitud {m_kf0['scale_sim']:.3f}  (pre-GBA)")

    if not args.sin_gba:
        etiqueta = ("con residuo estereo" if s.residuo_bf > 0
                    else "residuo APAGADO (ablacion)")
        print(f"\nBA GLOBAL offline ({s.GBA_ITERS} iteraciones, {etiqueta})...")
        t0 = time.perf_counter()
        s.global_bundle_adjustment()
        m_kf = evaluar_kfs(s, ts, gt_ts, gt_pos)
        print(f"ATE final-KF (RIGIDO): {100*m_kf['rmse']:5.1f} cm | "
              f"escala similitud {m_kf['scale_sim']:.3f}  "
              f"({time.perf_counter()-t0:.0f} s)")
        print("\nLa linea completa (todo RIGIDO, metros del rig):")
        print(f"  online {100*m_on['rmse']:.1f} -> final-KF "
              f"{100*m_kf0['rmse']:.1f} -> + BA global {100*m_kf['rmse']:.1f} cm"
              f" | escala {m_kf['scale_sim']:.3f}")
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
        ax1.plot(gtm[:, 0], gtm[:, 1], "k--", lw=1.2, label="GT (camara)")
        ax1.plot(al[:, 0], al[:, 1], lw=1.2, label="keyframes (RIGIDO)")
        ax1.set_xlabel("x [m]"), ax1.set_ylabel("y [m]")
        ax1.set_title(f"{root.name}: un dron en 6-DoF, en metros")
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

    print("Ahora corre `python verificacion.py` (no necesita el dataset).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
