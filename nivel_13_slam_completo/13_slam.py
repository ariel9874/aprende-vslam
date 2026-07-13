#!/usr/bin/env python3
"""
Nivel 13 — El SLAM completo
===========================

Todo el curso, ensamblado y corriendo sobre el CORREDOR (ida y vuelta, con
un bucle genuino al final):

    INIT -> TRACK (PnP contra mapa local) -> keyframes -> BA de ventana
         -> cierre de bucle VERIFICADO -> grafo Sim(3)

Y la ablacion que demuestra que cada pieza carga peso:

    | sistema completo | sin cierre de bucle | sin BA |

Uso:
    python 13_slam.py                # el sistema completo + la ablacion
    python 13_slam.py --solo-completo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from evaluacion import ate, load_tum_positions, umeyama_alignment
from slam import SLAM

AQUI = Path(__file__).resolve().parent
DATOS = AQUI / "data" / "corredor"


def leer_calibracion(path: Path) -> np.ndarray:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            fx, fy, cx, cy = [float(v) for v in line.split()[:4]]
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    raise SystemExit("calib.txt vacio")


def correr(grises, K, usar_ba: bool, usar_bucle: bool, verbose: bool = False):
    """Corre el SLAM y devuelve (poses_online, tracker)."""
    s = SLAM(K, usar_ba=usar_ba, usar_bucle=usar_bucle)
    poses = []
    for i, gray in enumerate(grises):
        T, info = s.procesar(gray)
        poses.append(T.copy())
        if verbose and (i % 25 == 0 or info["loop"] or info["estado"] == "LOST"):
            extra = ""
            if info["kf"]:
                extra += "  <- keyframe"
            if info["loop"]:
                extra += f"  <<< BUCLE {info['loop'][0]} -> {info['loop'][1]}"
            print(f"frame {i:3d} | {info['estado']:6s} | inliers "
                  f"{info['n_inliers']:4d} | mapa {info['n_mapa']:5d}{extra}")
    return poses, s


def ate_keyframes(tracker, gt) -> dict:
    """ATE sobre la trayectoria FINAL de keyframes (la metrica honesta).

    Cada keyframe sabe en que frame nacio, asi que se compara contra el GT de
    ESE instante. (Ver el docstring de SLAM.trayectoria_kfs: la trayectoria
    online no ve las correcciones del backend.)
    """
    frames, pos = tracker.trayectoria_kfs()
    return ate(pos, gt[frames])


def main() -> int:
    parser = argparse.ArgumentParser(description="Nivel 13: SLAM completo")
    parser.add_argument("--solo-completo", action="store_true")
    args = parser.parse_args()

    img_dir = DATOS / "images"
    rutas = sorted(img_dir.glob("*.png"))
    if not rutas:
        raise SystemExit(f"No hay imagenes en {img_dir}. "
                         "Corre `python genera_datos.py` primero.")
    K = leer_calibracion(DATOS / "calib.txt")
    grises = [cv2.imread(str(r), cv2.IMREAD_GRAYSCALE) for r in rutas]
    gt = load_tum_positions(DATOS / "groundtruth.txt")
    salida = AQUI / "salida"
    salida.mkdir(exist_ok=True)

    print(f"Corredor: {len(grises)} frames (la camara va a x=7 y VUELVE a 0)\n")

    # ── El sistema completo ───────────────────────────────────────────────────
    poses, s = correr(grises, K, usar_ba=True, usar_bucle=True, verbose=True)
    est = np.stack([T[:3, 3] for T in poses])
    m_online = ate(est, gt)
    m_kf = ate_keyframes(s, gt)
    print(f"\nMapa: {len(s.mapa)} puntos | {len(s.kf_poses)} keyframes | "
          f"{len(s.eventos_bucle)} bucles | {s.n_perdidos} frames perdidos")
    print(f"\nATE online (poses emitidas):        {100*m_online['rmse']:5.1f} cm")
    print(f"ATE final de keyframes (la buena):  {100*m_kf['rmse']:5.1f} cm")
    print("Las dos metricas NO miden lo mismo. Las poses emitidas se CONGELAN")
    print("al salir: cuando el bucle corrige el mapa en el frame 190, nadie")
    print("reescribe los frames 0..189. La trayectoria de keyframes SI se")
    print("reescribe (el BA y el grafo la tocan): es la que refleja el estado")
    print("real del sistema, y la que reporta ORB-SLAM.")

    if args.solo_completo:
        return 0

    # ── La ablacion: cada pieza carga peso ───────────────────────────────────
    print("\n" + "=" * 66)
    print("ABLACION (misma secuencia, mismo frontend; ATE de keyframes):")
    print(f"  {'configuracion':22s} {'ATE-KF':>9s} {'ATE online':>11s} "
          f"{'perdidos':>9s}")
    res = {}
    for nombre, ba, bucle in [("sin BA (ni bucle)", False, False),
                              ("sin cierre de bucle", True, False),
                              ("sistema completo", True, True)]:
        if nombre == "sistema completo":
            s2, p2 = s, poses
        else:
            p2, s2 = correr(grises, K, usar_ba=ba, usar_bucle=bucle)
        e2 = np.stack([T[:3, 3] for T in p2])
        mo = ate(e2, gt)
        mk = ate_keyframes(s2, gt)
        res[nombre] = mk["rmse"]
        print(f"  {nombre:22s} {100*mk['rmse']:7.1f} cm {100*mo['rmse']:9.1f} cm "
              f"{s2.n_perdidos:9d}")
    print("=" * 66)
    print("Sin BA, el mapa nace torcido y nada lo endereza: la triangulacion")
    print("y el PnP se retroalimentan con su propio error. El BA es la pieza")
    print("que sostiene el mapa; el bucle refina lo que el BA ya hizo bien.")

    # ── Grafico: la serie temporal, no la planta ─────────────────────────────
    # (Leccion 16 del repo padre: una trayectoria de IDA Y VUELTA se solapa
    #  consigo misma en planta y no se ve nada. Hay que graficar x(t).)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sc, R, t = umeyama_alignment(est, gt)
        al = (sc * (R @ est.T)).T + t
        err = np.linalg.norm(al - gt, axis=1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
        ax1.plot(gt[:, 0], "k--", lw=1.5, label="verdad")
        ax1.plot(al[:, 0], lw=1.5, label="SLAM (alineado)")
        ax1.set_ylabel("x [m]")
        ax1.set_title("El corredor: ida y vuelta (por eso NO se grafica en planta)")
        ax1.legend(), ax1.grid(alpha=0.3)
        ax2.plot(100 * err, color="tab:red", lw=1.2)
        for kf_v, kf_n in s.eventos_bucle:
            ax2.axvline(len(grises) * kf_n / max(len(s.kf_poses), 1), color="tab:blue",
                        ls=":", alpha=0.7)
        ax2.set_xlabel("frame"), ax2.set_ylabel("error [cm]")
        ax2.grid(alpha=0.3)
        fig.savefig(salida / "corredor.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\nGrafica: {salida / 'corredor.png'}")
    except ImportError:
        pass

    print("Ahora corre `python verificacion.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
