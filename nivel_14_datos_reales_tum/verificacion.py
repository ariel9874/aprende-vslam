#!/usr/bin/env python3
"""Examen del nivel 14: el SLAM sobreviviendo a datos REALES.

Corre sobre los primeros 600 frames de fr2_xyz (el examen completo de los
3669 esta en el README: son los numeros del mantenedor). Tres actos, con los
numeros medidos al construir el nivel:

  1. El sistema completo (guiado ON) trackea fr2_xyz sin perderse (0
     perdidos, 41 KFs, 2 bucles) y el ATE final de keyframes tras el BA
     global queda en POCOS cm (medido: 1.4 -> 0.8 cm).
  2. El mapa-espejismo, evitado (leccion 14): la covisibilidad extiende el
     mapa local mas alla de la recencia y los keyframes OBSERVAN antes de
     crear — el mapa queda compacto y re-observado. Construyendo este nivel
     lo medimos al reves: sin esto, la sesion entera acumulaba 96 000 puntos
     duplicados y 35 cm de ATE que ningun BA podia bajar.
  3. La ablacion: sin matching guiado los inliers caen de forma medible
     (medido: mediana 914 ON vs 574 OFF, +59%).

Necesita el dataset (python descarga_datos.py, ~2.1 GB) o `--root <ruta>`
apuntando a una copia ya descargada de rgbd_dataset_freiburg2_xyz.
Dura ~10 min en un CPU moderno (dos pasadas + un BA global de 50 iteraciones).

Uso:
    python verificacion.py [--root <ruta_fr2_xyz>]
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

from dataset import leer_trayectoria_tum

spec = importlib.util.spec_from_file_location("n14", AQUI / "14_datos_reales.py")
n14 = importlib.util.module_from_spec(spec)
sys.modules["n14"] = n14
spec.loader.exec_module(n14)

N_FRAMES = 600      # el examen usa un tramo; el README reporta la secuencia entera

fallos = []


def check(nombre: str, ok: bool, detalle: str = "") -> None:
    estado = "OK " if ok else "FALLO"
    print(f"  [{estado}] {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="secuencia fr2_xyz ya descargada")
    args = parser.parse_args()

    root = Path(args.root) if args.root else n14.DATASET_DEFAULT
    if not (root / "rgb.txt").is_file():
        raise SystemExit(f"No hay dataset en {root}.\n"
                         "Corre `python descarga_datos.py` (~2.1 GB) o pasa "
                         "--root <ruta a rgbd_dataset_freiburg2_xyz>.")

    gt_ts, gt_pos = leer_trayectoria_tum(root / "groundtruth.txt")
    print(f"Verificando sobre {root.name} (primeros {N_FRAMES} frames)\n")

    # ── Acto 1: el sistema completo ──────────────────────────────────────────
    print("[1/3] Sistema completo (guiado ON)...")
    ts, pos, estados, s = n14.correr(root, max_frames=N_FRAMES)

    check("el sistema inicializa y trackea sin perderse", s.n_perdidos == 0,
          f"{s.n_perdidos} frames perdidos")
    check("se insertaron >=25 keyframes", len(s.kf_poses) >= 25,
          f"{len(s.kf_poses)} (medido: 41)")
    check("se cerro al menos 1 bucle", len(s.eventos_bucle) >= 1,
          f"{len(s.eventos_bucle)} bucles: {s.eventos_bucle}")
    med_on = int(np.median(s.inliers_hist)) if s.inliers_hist else 0
    check("inliers de PnP: mediana >= 300", med_on >= 300,
          f"mediana {med_on} (medido: 914)")

    m_on = n14.evaluar_online(ts, pos, estados, gt_ts, gt_pos)
    m_kf0 = n14.evaluar_kfs(s, ts, gt_ts, gt_pos)

    # El BA global offline (leccion 26-27): converge y no rompe nada.
    print(f"\n      BA global offline ({s.GBA_ITERS} iteraciones)...")
    historial: list = []
    s.global_bundle_adjustment(historial=historial)
    m_kf = n14.evaluar_kfs(s, ts, gt_ts, gt_pos)

    print(f"\n  ATE online:               {100*m_on['rmse']:6.1f} cm")
    print(f"  ATE final-KF (pre-GBA):   {100*m_kf0['rmse']:6.1f} cm")
    print(f"  ATE final-KF (post-GBA):  {100*m_kf['rmse']:6.1f} cm  <- EL numero\n")

    check("ATE final de keyframes < 5 cm", m_kf["rmse"] < 0.05,
          f"{100*m_kf['rmse']:.1f} cm (medido: 0.8)")
    check("el BA global CONVERGE (el costo baja)",
          len(historial) >= 2 and historial[-1] < historial[0],
          f"costo {historial[0]:.0f} -> {historial[-1]:.0f}"
          if len(historial) >= 2 else "sin historial")
    check("el BA global no rompe el mapa (ATE no empeora >20%)",
          m_kf["rmse"] <= m_kf0["rmse"] * 1.2 + 1e-6,
          f"{100*m_kf0['rmse']:.1f} -> {100*m_kf['rmse']:.1f} cm")

    # ── Acto 2: el mapa-espejismo, evitado (leccion 14) ──────────────────────
    # Con el mapa local por RECENCIA (nivel 13), cada re-visita fabricaba
    # duplicados desplazados por la deriva: medimos 16 566 puntos en este
    # mismo tramo (y 96 000 y 35 cm de ATE en la sesion entera — un mapa que
    # ningun BA puede arreglar, porque dos copias coherentes del mismo mundo
    # no se reconcilian optimizando). La covisibilidad + "observar antes que
    # crear" mantienen el mapa compacto y RE-OBSERVADO.
    print("\n[2/3] Leccion 14: el mapa-espejismo, evitado...")
    n_locales = len(s._kfs_locales())
    obs_por_pt = len(s.mapa.obs) / max(len(s.mapa), 1)
    print(f"  mapa: {len(s.mapa)} puntos | {len(s.mapa.obs)} observaciones "
          f"({obs_por_pt:.1f} por punto) | mapa local: {n_locales} KFs")
    check("la covisibilidad extiende el mapa local mas alla de la recencia",
          n_locales > s.LOCAL_KFS,
          f"{n_locales} KFs locales (recencia sola: {s.LOCAL_KFS})")
    check("el mapa queda compacto: < 12000 puntos en este tramo",
          len(s.mapa) < 12000,
          f"{len(s.mapa)} (medido: 8882; por recencia era 16566)")
    check("los puntos se RE-observan (>= 3 obs por punto en promedio)",
          obs_por_pt >= 3.0,
          f"{obs_por_pt:.1f} obs/punto (por recencia era ~2.4)")

    # ── Acto 3: la ablacion del guiado ───────────────────────────────────────
    print("\n[3/3] Ablacion: matching guiado OFF (misma secuencia)...")
    _, _, _, s_off = n14.correr(root, max_frames=N_FRAMES, usar_guiado=False)
    med_off = int(np.median(s_off.inliers_hist)) if s_off.inliers_hist else 0
    print(f"  inliers mediana: guiado ON {med_on} vs OFF {med_off} | "
          f"perdidos: ON {s.n_perdidos} vs OFF {s_off.n_perdidos}")
    check("el guiado sube los inliers de forma medible (>=15%)",
          med_on >= 1.15 * max(med_off, 1),
          f"ON {med_on} vs OFF {med_off} "
          f"({100*(med_on/max(med_off,1)-1):.0f}% mas)")
    check("el guiado no pierde mas frames que el matching global",
          s.n_perdidos <= s_off.n_perdidos,
          f"ON {s.n_perdidos} vs OFF {s_off.n_perdidos}")

    print()
    if fallos:
        print(f"NIVEL 14: {len(fallos)} fallo(s): {', '.join(fallos)}")
        return 1
    print("NIVEL 14: VERIFICADO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
