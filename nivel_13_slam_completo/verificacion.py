#!/usr/bin/env python3
"""Examen del nivel 13: el SLAM completo, y la ablacion que lo justifica.

Genera los datos si faltan. Numeros medidos (corredor de 200 frames):
completo 5.8 cm ATE-KF / 2 bucles / 0 perdidos; sin bucle 6.1; sin BA 148.5.
Si todo pasa: NIVEL 13: VERIFICADO.

Uso:
    python verificacion.py
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
DATOS = AQUI / "data" / "corredor"

from evaluacion import ate, load_tum_positions
from slam import SLAM

spec = importlib.util.spec_from_file_location("n13", AQUI / "13_slam.py")
n13 = importlib.util.module_from_spec(spec)
sys.modules["n13"] = n13
spec.loader.exec_module(n13)

fallos = []


def check(nombre: str, ok: bool, detalle: str = "") -> None:
    estado = "OK " if ok else "FALLO"
    print(f"  [{estado}] {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def main() -> int:
    if not (DATOS / "images").is_dir():
        print("Generando el corredor...")
        if subprocess.run([sys.executable, str(AQUI / "genera_datos.py")]).returncode:
            raise SystemExit("genera_datos.py fallo")

    K = n13.leer_calibracion(DATOS / "calib.txt")
    rutas = sorted((DATOS / "images").glob("*.png"))
    grises = [cv2.imread(str(r), cv2.IMREAD_GRAYSCALE) for r in rutas]
    gt = load_tum_positions(DATOS / "groundtruth.txt")
    print(f"Verificando sobre el corredor ({len(grises)} frames)\n")

    # ── El sistema completo ─────────────────────────────────────────────────
    poses, s = n13.correr(grises, K, usar_ba=True, usar_bucle=True)
    est = np.stack([T[:3, 3] for T in poses])
    m_online = ate(est, gt)
    m_kf = n13.ate_keyframes(s, gt)

    check("el sistema inicializa y trackea sin perderse", s.n_perdidos == 0,
          f"{s.n_perdidos} frames perdidos")
    check("se insertaron >=10 keyframes", len(s.kf_poses) >= 10,
          f"{len(s.kf_poses)} (medido: 18)")
    check("el mapa tiene >=2000 puntos", len(s.mapa) >= 2000,
          f"{len(s.mapa)} pts (medido: 5246)")

    # El bucle DISPARA (y es un bucle de verdad: el corredor vuelve al inicio,
    # asi que los keyframes emparejados deben estar en extremos opuestos de
    # la secuencia).
    check("se cerro al menos 1 bucle", len(s.eventos_bucle) >= 1,
          f"{len(s.eventos_bucle)} bucles: {s.eventos_bucle}")
    if s.eventos_bucle:
        gaps = [n - v for v, n in s.eventos_bucle]
        check("los bucles son LEJANOS en el tiempo (no con el vecino)",
              min(gaps) >= s.LOOP_TEMPORAL_GAP,
              f"gap minimo {min(gaps)} keyframes (umbral {s.LOOP_TEMPORAL_GAP})")

    # EL numero del nivel.
    print(f"\n  ATE online:     {100*m_online['rmse']:6.1f} cm")
    print(f"  ATE keyframes:  {100*m_kf['rmse']:6.1f} cm  <- la metrica honesta\n")
    check("ATE de keyframes < 15 cm", m_kf["rmse"] < 0.15,
          f"{100*m_kf['rmse']:.1f} cm (medido: 5.8)")

    # ── La ablacion: sin BA, el sistema colapsa ─────────────────────────────
    p_noba, s_noba = n13.correr(grises, K, usar_ba=False, usar_bucle=False)
    m_noba = n13.ate_keyframes(s_noba, gt)
    check("sin BA el sistema COLAPSA (>10x peor)",
          m_noba["rmse"] > 10 * m_kf["rmse"],
          f"{100*m_noba['rmse']:.1f} cm vs {100*m_kf['rmse']:.1f} cm con BA")

    p_nolo, s_nolo = n13.correr(grises, K, usar_ba=True, usar_bucle=False)
    m_nolo = n13.ate_keyframes(s_nolo, gt)
    check("el cierre de bucle no empeora la trayectoria de keyframes",
          m_kf["rmse"] <= m_nolo["rmse"] * 1.1,
          f"sin bucle {100*m_nolo['rmse']:.1f} -> con bucle {100*m_kf['rmse']:.1f} cm")

    # ── Sanidad geometrica del mapa ─────────────────────────────────────────
    # El gauge se fijo en la INIT: la profundidad mediana de los puntos
    # iniciales vale 1 (nivel 10). Y todo el mapa debe estar DELANTE de la
    # primera camara.
    from slam import proyectar
    pts = np.stack(list(s.mapa.puntos.values()))
    _, Zs = proyectar(K, np.eye(4), pts)
    check("todo el mapa esta delante de la camara inicial",
          bool((Zs > 0).mean() > 0.99),
          f"{100*(Zs > 0).mean():.1f}% de los puntos")

    print()
    if fallos:
        print(f"NIVEL 13: {len(fallos)} fallo(s): {', '.join(fallos)}")
        return 1
    print("NIVEL 13: VERIFICADO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
