#!/usr/bin/env python3
"""Examen del nivel 10: el mapa persistente gana a la odometria 2D-2D.

Genera los datos si faltan y corre AMBOS sistemas sobre la misma secuencia.
Numeros medidos: VO 2D-2D 18.6 cm, PnP contra mapa 8.7 cm (53% mejor),
2046 puntos en 6 keyframes, 0 frames perdidos.
Si todo pasa: NIVEL 10: VERIFICADO.

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

from evaluacion import ate, load_tum_positions

AQUI = Path(__file__).resolve().parent
DATOS = AQUI / "data" / "secuencia"

spec = importlib.util.spec_from_file_location("n10", AQUI / "10_pnp_mapa.py")
n10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n10)

fallos = []


def check(nombre: str, ok: bool, detalle: str = "") -> None:
    estado = "OK " if ok else "FALLO"
    print(f"  [{estado}] {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def main() -> int:
    if not (DATOS / "images").is_dir():
        print("Generando la secuencia sintetica...")
        if subprocess.run([sys.executable, str(AQUI / "genera_datos.py")]).returncode:
            raise SystemExit("genera_datos.py fallo")

    K = n10.leer_calibracion(DATOS / "calib.txt")
    rutas = sorted((DATOS / "images").glob("*.png"))
    grises = [cv2.imread(str(r), cv2.IMREAD_GRAYSCALE) for r in rutas]
    gt = load_tum_positions(DATOS / "groundtruth.txt")
    print(f"Verificando sobre {len(grises)} frames\n")

    # Correr el tracker PnP y la VO 2D-2D sobre las MISMAS imagenes.
    tr = n10.TrackerPnP(K)
    poses, estados = [], []
    for g in grises:
        T, info = tr.procesar(g)
        poses.append(T.copy())
        estados.append(info["estado"])

    vo = n10.VO2D2D(K)
    poses_vo = [vo.procesar(g).copy() for g in grises]

    # 1. El sistema inicializa y trackea de principio a fin.
    check("el tracker inicializa (sale de INIT)", "TRACK" in estados,
          f"init en el frame {estados.index('TRACK') if 'TRACK' in estados else -1}")
    n_coast = sum(1 for e in estados if e == "COAST")
    check("ningun frame perdido (COAST)", n_coast == 0, f"{n_coast} en coast")

    # 2. El mapa crece por keyframes y tiene tamano razonable.
    check("se insertaron >=4 keyframes", tr.n_keyframes >= 4,
          f"{tr.n_keyframes} (medido: 6)")
    check("el mapa tiene >=500 puntos", len(tr.mapa) >= 500,
          f"{len(tr.mapa)} pts (medido: 2046)")

    # 3. El mapa es GEOMETRICAMENTE sano: todo delante de la camara inicial.
    _, Z = n10.proyectar(K, np.eye(4), tr.mapa.puntos)
    check("todo el mapa esta delante de la camara inicial", bool((Z > 0).all()),
          f"Z min {Z.min():.2f}")

    # El GAUGE se fija en la INIT: la profundidad mediana de los puntos
    # INICIALES vale 1.0 por convencion. Los puntos que nacen despues HEREDAN
    # esa escala pero no la re-fijan (miran zonas mas lejanas de la escena),
    # asi que la mediana del mapa COMPLETO no tiene por que ser 1 — y de hecho
    # sale 2.1. Medir el gauge sobre todo el mapa seria medir otra cosa.
    Z_init = Z[:tr.n_puntos_init]
    check("el gauge esta aplicado (mediana de los puntos de INIT == 1.0)",
          abs(float(np.median(Z_init)) - 1.0) < 1e-6,
          f"init: {np.median(Z_init):.6f} | mapa completo: {np.median(Z):.2f} "
          "(los puntos nuevos heredan la escala, no la fijan)")

    # 4. EL NUMERO DEL NIVEL: el PnP contra el mapa gana a la VO 2D-2D.
    est_pnp = np.stack([T[:3, 3] for T in poses])
    est_vo = np.stack([T[:3, 3] for T in poses_vo])
    m_pnp, m_vo = ate(est_pnp, gt), ate(est_vo, gt)
    print(f"\n  VO 2D-2D:        {100*m_vo['rmse']:5.1f} cm")
    print(f"  PnP + mapa:      {100*m_pnp['rmse']:5.1f} cm")
    print(f"  mejora:          {100*(1-m_pnp['rmse']/m_vo['rmse']):5.0f}%\n")
    check("ATE del PnP < 12 cm", m_pnp["rmse"] < 0.12,
          f"{100*m_pnp['rmse']:.1f} cm (medido: 8.7)")
    check("el PnP mejora >=30% sobre la VO 2D-2D",
          m_pnp["rmse"] < 0.7 * m_vo["rmse"],
          f"{100*m_vo['rmse']:.1f} -> {100*m_pnp['rmse']:.1f} cm")

    print()
    if fallos:
        print(f"NIVEL 10: {len(fallos)} fallo(s): {', '.join(fallos)}")
        return 1
    print("NIVEL 10: VERIFICADO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
