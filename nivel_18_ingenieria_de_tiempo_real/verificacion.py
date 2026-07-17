#!/usr/bin/env python3
"""Examen del nivel 18: el metodo — perfilar, sustituir, VERIFICAR.

Tres actos. Los dos primeros NO necesitan dataset (los tests de equivalencia
son sinteticos — y son LA leccion); el tercero mide la escalera real sobre
fr2_xyz (600 frames, ~10 min).

  1. EQUIVALENCIA del BA: la gemela vectorizada (ba_rapido.py) da LO MISMO
     que el didactico sobre el mismo problema — a precision de maquina
     (medido: dif. maxima 4e-16) — y mas rapido.
  2. EQUIVALENCIA del BoW: sobre una base sintetica, la consulta BoW
     devuelve el MISMO candidato que la fuerza bruta.
  3. LA ESCALERA: con las gemelas enchufadas, el sistema es mas rapido con
     el MISMO resultado (paridad de ATE — si el ATE cambiara, la
     "optimizacion" seria un bug con buena prensa).

Uso:
    python verificacion.py [--root <ruta_fr2_xyz>]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

from ba_rapido import bundle_adjustment_rapido
from bow import BolsaDePalabras
from bundle_adjustment import bundle_adjustment
from dataset import leer_trayectoria_tum

spec = importlib.util.spec_from_file_location("n18", AQUI / "18_tiempo_real.py")
n18 = importlib.util.module_from_spec(spec)
sys.modules["n18"] = n18
spec.loader.exec_module(n18)

fallos = []


def check(nombre: str, ok: bool, detalle: str = "") -> None:
    estado = "OK " if ok else "FALLO"
    print(f"  [{estado}] {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def problema_sintetico(rng):
    """Una ventana de BA realista: 8 camaras en arco, 400 puntos, ruido."""
    K = np.array([[520.0, 0, 320], [0, 520, 240], [0, 0, 1]])
    pts_gt = {p: np.array([rng.uniform(-2, 2), rng.uniform(-1.5, 1.5),
                           rng.uniform(4, 8)]) for p in range(400)}
    poses_gt = {}
    for k in range(8):
        T = np.eye(4)
        T[:3, 3] = [k * 0.3, 0.02 * k, 0.0]
        poses_gt[k] = T
    obs = []
    for k, T in poses_gt.items():
        Tc = np.linalg.inv(T)
        for p, X in pts_gt.items():
            Xc = Tc[:3, :3] @ X + Tc[:3, 3]
            if Xc[2] > 0.1:
                uv = np.array([520 * Xc[0] / Xc[2] + 320,
                               520 * Xc[1] / Xc[2] + 240])
                obs.append((k, p, uv + rng.normal(0, 0.5, 2)))
    poses0 = {k: T.copy() for k, T in poses_gt.items()}
    for k in range(2, 8):
        poses0[k][:3, 3] += rng.normal(0, 0.05, 3)
    pts0 = {p: X + rng.normal(0, 0.05, 3) for p, X in pts_gt.items()}
    return K, poses0, pts0, obs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="secuencia fr2_xyz ya descargada")
    args = parser.parse_args()

    # ── Acto 1: la equivalencia del BA (sin dataset) ─────────────────────────
    print("[1/3] Test de equivalencia: BA didactico vs gemela vectorizada...")
    rng = np.random.default_rng(3)
    K, poses0, pts0, obs = problema_sintetico(rng)
    h1, h2 = [], []
    t0 = time.perf_counter()
    pd_, xd = bundle_adjustment(K, poses0, pts0, obs, fixed_kfs={0, 1},
                                iterations=8, historial=h1)
    t_did = time.perf_counter() - t0
    t0 = time.perf_counter()
    pr_, xr = bundle_adjustment_rapido(K, poses0, pts0, obs, fixed_kfs={0, 1},
                                       iterations=8, historial=h2)
    t_rap = time.perf_counter() - t0

    dif_p = max(np.abs(pd_[k] - pr_[k]).max() for k in pd_)
    dif_x = max(np.abs(xd[p] - xr[p]).max() for p in xd)
    print(f"  {len(obs)} observaciones | didactico {t_did:.2f} s | "
          f"gemela {t_rap:.2f} s")
    check("la gemela da LO MISMO (poses, dif < 1e-9)", dif_p < 1e-9,
          f"dif maxima {dif_p:.1e} (medido: 4e-16 — precision de maquina)")
    check("la gemela da LO MISMO (puntos, dif < 1e-9)", dif_x < 1e-9,
          f"dif maxima {dif_x:.1e}")
    check("los costos por iteracion COINCIDEN (mismo camino LM)",
          len(h1) == len(h2) and max(abs(a - b) / max(abs(a), 1)
                                     for a, b in zip(h1, h2)) < 1e-9,
          f"{len(h1)} evaluaciones identicas")
    check("y es MAS RAPIDA (>= 2x en este problema)", t_did / t_rap >= 2.0,
          f"{t_did/t_rap:.1f}x (medido: ~4.4x; crece con el problema)")

    # ── Acto 2: la equivalencia del BoW (sin dataset) ────────────────────────
    print("\n[2/3] Test de equivalencia: BoW vs fuerza bruta...")
    rng = np.random.default_rng(5)
    # 20 "keyframes" sinteticos de 300 descriptores ORB; el query es una
    # copia del KF 7 con ruido de bits (la misma escena, re-observada).
    base = [rng.integers(0, 256, (300, 32), dtype=np.uint8) for _ in range(20)]
    query = base[7].copy()
    ruido = rng.integers(0, 256, query.shape, dtype=np.uint8) \
        & (rng.random(query.shape) < 0.02).astype(np.uint8) * 255
    query = np.bitwise_xor(query, ruido)

    bow = BolsaDePalabras(n_palabras=256)
    bow.entrenar(np.vstack(base[:6]))
    for i, d in enumerate(base):
        bow.indexar(i, d)
    top = bow.consultar(query, top_k=3)

    import cv2
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    votos = []
    for i, d in enumerate(base):
        ms = bf.knnMatch(query, d, k=2)
        votos.append((sum(1 for m, n_ in ms if m.distance < 0.75 * n_.distance), i))
    fuerza_bruta = max(votos)[1]
    print(f"  BoW top-1: KF {top[0][0]} (coseno {top[0][1]:.3f}) | "
          f"fuerza bruta: KF {fuerza_bruta} | verdad: KF 7")
    check("BoW y fuerza bruta coinciden en el candidato (y es el correcto)",
          top and top[0][0] == 7 and fuerza_bruta == 7,
          "mismo contrato, costo sub-lineal")

    # ── Acto 3: la escalera sobre datos reales ───────────────────────────────
    root = Path(args.root) if args.root else n18.DATASET_DEFAULT
    if not (root / "rgb.txt").is_file():
        print("\n[3/3] SALTADO: no hay fr2_xyz (corre descarga_datos.py o "
              "pasa --root). Los actos 1-2 bastan para el examen de las "
              "gemelas; la escalera completa es 18_tiempo_real.py.")
        print()
        if fallos:
            print(f"NIVEL 18: {len(fallos)} fallo(s): {', '.join(fallos)}")
            return 1
        print("NIVEL 18: VERIFICADO (sin el acto 3: falta el dataset)")
        return 0

    print("\n[3/3] La escalera sobre fr2_xyz (600 frames, ~10 min)...")
    gt_ts, gt_pos = leer_trayectoria_tum(root / "groundtruth.txt")
    ts, s0, fps0, gba0, _ = n18.correr(root, max_frames=600, verbose=True)
    ate0 = n18.ate_kfs(s0, ts, gt_ts, gt_pos)
    ts, s2, fps2, gba2, _ = n18.correr(root, max_frames=600,
                                       ba_fn=bundle_adjustment_rapido,
                                       bow=BolsaDePalabras(), verbose=True)
    ate2 = n18.ate_kfs(s2, ts, gt_ts, gt_pos)
    print(f"  referencia: {fps0:.1f} fps, GBA {gba0:.0f} s, "
          f"ATE {100*ate0:.1f} cm | gemelas: {fps2:.1f} fps, "
          f"GBA {gba2:.0f} s, ATE {100*ate2:.1f} cm")
    check("las gemelas aceleran el tracking (fps >= 1.2x)",
          fps2 >= 1.2 * fps0, f"{fps2/fps0:.2f}x")
    check("y el BA global (>= 2x)", gba0 >= 2.0 * gba2,
          f"{gba0/max(gba2,1e-9):.1f}x")
    check("con PARIDAD de resultado (|dif ATE| < 0.5 cm)",
          abs(ate2 - ate0) < 0.005,
          f"{100*ate0:.2f} vs {100*ate2:.2f} cm")

    print()
    if fallos:
        print(f"NIVEL 18: {len(fallos)} fallo(s): {', '.join(fallos)}")
        return 1
    print("NIVEL 18: VERIFICADO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
