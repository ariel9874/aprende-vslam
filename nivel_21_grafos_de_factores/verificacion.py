#!/usr/bin/env python3
"""Examen del nivel 21: cuatro grafos, un mundo, y las diferencias MEDIDAS.

Sin dataset (el mundo es sintético y sembrado): corre en segundos.

  1. SE(2): la geometría de juguete es correcta (componer∘between = id).
  2. El grafo COMPLETO clava el problema (8.2 cm desde 56 de odometría).
  3. La MARGINALIZACIÓN es EXACTA en el sistema lineal (Schur vs resolver
     todo: diferencia a precisión de máquina) — y su precio, el FILL-IN,
     se cuenta en entradas no nulas.
  4. El prior de Schur CONSERVA la información: ventana marginalizada
     medible-mente mejor que ventana cortada (medido: 25.9 vs 59.1 cm).
  5. La jerarquía completa: el grafo completo gana a todos; los bucles
     salvan al grafo de poses; el filtro emite y sella.

Uso:
    python verificacion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import filtro_ekf
import grafo_completo
import grafo_de_poses
import ventana_marginalizada as ventana
from mundo import between, componer, envolver, generar, rmse_xy

fallos = []


def check(nombre: str, ok: bool, detalle: str = "") -> None:
    estado = "OK " if ok else "FALLO"
    print(f"  [{estado}] {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def main() -> int:
    print("Verificando los cuatro grafos (sin dataset)\n")
    rng = np.random.default_rng(1)

    # ── Acto 1: SE(2) ────────────────────────────────────────────────────────
    print("[1/5] La geometria SE(2)...")
    peor = 0.0
    for _ in range(50):
        a = np.array([*rng.normal(0, 3, 2), rng.uniform(-np.pi, np.pi)])
        b = np.array([*rng.normal(0, 3, 2), rng.uniform(-np.pi, np.pi)])
        r = componer(a, between(a, b)) - b
        r[2] = envolver(r[2])
        peor = max(peor, float(np.abs(r).max()))
    check("componer(p, between(p, q)) == q (50 pares al azar)",
          peor < 1e-12, f"dif maxima {peor:.1e}")

    # ── Acto 2: el grafo completo ────────────────────────────────────────────
    print("\n[2/5] El grafo completo (el patron oro)...")
    m = generar()
    gt = m["gt"]
    rmse_odo = rmse_xy(m["inicial"], gt)
    rc = grafo_completo.optimizar(m)
    rmse_c = rmse_xy(rc["poses"], gt)
    print(f"  odometria pura: {100*rmse_odo:.1f} cm -> completo: "
          f"{100*rmse_c:.1f} cm")
    check("converge y clava el circuito (RMSE < 15 cm)", rmse_c < 0.15,
          f"{100*rmse_c:.1f} cm (medido: 8.2)")
    check("mejora >= 4x sobre la odometria", rmse_odo / rmse_c >= 4.0,
          f"{rmse_odo/rmse_c:.1f}x (medido: 6.8x)")
    check("el costo baja monotonicamente (Gauss-Newton sano)",
          all(b <= a * 1.001 for a, b in zip(rc["costos"], rc["costos"][1:])),
          f"{rc['costos'][0]:.0f} -> {rc['costos'][-1]:.0f}")

    # ── Acto 3: la marginalizacion es EXACTA (y su precio) ───────────────────
    print("\n[3/5] Schur: marginalizar es exacto — y el fill-in, real...")
    N, M = len(m["inicial"]), len(m["landmarks"])
    corte = N // 2
    poses0 = m["inicial"].copy()
    lms0 = np.zeros((M, 2))
    vistos = set()
    for i, j, z in m["obs"]:
        if j not in vistos:
            p = poses0[i]
            c, s = np.cos(p[2]), np.sin(p[2])
            lms0[j] = [p[0] + c * z[0] - s * z[1], p[1] + s * z[0] + c * z[1]]
            vistos.add(j)
    H, g, _ = grafo_completo.linearizar(m, poses0, lms0)
    idx_A, idx_B, _ = ventana.indices_a_marginalizar(m, corte, N, M)
    delta_full = np.linalg.solve(H + 1e-9 * np.eye(len(H)), -g)
    H_r, g_r = ventana.schur(H, g, idx_A, idx_B)
    delta_B = np.linalg.solve(H_r + 1e-9 * np.eye(len(H_r)), -g_r)
    dif = float(np.abs(delta_B - delta_full[idx_B]).max())
    check("resolver lo marginalizado == resolver TODO y mirar B",
          dif < 1e-6, f"dif maxima {dif:.1e} — el Schur del nivel 11, "
          "ahora borrando el pasado")
    nnz_antes = int((np.abs(H[np.ix_(idx_B, idx_B)]) > 1e-9).sum())
    nnz_despues = int((np.abs(H_r) > 1e-9).sum())
    check("el FILL-IN es real (el prior denso tiene mas entradas)",
          nnz_despues > 1.15 * nnz_antes,
          f"{nnz_antes} -> {nnz_despues} entradas no nulas "
          f"({nnz_despues/nnz_antes:.2f}x; medido: 1.33x — modesto aqui "
          "porque la ventana ya estaba muy acoplada; crece con el lag)")

    # ── Acto 4: conservar vs tirar ───────────────────────────────────────────
    print("\n[4/5] El prior de Schur conserva; cortar tira...")
    rv = ventana.optimizar_ventana(m, corte, marginalizar=True)
    rx = ventana.optimizar_ventana(m, corte, marginalizar=False)
    rmse_v = rmse_xy(rv["poses"][corte:], gt[corte:])
    rmse_x = rmse_xy(rx["poses"][corte:], gt[corte:])
    print(f"  ventana marginalizada: {100*rmse_v:.1f} cm | "
          f"cortada: {100*rmse_x:.1f} cm")
    check("marginalizar es medible-mente mejor que cortar (>= 1.4x)",
          rmse_x >= 1.4 * rmse_v,
          f"{rmse_x/rmse_v:.1f}x (medido: 2.3x)")

    # ── Acto 5: la jerarquia completa ────────────────────────────────────────
    print("\n[5/5] La jerarquia...")
    rp = grafo_de_poses.optimizar(m, con_bucles=True)
    rp0 = grafo_de_poses.optimizar(m, con_bucles=False)
    rf = filtro_ekf.correr(m)
    rmse_p = rmse_xy(rp["poses"], gt)
    rmse_p0 = rmse_xy(rp0["poses"], gt)
    rmse_f = rmse_xy(rf["tray"], gt)
    print(f"  poses con bucles {100*rmse_p:.1f} | sin bucles "
          f"{100*rmse_p0:.1f} | EKF online {100*rmse_f:.1f} cm")
    check("el grafo completo gana a TODAS las otras formas",
          rmse_c <= min(rmse_p, rmse_v, rmse_f) + 1e-9,
          f"completo {100*rmse_c:.1f} cm — conserva TODA la informacion")
    check("los bucles salvan al grafo de poses",
          rmse_p < 0.7 * rmse_p0,
          f"{100*rmse_p0:.1f} -> {100*rmse_p:.1f} cm con "
          f"{len(rp['bucles'])} bucles")
    check("el filtro mejora la odometria (y pierde ante el smoother)",
          rmse_f < rmse_odo and rmse_f >= rmse_c - 1e-9,
          f"EKF {100*rmse_f:.1f} cm — sorprendentemente cerca del completo "
          "en un mundo amable; su talon (consistencia) es el ejercicio 4")

    print()
    if fallos:
        print(f"NIVEL 21: {len(fallos)} fallo(s): {', '.join(fallos)}")
        return 1
    print("NIVEL 21: VERIFICADO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
