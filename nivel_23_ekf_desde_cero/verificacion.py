#!/usr/bin/env python3
"""Examen del nivel 23: el EKF desde cero, verificado.

Sin dataset (todo sintético y sembrado); corre en segundos. Cuatro actos:

  1. FUSIONAR: la fórmula K·innovación == mínimos cuadrados pesados, y el
     recursivo == el batch (la información suma; sigma cae como 1/√N).
  2. EL FILTRO LINEAL: el KF es EXACTAMENTE el grafo de factores lineal
     resuelto por recursión (estado Y covarianza, a precisión de máquina) —
     y estima la velocidad sin sensor de velocidad.
  3. EL EKF: linealizar basta para localizar (20x mejor que integrar y
     rezar) — y el bug del ángulo sin envolver, reproducido y medido.
  4. EL EKF DE ERROR: cruza el apagón del nivel 22, descubre el sesgo en
     vivo, y pierde ante el smoother EXACTAMENTE donde la teoría dice
     (la novatada de la primera vuelta; lección 25).

Uso:
    python verificacion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import ekf_de_error
import ekf_localizacion
import fusion_1d
import kalman_lineal
import mundo_imu

# las referencias del nivel 22, medidas sobre ESTE MISMO mundo y semilla
# (el mundo esta duplicado bit a bit: la comparacion es legitima)
COAST_APAGON_22 = 0.622
GRAFO_TOTAL_22 = 0.047
GRAFO_APAGON_22 = 0.048

fallos = []


def check(nombre: str, ok: bool, detalle: str = "") -> None:
    estado = "OK " if ok else "FALLO"
    print(f"  [{estado}] {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def main() -> int:
    print("Verificando el EKF desde cero (sin dataset)\n")

    # ── Acto 1: fusionar ─────────────────────────────────────────────────────
    print("[1/4] Fusionar dos numeros...")
    rng = np.random.default_rng(7)
    z1, s1, z2, s2 = 4.1, 0.3, 3.8, 0.1
    x_k, _ = fusion_1d.fusionar(z1, s1, z2, s2)
    # el mismo problema como minimos cuadrados pesados, resuelto por lstsq
    A = np.array([[1.0 / s1], [1.0 / s2]])
    b = np.array([z1 / s1, z2 / s2])
    x_ls = float(np.linalg.lstsq(A, b, rcond=None)[0][0])
    check("K*innovacion == minimos cuadrados pesados",
          abs(x_k - x_ls) < 1e-12, f"difieren {abs(x_k - x_ls):.1e}")
    zs = 5.0 + rng.normal(0, 0.5, 1000)
    x_rec, s_rec, _ = fusion_1d.estimar_recursivo(zs, 0.5)
    check("procesar de a una == guardarlo todo y promediar",
          abs(x_rec - float(np.mean(zs))) < 1e-12,
          f"difieren {abs(x_rec - float(np.mean(zs))):.1e}")
    check("la informacion SUMA: sigma cae como 1/sqrt(N)",
          abs(s_rec - 0.5 / np.sqrt(1000)) < 1e-12,
          f"{s_rec:.6f} == {0.5/np.sqrt(1000):.6f}")

    # ── Acto 2: el filtro lineal ─────────────────────────────────────────────
    print("\n[2/4] El filtro de Kalman lineal (el carrito)...")
    sim = kalman_lineal.simular()
    kf = kalman_lineal.filtrar(sim["z"])
    lote = kalman_lineal.resolver_batch(sim["z"])
    d_x = float(np.abs(kf["x"][-1] - lote["x_final"]).max())
    d_P = float(np.abs(kf["P"] - lote["P_final"]).max())
    check("el filtro == el grafo lineal (el estado final)",
          d_x < 1e-8, f"difieren {d_x:.1e} (802 incognitas resueltas de golpe"
          " vs 2 numeros en memoria)")
    check("y su P == la covarianza marginal del grafo (Schur implicito)",
          d_P < 1e-10, f"difieren {d_P:.1e} -- marginalizar el pasado ES lo "
          "que el filtro hace paso a paso")
    e_vel = float(np.sqrt(np.mean((kf["x"][:, 1] - sim["verdad"][:, 1]) ** 2)))
    v_diff = np.diff(sim["z"]) / kalman_lineal.DT
    e_diff = float(np.sqrt(np.mean((v_diff - sim["verdad"][1:, 1]) ** 2)))
    check("la velocidad, estimada SIN sensor de velocidad (>= 5x vs derivar)",
          e_diff > 5 * e_vel, f"derivar la medicion: {e_diff:.1f} m/s; el "
          f"filtro: {e_vel:.2f} m/s ({e_diff/e_vel:.0f}x) -- el canal es "
          "P[0,1]")

    # ── Acto 3: el EKF ───────────────────────────────────────────────────────
    print("\n[3/4] El EKF (localizacion, mapa conocido)...")
    m3 = ekf_localizacion.generar()
    e_dr = ekf_localizacion.rmse_xy(ekf_localizacion.dead_reckoning(m3),
                                    m3["gt"])
    e_ekf = ekf_localizacion.rmse_xy(ekf_localizacion.correr_ekf(m3),
                                     m3["gt"])
    e_mal = ekf_localizacion.rmse_xy(
        ekf_localizacion.correr_ekf(m3, con_envolver=False), m3["gt"])
    check("linealizar basta: EKF >= 5x mejor que dead reckoning",
          e_dr > 5 * e_ekf,
          f"{100*e_dr:.1f} -> {100*e_ekf:.1f} cm ({e_dr/e_ekf:.0f}x)")
    check("el bug del angulo: sin envolver la innovacion, >= 5x peor",
          e_mal > 5 * e_ekf,
          f"{100*e_mal:.0f} cm ({e_mal/e_ekf:.0f}x peor) -- una resta ciega "
          "de rumbos convirtio 0.01 rad de error en una 'sorpresa' de 2*pi")

    # ── Acto 4: el EKF de error ──────────────────────────────────────────────
    print("\n[4/4] El EKF de error (el mundo del nivel 22)...")
    m4 = mundo_imu.generar()
    apag = np.array(m4["apagados"])
    gt = m4["gt"]

    def rmse_en(idx, poses):
        d = poses[idx, :2] - gt[idx, :2]
        return float(np.sqrt((d ** 2).sum(axis=1).mean()))

    r = ekf_de_error.correr(m4)
    e_tot = mundo_imu.rmse_xy(r["poses"], gt)
    e_apag = rmse_en(apag, r["poses"])
    check("el ESKF clava el circuito EN LINEA (RMSE total < 10 cm)",
          e_tot < 0.10, f"{100*e_tot:.1f} cm (medido: 7.2) -- online, sin "
          "volver a tocar ninguna pose emitida")
    check("y cruza el apagon >= 2.5x mejor que el coast del nivel 22",
          e_apag < COAST_APAGON_22 / 2.5,
          f"coast {100*COAST_APAGON_22:.1f} -> ESKF {100*e_apag:.1f} cm "
          f"({COAST_APAGON_22/e_apag:.1f}x): la IMU MIDE el giro a oscuras")
    b_fin = float(r["biases"][-1])
    check("el sesgo se descubre EN VIVO (dentro del 25% del real)",
          abs(b_fin - mundo_imu.BIAS_GYRO_REAL)
          < 0.25 * mundo_imu.BIAS_GYRO_REAL,
          f"estimado {b_fin:.4f} vs real {mundo_imu.BIAS_GYRO_REAL} -- y ya "
          "estaba en 0.028 a los 12 s (el grafo del 22 lo supo AL FINAL)")
    r_sinb = ekf_de_error.correr(m4, estimar_bias=False)
    e_sinb = mundo_imu.rmse_xy(r_sinb["poses"], gt)
    check("no modelar el sesgo rompe al filtro (>= 5x peor: consistencia)",
          e_sinb > 5 * e_tot,
          f"{100*e_sinb:.0f} vs {100*e_tot:.1f} cm ({e_sinb/e_tot:.0f}x). El "
          "grafo del 22 sin sesgo dio 4.3 cm: el smoother re-pondera todo; "
          "el filtro overconfiado se envenena a si mismo")
    mitad = len(gt) // 2
    e_v1 = rmse_en(np.arange(1, mitad), r["poses"])
    e_v2 = rmse_en(np.arange(mitad, len(gt)), r["poses"])
    check("la novatada de la primera vuelta (vuelta 2 al menos 2x mejor)",
          e_v1 > 2 * e_v2,
          f"vuelta 1: {100*e_v1:.1f} cm | vuelta 2: {100*e_v2:.1f} cm -- el "
          "mapa converge, pero las poses de la vuelta 1 ya fueron emitidas")
    check("filtro vs smoother: mismo mundo, y el smoother gana (leccion 25)",
          GRAFO_TOTAL_22 < e_tot < 3 * GRAFO_TOTAL_22,
          f"ESKF {100*e_tot:.1f} vs grafo {100*GRAFO_TOTAL_22:.1f} cm total; "
          f"en el apagon {100*e_apag:.1f} vs {100*GRAFO_APAGON_22:.1f} -- el "
          "filtro emite y sella; el smoother corrige el pasado")

    print()
    if fallos:
        print(f"NIVEL 23: {len(fallos)} fallo(s): {', '.join(fallos)}")
        return 1
    print("NIVEL 23: VERIFICADO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
