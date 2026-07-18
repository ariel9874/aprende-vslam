#!/usr/bin/env python3
"""Examen del nivel 22: el factor de IMU, verificado.

Sin dataset (mundo sintético sembrado); corre en segundos. Cinco actos:

  1. La PREINTEGRACIÓN es correcta: con IMU perfecta y sin sesgo, encadenar
     los deltas reproduce la verdad del circuito entero (62 s a 100 Hz).
  2. El TRUCO DE FORSTER funciona: corregir los deltas a primer orden con
     el jacobiano del sesgo ≈ re-integrar con el sesgo nuevo (sin pagar la
     re-integración). El error remanente es O(b²): pequeño y medido.
  3. La IMU SOLA no navega: el dead reckoning con el sesgo sin corregir
     deriva metros (la doble integración no perdona).
  4. EL APAGÓN (el acto estrella): con la visión apagada en plena curva,
     el factor coast pierde por mucho contra el factor de IMU — el coast
     ASUME velocidad constante; la IMU MIDIÓ el giro.
  5. El SESGO se descubre solo: el grafo lo estima al 97% de su valor real
     — y donde de verdad paga es dentro del apagón.

Uso:
    python verificacion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import grafo_vio
from mundo_imu import BIAS_GYRO_REAL, generar, rmse_xy
from preintegracion import corregir_por_bias, integrar_muerto, preintegrar

fallos = []


def check(nombre: str, ok: bool, detalle: str = "") -> None:
    estado = "OK " if ok else "FALLO"
    print(f"  [{estado}] {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def rmse_en(idx, poses, gt):
    d = poses[idx, :2] - gt[idx, :2]
    return float(np.sqrt((d ** 2).sum(axis=1).mean()))


def main() -> int:
    print("Verificando el factor de IMU (sin dataset)\n")

    # ── Acto 1: la preintegracion reproduce la verdad ────────────────────────
    print("[1/5] Preintegracion con IMU perfecta...")
    mp = generar(sin_ruido=True)
    dr = integrar_muerto(mp["tramos"], np.zeros(3), np.array([1.0, 0.0]))
    err = rmse_xy(dr, mp["gt"])
    check("encadenar los deltas reproduce el circuito (62 s, RMSE < 5 cm)",
          err < 0.05, f"{100*err:.2f} cm en 26 m de recorrido (medido: 0.02 "
          "-- punto medio; con Euler puro eran 20 cm)")

    # ── Acto 2: el truco de Forster ──────────────────────────────────────────
    print("\n[2/5] La correccion de primer orden del sesgo...")
    # el tramo con MAS curva (en una recta a=0 y no habria nada que
    # corregir: el primer intento de este examen cayo justo ahi)
    tramo = max(mp["tramos"], key=lambda t: sum(abs(a[1]) for _, a in t))
    b = 0.03
    pre0 = preintegrar(tramo, 0.0)
    pre_b = preintegrar(tramo, b)             # la verdad: re-integrado
    d_th_c, d_v_c, d_p_c = corregir_por_bias(pre0, b)
    e_sin = abs(pre0["d_th"] - pre_b["d_th"])
    e_con = abs(d_th_c - pre_b["d_th"])
    check("en theta la correccion es EXACTA (delta-theta es lineal en el sesgo)",
          e_con < 1e-12, f"sin corregir {e_sin:.2e} -> corregido {e_con:.2e}")
    e_v_sin = float(np.abs(pre0["d_v"] - pre_b["d_v"]).max())
    e_v_con = float(np.abs(d_v_c - pre_b["d_v"]).max())
    check("en delta-v el error queda a segundo orden (>= 20x menos)",
          e_v_con < e_v_sin / 20,
          f"{e_v_sin:.2e} -> {e_v_con:.2e} ({e_v_sin/max(e_v_con,1e-18):.0f}x"
          " menos, sin re-integrar ni una muestra)")

    # ── Acto 3: la IMU sola no navega ────────────────────────────────────────
    print("\n[3/5] Dead reckoning (la IMU sola)...")
    m = generar()
    dr_b = integrar_muerto(m["tramos"], np.zeros(3), np.array([1.0, 0.0]))
    dr_ok = integrar_muerto(m["tramos"], np.zeros(3), np.array([1.0, 0.0]),
                            bias=BIAS_GYRO_REAL)
    r_b, r_ok = rmse_xy(dr_b, m["gt"]), rmse_xy(dr_ok, m["gt"])
    print(f"  con el sesgo sin corregir: {100*r_b:.0f} cm | "
          f"con sesgo perfecto: {100*r_ok:.0f} cm")
    check("el sesgo sin corregir cuesta >= 5x (rota el mundo entero)",
          r_b > 5 * r_ok, f"{r_b/r_ok:.1f}x (medido: 13x -- 771 vs 60 cm)")
    check("y AUN con sesgo perfecto deriva (la doble integracion)",
          r_ok > 0.2, f"{100*r_ok:.0f} cm: por eso la IMU se FUSIONA, "
          "no se integra a secas")

    # ── Acto 4: el apagon en la curva ────────────────────────────────────────
    print("\n[4/5] El apagon visual en plena curva (el acto estrella)...")
    apag = np.array(m["apagados"])
    r_coast = grafo_vio.optimizar(m, usar_imu=False)
    r_vio = grafo_vio.optimizar(m, usar_imu=True)
    a_coast = rmse_en(apag, r_coast["poses"], m["gt"])
    a_vio = rmse_en(apag, r_vio["poses"], m["gt"])
    print(f"  error EN el apagon: coast {100*a_coast:.1f} cm | "
          f"VIO {100*a_vio:.1f} cm")
    check("la IMU cruza el apagon >= 5x mejor que el coast",
          a_coast > 5 * a_vio,
          f"{a_coast/a_vio:.1f}x (medido: 13.5x -- 62.2 vs 4.6 cm). El coast "
          "ASUME; la IMU MIDE -- la informacion que al nivel 17 le faltaba")
    check("y el VIO completo clava el circuito (RMSE total < 10 cm)",
          rmse_xy(r_vio["poses"], m["gt"]) < 0.10,
          f"{100*rmse_xy(r_vio['poses'], m['gt']):.1f} cm (medido: 4.7)")

    # ── Acto 5: el sesgo, descubierto ────────────────────────────────────────
    print("\n[5/5] El sesgo del giroscopo, estimado por el grafo...")
    b_est = float(np.median(r_vio["biases"]))
    check("el grafo descubre el sesgo (dentro del 25% del real)",
          abs(b_est - BIAS_GYRO_REAL) < 0.25 * BIAS_GYRO_REAL,
          f"estimado {b_est:.4f} vs real {BIAS_GYRO_REAL} -- nadie se lo dijo:"
          " lo dedujo de que vision e IMU no cuadraban de otro modo")
    r_sinb = grafo_vio.optimizar(m, usar_imu=True, estimar_bias=False)
    a_sinb = rmse_en(apag, r_sinb["poses"], m["gt"])
    check("y donde el sesgo PAGA es dentro del apagon (>= 1.2x)",
          a_sinb > 1.2 * a_vio,
          f"sin estimar {100*a_sinb:.1f} vs estimando {100*a_vio:.1f} cm -- "
          "fuera del apagon la vision corrige al giroscopo mentiroso; dentro,"
          " no hay quien lo corrija")

    print()
    if fallos:
        print(f"NIVEL 22: {len(fallos)} fallo(s): {', '.join(fallos)}")
        return 1
    print("NIVEL 22: VERIFICADO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
