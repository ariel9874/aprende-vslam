#!/usr/bin/env python3
"""Examen del nivel 16: el rig estéreo, verificado SIN el dataset.

EuRoC pesa ~1.1 GB, así que este examen fabrica su propio rig: dos cámaras
sintéticas con baseline horizontal CONOCIDO (10 cm, fx = 400 -> bf = 40) y
un par de imágenes de un plano fronto-paralelo a Z conocida. Tres actos:

  1. EL RIG: cv2.stereoRectify debe recuperar el baseline y el bf desde los
     sensor.yaml, y rectificar un par YA rectificado debe ser ~la identidad.
  2. LA PROFUNDIDAD: la imagen derecha es la izquierda desplazada d píxeles
     -> SGBM debe recuperar d -> z = bf/d exacta (aquí, 2.5 m).
  3. LA IDENTIDAD real=virtual: u_R medida (u_L − d) es EXACTAMENTE la
     u_R = u − bf/z que el nivel 15 sintetizaba, y el residuo del BA con
     bf > 0 crece a 3 filas y da ~0 en una observación consistente.

La secuencia real (V1_01_easy) es el trabajo del driver, no del examen:
sus números viven en el README.

Uso:
    python verificacion.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

from bundle_adjustment import residual_and_jacobians
from dataset import CargadorEstereo, RigEstereo

FX, FY, CX, CY, W, H = 400.0, 400.0, 320.0, 240.0, 640, 480
BASELINE = 0.10                              # 10 cm -> bf = fx·b = 40 px·m

fallos = []


def check(nombre: str, ok: bool, detalle: str = "") -> None:
    estado = "OK " if ok else "FALLO"
    print(f"  [{estado}] {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def _sensor_yaml(t_bs: str) -> str:
    """Un sensor.yaml EuRoC ya rectificado (sin distorsión)."""
    return ("sensor_type: camera\n"
            "T_BS:\n  cols: 4\n  rows: 4\n"
            f"  data: [{t_bs}]\n"
            f"resolution: [{W}, {H}]\n"
            "camera_model: pinhole\n"
            f"intrinsics: [{FX}, {FY}, {CX}, {CY}]\n"
            "distortion_model: radial-tangential\n"
            "distortion_coefficients: [0.0, 0.0, 0.0, 0.0]\n")


def fabricar_rig(imgs_izq, imgs_der) -> Path:
    """Escribe mav0/cam0 y cam1 con las imágenes dadas. La cámara derecha
    está desplazada +baseline en X del cuerpo (= frame de la izquierda)."""
    root = Path(tempfile.mkdtemp()) / "V_fixture"
    tbs_izq = "1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1"
    tbs_der = f"1,0,0,{BASELINE}, 0,1,0,0, 0,0,1,0, 0,0,0,1"
    for cam, tbs, imgs in (("cam0", tbs_izq, imgs_izq),
                           ("cam1", tbs_der, imgs_der)):
        d = root / "mav0" / cam
        (d / "data").mkdir(parents=True)
        (d / "sensor.yaml").write_text(_sensor_yaml(tbs), encoding="utf-8")
        lineas = ["#timestamp [ns],filename"]
        for i, img in enumerate(imgs):
            ts = 100_000_000 + i * 50_000_000
            cv2.imwrite(str(d / "data" / f"{ts}.png"), img)
            lineas.append(f"{ts},{ts}.png")
        (d / "data.csv").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return root


def main() -> int:
    print("Verificando el estereo con un rig FABRICADO (sin dataset)\n")

    # ── Acto 1: el rig recupera su propia geometria ──────────────────────────
    print("[1/3] El rig: stereoRectify recupera baseline y bf...")
    blanco = np.full((H, W), 128, np.uint8)
    root = fabricar_rig([blanco], [blanco])
    rig = RigEstereo(root)

    check("baseline recuperado (10 cm)", abs(rig.baseline - BASELINE) < 1e-3,
          f"{100*rig.baseline:.2f} cm")
    check("bf = fx * b (= 40 px*m)", abs(rig.bf - FX * BASELINE) < 0.1,
          f"{rig.bf:.2f}")
    check("la camara izquierda rectificada conserva fx",
          abs(rig.K[0, 0] - FX) < 1.0, f"fx {rig.K[0, 0]:.1f}")
    rng = np.random.default_rng(0)
    patron = rng.integers(0, 256, (H, W), np.uint8)
    L, _ = rig.rectify(patron, patron)
    nucleo = (slice(40, H - 40), slice(40, W - 40))
    dif = np.abs(L[nucleo].astype(int) - patron[nucleo].astype(int)).mean()
    check("rectificar lo YA rectificado ~ identidad", dif < 2.0,
          f"diferencia media {dif:.2f} niveles de gris")

    # ── Acto 2: profundidad por disparidad de un plano conocido ─────────────
    print("\n[2/3] La profundidad: un plano a Z conocida...")
    d = 16
    Z = FX * BASELINE / d                     # 40/16 = 2.5 m
    rng = np.random.default_rng(1)
    izq = rng.integers(0, 256, (H, W), np.uint8)
    izq = cv2.GaussianBlur(izq, (3, 3), 0)    # suaviza el aliasing sub-pixel
    der = np.zeros_like(izq)
    der[:, :W - d] = izq[:, d:]               # der(c) = izq(c+d) -> disp = d

    root2 = fabricar_rig([izq], [der])
    loader = CargadorEstereo(root2, num_disparities=32)
    _, _, prof = next(iter(loader))
    valida = prof > 0
    mediana = float(np.median(prof[valida])) if valida.any() else 0.0
    check("SGBM cubre el plano (>30% de pixeles con z)", valida.mean() > 0.3,
          f"{100*valida.mean():.0f}% validos")
    check("z = bf/d recupera el plano (2.5 m)", abs(mediana - Z) < 0.25,
          f"mediana {mediana:.2f} m (esperado {Z:.2f})")

    # ── Acto 3: la identidad real = virtual ──────────────────────────────────
    # u_R MEDIDA (el pixel en la derecha: u_L - d) contra u_R SINTETIZADA
    # desde z (u - bf/z, el nivel 15): la misma ecuacion, mismo numero.
    print("\n[3/3] La identidad: u_R medida == u_R sintetizada...")
    u_L = 300.0
    u_R_medida = u_L - d
    u_R_sintetizada = u_L - rig.bf / Z
    check("u_L - d == u_L - bf/z (real y virtual son la MISMA ecuacion)",
          abs(u_R_medida - u_R_sintetizada) < 1e-9,
          f"{u_R_medida:.3f} == {u_R_sintetizada:.3f}")

    # Y el residuo del BA: una observacion consistente [u, v, u_R] con bf>0
    # produce residuo (3,) ~ 0; con bf=0, residuo (2,) — la fila estereo
    # existe solo cuando hay medicion metrica.
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]])
    X_w = np.array([(u_L - CX) * Z / FX, (120.0 - CY) * Z / FY, Z])
    uv3 = np.array([u_L, 120.0, u_L - FX * BASELINE / Z])
    r3, _, _ = residual_and_jacobians(K, np.eye(4), X_w, uv3,
                                      bf=FX * BASELINE)
    r2, _, _ = residual_and_jacobians(K, np.eye(4), X_w, uv3, bf=0.0)
    check("el residuo del BA crece a 3 filas con bf>0 (y ~0 si es consistente)",
          len(r3) == 3 and float(np.abs(r3).max()) < 1e-9,
          f"residuo {np.abs(r3).max():.2e}")
    check("con bf=0 el residuo sigue siendo 2D (paridad monocular)",
          len(r2) == 2, f"{len(r2)} filas")

    print()
    if fallos:
        print(f"NIVEL 16: {len(fallos)} fallo(s): {', '.join(fallos)}")
        return 1
    print("NIVEL 16: VERIFICADO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
