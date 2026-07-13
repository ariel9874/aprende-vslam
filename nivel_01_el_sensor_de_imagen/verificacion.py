#!/usr/bin/env python3
"""Examen del nivel 01: la fisica del sensor, medida.

Si todo pasa: NIVEL 01: VERIFICADO.

Uso:
    python verificacion.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("n1", AQUI / "01_sensor.py")
n1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n1)

fallos = []


def check(nombre: str, ok: bool, detalle: str = "") -> None:
    estado = "OK " if ok else "FALLO"
    print(f"  [{estado}] {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def main() -> int:
    salida = AQUI / "salida"
    salida.mkdir(exist_ok=True)
    print("Verificando el sensor simulado\n")

    # 1. El shot noise sigue la ley de la raiz (pendiente 0.5 +- 0.05
    #    tras restar la varianza de lectura — photon transfer curve).
    pendiente = n1.experimento_ruido(salida)
    print()
    check("pendiente shot en 0.45..0.55", 0.45 <= pendiente <= 0.55,
          f"{pendiente:.3f}")

    # 2. La saturacion destruye informacion: un parche uniforme sobreexpuesto
    #    queda clavado en el full well con varianza ~0.
    e_sat = n1.exponer(np.full((100, 100), 1.0), exposicion=2.0)
    check("parche saturado: sigma < 1 e-", float(e_sat.std()) < 1.0,
          f"sigma {e_sat.std():.3f}, media {e_sat.mean():.0f}")

    # 3. El demosaico propio reconstruye la escena con error pequeno.
    rms = n1.experimento_bayer(salida)
    print()
    check("RMS del demosaico < 0.05", rms < 0.05, f"{rms:.4f} (medido: 0.022)")

    # 4. La inclinacion del rolling shutter coincide con la prediccion v/H.
    medida, esperada = n1.experimento_rolling(salida)
    print()
    check("pendiente rolling == v/H (+-10%)",
          abs(medida - esperada) <= 0.1 * esperada,
          f"{medida:.4f} vs {esperada:.4f}")

    print()
    if fallos:
        print(f"NIVEL 01: {len(fallos)} fallo(s): {', '.join(fallos)}")
        return 1
    print("NIVEL 01: VERIFICADO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
