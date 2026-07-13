#!/usr/bin/env python3
"""
Nivel 00 — Hola, píxeles
========================

Una imagen digital ES una matriz de números. Este script la trata como tal:

    cargar ─▶ inspeccionar (shape, dtype) ─▶ gris A MANO ─▶ verificar vs OpenCV
           ─▶ histograma ─▶ negativo ─▶ recorte

Léelo de arriba a abajo: cada paso imprime lo que descubre y guarda su
resultado en `salida/`.

Uso:
    python 00_hola_pixeles.py                      # primer frame del dataset
    python 00_hola_pixeles.py --imagen foto.png    # cualquier imagen tuya
    python 00_hola_pixeles.py --root <secuencia>   # dataset TUM en otra ruta
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent
DATASET_DEFAULT = AQUI / "data" / "rgbd_dataset_freiburg1_xyz"


def encontrar_imagen(args) -> Path:
    """Resuelve qué imagen abrir: --imagen > --root > dataset del nivel."""
    if args.imagen:
        return Path(args.imagen)
    root = Path(args.root) if args.root else DATASET_DEFAULT
    rgb = sorted((root / "rgb").glob("*.png"))
    if not rgb:
        raise SystemExit(
            f"No hay imagenes en {root / 'rgb'}.\n"
            "Corre `python descarga_datos.py` primero, o pasa --imagen/--root."
        )
    return rgb[0]  # los nombres son timestamps: el primero es el mas antiguo


def main() -> int:
    parser = argparse.ArgumentParser(description="Nivel 00: la imagen como matriz")
    parser.add_argument("--imagen", help="una imagen cualquiera (png/jpg)")
    parser.add_argument("--root", help="carpeta de una secuencia TUM (contiene rgb/)")
    args = parser.parse_args()

    ruta = encontrar_imagen(args)
    salida = AQUI / "salida"
    salida.mkdir(exist_ok=True)

    # ── PASO 1 · Cargar: la imagen es un ndarray ─────────────────────────────
    # OpenCV devuelve alto x ancho x 3 en orden BGR (azul primero, herencia
    # historica de las camaras de los 90). uint8 = entero de 8 bits: cada
    # pixel de cada canal es un numero entre 0 (nada de luz) y 255 (saturado).
    bgr = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"No pude leer {ruta}")

    alto, ancho, canales = bgr.shape
    print(f"Imagen: {ruta.name}")
    print(f"  shape = {bgr.shape}  (alto={alto}, ancho={ancho}, canales={canales})")
    print(f"  dtype = {bgr.dtype}  (rango {bgr.min()}..{bgr.max()})")
    print(f"  total de numeros almacenados: {bgr.size:,}")

    # Un pixel concreto es solo indexar la matriz: [fila, columna] = (B, G, R).
    b, g, r = bgr[alto // 2, ancho // 2]
    print(f"  pixel central [fila {alto//2}, col {ancho//2}] -> B={b} G={g} R={r}")

    # ── PASO 2 · Gris A MANO ─────────────────────────────────────────────────
    #
    # ─── La matemática: el gris como combinación ponderada ───
    # El ojo no pesa igual los colores: los conos M (sensibles al verde)
    # dominan nuestra percepción de brillo. El estándar ITU-R BT.601 (el de
    # la TV analógica, que necesitaba una señal de "luminancia" compatible
    # con los televisores en blanco y negro) fijó los pesos:
    #
    #     gris = 0.299·R + 0.587·G + 0.114·B      (suman 1.0)
    #
    # Un promedio simple (1/3, 1/3, 1/3) produce un gris "plano": el cielo
    # azul sale igual de brillante que el pasto verde, cosa que tu ojo no ve
    # así. IMPORTANTE numérico: la suma se hace en float y solo al final se
    # redondea a uint8 — si operas en uint8 directamente, cada producto se
    # trunca y acumulas error (y 0.299*200 ya no cabe en un "producto uint8").
    pesos = np.array([0.114, 0.587, 0.299])           # orden B, G, R !
    gris_mano = (bgr.astype(np.float64) @ pesos)      # (H,W,3)@(3,) -> (H,W)
    gris_mano = np.clip(np.round(gris_mano), 0, 255).astype(np.uint8)

    # Verificación contra la referencia (el patrón de TODO el curso:
    # implementas tú, compruebas contra una implementación de confianza).
    gris_cv2 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dif_max = int(np.abs(gris_mano.astype(int) - gris_cv2.astype(int)).max())
    print(f"\nGris a mano vs cv2.cvtColor: diferencia maxima = {dif_max} "
          f"(<=1 es exito: solo redondeo)")

    cv2.imwrite(str(salida / "gris_a_mano.png"), gris_mano)

    # ── PASO 3 · Histograma: la radiografía de la exposición ────────────────
    # Cuenta cuantos pixeles hay de cada intensidad 0..255. Montañas a la
    # izquierda = imagen oscura; un pico en 255 = zonas quemadas (saturadas,
    # informacion PERDIDA — el sensor no puede contar mas fotones).
    hist, _ = np.histogram(gris_mano, bins=256, range=(0, 256))
    print(f"Histograma: media={gris_mano.mean():.1f}  mediana={np.median(gris_mano):.0f}  "
          f"pixeles saturados (255): {int(hist[255]):,}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.fill_between(range(256), hist, color="0.4")
        ax.set_xlim(0, 255), ax.set_xlabel("intensidad"), ax.set_ylabel("pixeles")
        ax.set_title("Histograma del gris")
        fig.savefig(salida / "histograma.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        print("[aviso] matplotlib no instalado: se omite histograma.png")

    # ── PASO 4 · Aritmética de píxeles: negativo y recorte ──────────────────
    # El negativo es restar a 255: los arrays operan ELEMENTO A ELEMENTO,
    # no hay bucles (esa es la gracia de numpy: el bucle vive en C).
    negativo = 255 - gris_mano
    cv2.imwrite(str(salida / "negativo.png"), negativo)

    # Recortar es indexar con rangos [filas, columnas]: el cuarto central.
    recorte = gris_mano[alto // 4: 3 * alto // 4, ancho // 4: 3 * ancho // 4]
    cv2.imwrite(str(salida / "recorte.png"), recorte)
    print(f"\nGuardado en {salida}: gris_a_mano.png, histograma.png, "
          f"negativo.png, recorte.png ({recorte.shape[1]}x{recorte.shape[0]})")

    print("\nListo. Ahora corre `python verificacion.py` para el examen del nivel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
