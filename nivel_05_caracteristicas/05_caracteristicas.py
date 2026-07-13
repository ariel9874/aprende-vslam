#!/usr/bin/env python3
"""
Nivel 05 — Características
==========================

Tres experimentos, de la teoría a la medición:

    1. Harris A MANO en numpy, verificado contra cv2.cornerHarris
    2. Comparador de detectores: GFTT vs ORB vs SIFT (conteos y tiempos)
    3. Invarianza MEDIDA: repetibilidad bajo rotación de 30 grados y escala 0.5x

Uso:
    python 05_caracteristicas.py                 # primer frame del dataset
    python 05_caracteristicas.py --root <secuencia_TUM>
    python 05_caracteristicas.py --imagen foto.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent
DATASET_DEFAULT = AQUI / "data" / "rgbd_dataset_freiburg1_xyz"


def encontrar_imagen(args) -> Path:
    if args.imagen:
        return Path(args.imagen)
    root = Path(args.root) if args.root else DATASET_DEFAULT
    rgb = sorted((root / "rgb").glob("*.png"))
    if not rgb:
        raise SystemExit(f"No hay imagenes en {root / 'rgb'}. "
                         "Corre `python descarga_datos.py` o pasa --imagen/--root.")
    return rgb[0]


# ───────────────────────── 1 · Harris a mano ─────────────────────────────────

def harris_a_mano(gray: np.ndarray, block: int = 3, ksize: int = 3,
                  k: float = 0.04) -> np.ndarray:
    """Mapa de respuesta de Harris, calculado a mano con numpy + Sobel.

    ─── La matemática: el tensor de estructura ───
    ¿Cuánto cambia el parche alrededor de (x, y) si lo desplazo (u, v)?
    Con Taylor a primer orden, I(x+u, y+v) ≈ I + Ix·u + Iy·v, y la suma de
    diferencias cuadráticas sobre la ventana W queda como forma cuadrática:

        E(u, v) ≈ [u v] · M · [u v]ᵀ ,   M = Σ_W [Ix²   Ix·Iy]
                                                 [Ix·Iy   Iy²]

    M (el TENSOR DE ESTRUCTURA) resume la geometría local. Sus autovalores
    λ1 ≥ λ2 miden el contraste en las dos direcciones principales:

        zona plana:  λ1 ≈ λ2 ≈ 0     (moverse no cambia nada)
        borde:       λ1 ≫ λ2 ≈ 0     (deslizarse A LO LARGO no cambia nada:
                                      el problema de APERTURA)
        esquina:     λ1, λ2 grandes  (cualquier movimiento se nota)

    Calcular autovalores por píxel es caro; Harris (1988) propuso el proxy

        R = det(M) − k·tr(M)²  =  λ1·λ2 − k·(λ1+λ2)²

    que es grande solo si AMBOS autovalores lo son (k ≈ 0.04, empírico).
    Shi-Tomasi (el "GFTT" de OpenCV) usa directamente min(λ1, λ2) — mismo
    espíritu, umbral más interpretable.
    """
    g = gray.astype(np.float32)
    # Gradientes por Sobel (mismo operador que usa cv2.cornerHarris).
    Ix = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=ksize)
    # La suma sobre la ventana W es un filtro de caja SIN normalizar
    # (block x block), igual que blockSize en cv2.cornerHarris.
    Ixx = cv2.boxFilter(Ix * Ix, -1, (block, block), normalize=False)
    Iyy = cv2.boxFilter(Iy * Iy, -1, (block, block), normalize=False)
    Ixy = cv2.boxFilter(Ix * Iy, -1, (block, block), normalize=False)

    det = Ixx * Iyy - Ixy * Ixy
    tr = Ixx + Iyy
    return det - k * tr * tr


def top_esquinas(R: np.ndarray, n: int = 50, radio_nms: int = 5) -> np.ndarray:
    """Las n esquinas más fuertes con supresión de no-máximos.

    Un máximo local es un píxel que iguala al máximo de su vecindario
    (dilatación morfológica): sin esto, las n mejores respuestas serían n
    píxeles pegados de LA MISMA esquina.
    """
    vecindario = cv2.dilate(R, np.ones((radio_nms, radio_nms), np.uint8))
    es_maximo = (R == vecindario) & (R > 0)
    ys, xs = np.nonzero(es_maximo)
    orden = np.argsort(R[ys, xs])[::-1][:n]
    return np.stack([xs[orden], ys[orden]], axis=1)  # (n, 2) en (x, y)


def coincidencia(pts_a: np.ndarray, pts_b: np.ndarray, radio: float = 3.0) -> float:
    """Fracción de puntos de A que tienen un punto de B a menos de `radio` px."""
    if len(pts_a) == 0 or len(pts_b) == 0:
        return 0.0
    d = np.linalg.norm(pts_a[:, None, :] - pts_b[None, :, :], axis=2)
    return float((d.min(axis=1) < radio).mean())


# ─────────────────── 3 · repetibilidad (invarianza medida) ───────────────────

def repetibilidad(gray: np.ndarray, detectar, M: np.ndarray,
                  radio: float = 2.0) -> float:
    """% de keypoints re-detectados tras transformar la imagen con M (2x3 afín).

    Detecta en la imagen original y en la transformada; mapea los segundos de
    vuelta con la inversa de M y cuenta cuántos originales tienen un
    re-detectado a < radio px. Solo cuentan los originales cuya posicion
    transformada cae DENTRO de la imagen (la rotación recorta las esquinas).
    """
    h, w = gray.shape
    pts0 = detectar(gray)
    warped = cv2.warpAffine(gray, M, (w, h))
    pts1 = detectar(warped)
    if len(pts0) == 0 or len(pts1) == 0:
        return 0.0

    # ¿Qué originales siguen visibles tras la transformación?
    pts0_h = np.hstack([pts0, np.ones((len(pts0), 1))])
    pts0_en_warp = pts0_h @ M.T                       # (n, 2)
    visibles = ((pts0_en_warp[:, 0] >= 0) & (pts0_en_warp[:, 0] < w) &
                (pts0_en_warp[:, 1] >= 0) & (pts0_en_warp[:, 1] < h))
    if visibles.sum() == 0:
        return 0.0

    # Los detectados en la imagen transformada, de vuelta al marco original.
    M_inv = cv2.invertAffineTransform(M)
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts1_orig = pts1_h @ M_inv.T

    d = np.linalg.norm(pts0[visibles][:, None, :] - pts1_orig[None, :, :], axis=2)
    return float((d.min(axis=1) < radio).mean())


def como_detector(objeto_cv2):
    """Adapta un detector de OpenCV a 'imagen -> array (n,2) de posiciones'."""
    def detectar(gray):
        kps = objeto_cv2.detect(gray, None)
        return np.array([kp.pt for kp in kps]) if kps else np.zeros((0, 2))
    return detectar


# ─────────────────────────────── programa ────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Nivel 05: caracteristicas")
    parser.add_argument("--imagen")
    parser.add_argument("--root")
    args = parser.parse_args()

    ruta = encontrar_imagen(args)
    gray = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
    salida = AQUI / "salida"
    salida.mkdir(exist_ok=True)
    print(f"Imagen: {ruta.name}  ({gray.shape[1]}x{gray.shape[0]})\n")

    # ── 1 · Harris a mano vs cv2 (el patron del curso: implementar y verificar)
    R_mano = harris_a_mano(gray)
    R_cv2 = cv2.cornerHarris(gray.astype(np.float32), blockSize=3, ksize=3, k=0.04)
    top_mano = top_esquinas(R_mano, 50)
    top_cv2 = top_esquinas(R_cv2, 50)
    coinc = coincidencia(top_mano, top_cv2)
    print(f"1. Harris a mano vs cv2.cornerHarris (top-50 esquinas, radio 3 px):")
    print(f"   coincidencia = {100*coinc:.0f}%  (>=80% es exito)")

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x, y in top_cv2:
        cv2.circle(vis, (int(x), int(y)), 6, (0, 165, 255), 1)   # cv2: naranja
    for x, y in top_mano:
        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)    # propio: verde
    cv2.imwrite(str(salida / "harris_propio_vs_cv2.png"), vis)

    # ── 2 · Comparador de detectores ──────────────────────────────────────────
    detectores = {
        "gftt": cv2.GFTTDetector_create(maxCorners=1000, qualityLevel=0.01),
        "orb": cv2.ORB_create(nfeatures=2000),
        "sift": cv2.SIFT_create(nfeatures=2000),
    }
    print("\n2. Detectores sobre el mismo frame:")
    print(f"   {'detector':8s} {'kps':>6s} {'ms':>7s}")
    paneles = []
    for nombre, det in detectores.items():
        t0 = time.perf_counter()
        kps = det.detect(gray, None)
        ms = 1000 * (time.perf_counter() - t0)
        print(f"   {nombre:8s} {len(kps):6d} {ms:7.1f}")
        panel = cv2.drawKeypoints(gray, kps, None, color=(0, 255, 0))
        cv2.putText(panel, f"{nombre}: {len(kps)}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        paneles.append(panel)
    cv2.imwrite(str(salida / "detectores.png"), np.hstack(paneles))

    # ── 3 · Invarianza medida (repetibilidad) ─────────────────────────────────
    h, w = gray.shape
    M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), 30.0, 1.0)   # rotar 30 grados
    M_esc = cv2.getRotationMatrix2D((w / 2, h / 2), 0.0, 0.5)    # escalar 0.5x
    print("\n3. Repetibilidad (% de kps re-detectados a <2 px):")
    print(f"   {'detector':8s} {'rot 30':>8s} {'esc 0.5':>8s}")
    for nombre, det in detectores.items():
        f = como_detector(det)
        r_rot = repetibilidad(gray, f, M_rot)
        r_esc = repetibilidad(gray, f, M_esc)
        print(f"   {nombre:8s} {100*r_rot:7.0f}% {100*r_esc:7.0f}%")
    print("   (Medido, no asumido: ORB — esquinas FAST en piramide — es el que")
    print("    mejor sobrevive a media resolucion. Y ojo con SIFT: 'invariante")
    print("    a escala' significa que EMPAREJA entre escalas, no que re-detecte")
    print("    los mismos puntos — sus blobs mas finos ya no existen a 0.5x,")
    print("    la informacion desaparecio con los pixeles. Ver EJERCICIOS 3.)")

    print(f"\nGuardado en {salida}: harris_propio_vs_cv2.png, detectores.png")
    print("Ahora corre `python verificacion.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
