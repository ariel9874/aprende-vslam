#!/usr/bin/env python3
"""
Nivel 04 — Calibración con tablero
==================================

El flujo estándar completo, sobre las vistas sintéticas del generador
(cámara con la distorsión REAL de TUM fr1, verdad conocida):

    1. detectar esquinas (findChessboardCorners + refinado sub-pixel)
    2. calibrar (cv2.calibrateCamera) y leer el error de reproyeccion
    3. comparar con la VERDAD: intrinsecos y CAMPO de distorsion
    4. rectificar: las rectas vuelven a ser rectas (y se mide)

Uso:
    python 04_calibracion.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent
DATOS = AQUI / "data" / "tablero"

COLS, ROWS = 9, 6            # esquinas INTERNAS del tablero 10x7


def leer_gt() -> tuple[np.ndarray, np.ndarray, float]:
    lineas = [l for l in (DATOS / "gt_calibracion.txt")
              .read_text(encoding="utf-8").splitlines()
              if l.strip() and not l.startswith("#")]
    fx, fy, cx, cy = [float(v) for v in lineas[0].split()]
    dist = np.array([float(v) for v in lineas[1].split()])
    sq = float(lineas[2])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return K, dist, sq


def detectar_esquinas(rutas: list[Path]) -> tuple[list, list, tuple]:
    """findChessboardCorners + cornerSubPix en cada vista."""
    obj_todos, img_todos = [], []
    shape = None
    # Las esquinas 3D del tablero: una rejilla plana PERFECTA (por eso el
    # tablero calibra: su geometria se conoce sin error).
    objp = np.zeros((ROWS * COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2)   # en unidades de cuadro

    for ruta in rutas:
        gray = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
        shape = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, (COLS, ROWS), None)
        if not ok:
            print(f"   [aviso] sin tablero en {ruta.name}")
            continue
        # refinado sub-pixel: el detector da el pixel; la esquina REAL esta
        # entre pixeles (gradiente local). La calibracion vive de esto.
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
        obj_todos.append(objp)
        img_todos.append(corners)
    return obj_todos, img_todos, shape


def campo_de_distorsion(K: np.ndarray, dist: np.ndarray,
                        n: int = 25) -> np.ndarray:
    """Desplazamiento (px) que la distorsión aplica en una rejilla de la imagen.

    Es la comparación HONESTA entre calibraciones: k1/k2/k3 son un polinomio
    correlacionado (dos juegos distintos pueden dar casi la misma curva);
    lo que importa es cuántos píxeles mueve la lente en cada zona.
    """
    W, H = 640, 480
    uu, vv = np.meshgrid(np.linspace(20, W - 20, n), np.linspace(20, H - 20, n))
    pix = np.stack([uu.ravel(), vv.ravel()], axis=1)
    # ideal -> distorsionado con este modelo: normalizar con K, aplicar el
    # modelo (cv2.projectPoints con rvec=tvec=0 lo hace por nosotros).
    xy = (pix - [K[0, 2], K[1, 2]]) / [K[0, 0], K[1, 1]]
    X = np.hstack([xy, np.ones((len(xy), 1))]).astype(np.float64)
    proy, _ = cv2.projectPoints(X, np.zeros(3), np.zeros(3), K, dist)
    return proy.reshape(-1, 2) - pix


def main() -> int:
    rutas = sorted(DATOS.glob("vista_*.png"))
    if not rutas:
        raise SystemExit("No hay vistas. Corre `python genera_tablero.py` primero.")
    salida = AQUI / "salida"
    salida.mkdir(exist_ok=True)
    K_gt, dist_gt, sq = leer_gt()

    # ── 1 · Detectar ──────────────────────────────────────────────────────────
    print(f"1. Detectando el tablero {COLS}x{ROWS} en {len(rutas)} vistas...")
    objs, imgs, shape = detectar_esquinas(rutas)
    print(f"   detectado en {len(imgs)}/{len(rutas)}")

    vis = cv2.imread(str(rutas[0]))
    cv2.drawChessboardCorners(vis, (COLS, ROWS), imgs[0], True)
    cv2.imwrite(str(salida / "deteccion.png"), vis)

    # ── 2 · Calibrar ──────────────────────────────────────────────────────────
    #
    # ─── La matemática: calibrar es minimizar reproyección ───
    # calibrateCamera resuelve, por Levenberg-Marquardt:
    #     min  Σ_vistas Σ_esquinas ‖ detectada − proyectar(esquina_3D;
    #                                       K, dist, R_vista, t_vista) ‖²
    # Incógnitas: 4 de K + 5 de distorsión + 6 por vista (su pose). Las
    # esquinas 3D se dan en "unidades de cuadro": la escala física del
    # tablero NO afecta a K ni a dist (solo a las t_vista) — la misma
    # invariancia de escala del monocular (nivel 07), aquí inofensiva.
    rms, K_est, dist_est, rvecs, tvecs = cv2.calibrateCamera(
        objs, imgs, shape, None, None)
    dist_est = dist_est.ravel()

    print(f"\n2. Error de reproyeccion RMS: {rms:.3f} px  (<0.5 = sano)")
    print("   intrinsecos (estimado vs verdad):")
    for nombre, est, gt in [("fx", K_est[0, 0], K_gt[0, 0]),
                            ("fy", K_est[1, 1], K_gt[1, 1]),
                            ("cx", K_est[0, 2], K_gt[0, 2]),
                            ("cy", K_est[1, 2], K_gt[1, 2])]:
        print(f"   {nombre}: {est:8.2f} vs {gt:8.2f}  "
              f"({100*abs(est-gt)/gt:.2f}% de error)")
    print(f"   coeficientes estimados: {np.round(dist_est, 4)}")
    print(f"   coeficientes verdad:    {np.round(dist_gt, 4)}")

    # ── 3 · La comparación honesta: el CAMPO de distorsión ───────────────────
    d_gt = campo_de_distorsion(K_gt, dist_gt)
    d_est = campo_de_distorsion(K_est, dist_est)
    dif = np.linalg.norm(d_gt - d_est, axis=1)
    mag = np.linalg.norm(d_gt, axis=1)
    print(f"\n3. Campo de distorsion (rejilla de 625 puntos):")
    print(f"   magnitud real:  media {mag.mean():.1f} px, max {mag.max():.1f} px")
    print(f"   |estimado - verdad|: media {dif.mean():.3f} px, max {dif.max():.3f} px")
    print("   (dos juegos de k's DISTINTOS, casi el mismo campo: por eso se")
    print("    compara el campo y no los coeficientes)")

    # ── 4 · Rectificar: las rectas vuelven a ser rectas ───────────────────────
    #
    # ─── La matemática: DÓNDE se curva una recta (y dónde no) ───
    # La distorsión radial empuja cada punto A LO LARGO de su radio desde el
    # centro óptico. Dos consecuencias que la mayoría de tutoriales omiten:
    #
    #  (a) Una recta que PASA POR EL CENTRO óptico no se curva: su radio es
    #      su propia dirección, así que los puntos se deslizan sobre ella. Lo
    #      poco que se curve es distorsión TANGENCIAL (p1, p2), que rompe la
    #      simetría radial. Es una sonda limpia para separar ambos efectos.
    #  (b) La curvatura crece con la distancia al centro: en fr1 una recta
    #      que roza el borde izquierdo se comba ~2 px, mientras la MISMA
    #      lente desplaza los puntos hasta 23 px (experimento 3). Desplazar
    #      no es curvar: el ojo perdona lo primero, la geometría no.
    def residuo_recta(pts: np.ndarray) -> float:
        """Distancia RMS de los puntos a su mejor recta (ajuste por SVD).
        Cero = perfectamente colineales."""
        c = pts - pts.mean(0)
        _, _, Vt = np.linalg.svd(c)
        return float(np.sqrt((np.abs(c @ Vt[1]) ** 2).mean()))

    def por_la_lente(pix: np.ndarray, K: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Pasa píxeles IDEALES por la lente (rvec=tvec=0: solo K y dist)."""
        xy = (pix - [K[0, 2], K[1, 2]]) / [K[0, 0], K[1, 1]]
        X = np.hstack([xy, np.ones((len(xy), 1))])
        p, _ = cv2.projectPoints(X, np.zeros(3), np.zeros(3), K, d)
        return p.reshape(-1, 2)

    print("\n4. Rectas del mundo vistas por la lente (curvatura RMS respecto a"
          "\n   su mejor recta; rectificadas con TU calibracion):")
    rectas = {
        "vertical junto al borde (u=15)":
            np.stack([np.full(60, 15.0), np.linspace(10, 470, 60)], axis=1),
        "horizontal baja (v=460)":
            np.stack([np.linspace(10, 630, 60), np.full(60, 460.0)], axis=1),
        "diagonal POR EL CENTRO optico":
            np.array([K_gt[0, 2], K_gt[1, 2]]) + np.linspace(-250, 250, 60)[:, None]
            * (np.array([620.0, 460.0]) / np.linalg.norm([620.0, 460.0])),
    }
    resultados = {}
    for nombre, ideal in rectas.items():
        real = por_la_lente(ideal, K_gt, dist_gt)
        rect = cv2.undistortPoints(real.reshape(-1, 1, 2), K_est, dist_est,
                                   P=K_est).reshape(-1, 2)
        a, d = residuo_recta(real), residuo_recta(rect)
        resultados[nombre] = (a, d)
        print(f"   {nombre:32s}: {a:5.2f} px  ->  {d:5.2f} px")
    print("   La diagonal por el centro casi no se curva (es (a) de arriba):")
    print("   lo que le queda es el residuo TANGENCIAL de la lente.")

    img = cv2.imread(str(rutas[0]), cv2.IMREAD_GRAYSCALE)
    und = cv2.undistort(img, K_est, dist_est)
    panel = np.hstack([img, und])
    cv2.putText(panel, "distorsionada", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, 255, 2)
    cv2.putText(panel, "rectificada", (648, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, 255, 2)
    cv2.imwrite(str(salida / "undistort_antes_despues.png"), panel)

    print(f"\nGuardado en {salida}. Ahora corre `python verificacion.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
