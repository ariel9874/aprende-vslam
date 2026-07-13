#!/usr/bin/env python3
"""Genera vistas sintéticas de un tablero de ajedrez con una cámara que
tiene la K y la DISTORSIÓN reales de TUM freiburg1.

Como la cámara es simulada, la verdad se conoce exacta: el nivel calibra
sobre estas imágenes y se compara contra ella. El flujo de render:

    tablero (textura) --homografía--> imagen pinhole IDEAL
                      --remap con la inversa de la distorsión--> imagen REAL

Salida (en data/tablero/):
    vista_00.png ... vista_13.png
    gt_calibracion.txt      la verdad (K y distorsión) — no mires antes ;)

Uso:
    python genera_tablero.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent

# La verdad: la camara de TUM freiburg1 (intrinsecos + Brown-Conrady).
FX, FY, CX, CY = 517.306408, 516.469215, 318.643040, 255.313989
DIST = np.array([0.262383, -0.953104, -0.005358, 0.002628, 1.163314])
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
W, H = 640, 480

# El tablero: 10x7 cuadros de 3 cm -> 9x6 esquinas INTERNAS (el estandar).
COLS_SQ, ROWS_SQ, SQ = 10, 7, 0.03


def distorsionar(xy_norm: np.ndarray) -> np.ndarray:
    """Aplica Brown-Conrady A MANO a puntos en coordenadas normalizadas.

    ─── La matemática: el modelo Brown-Conrady ───
    En coordenadas normalizadas (x, y) = ((u−cx)/fx, (v−cy)/fy), con
    r² = x² + y², la lente desplaza cada punto:

        x' = x·(1 + k1·r² + k2·r⁴ + k3·r⁶) + 2·p1·x·y + p2·(r² + 2x²)
        y' = y·(1 + k1·r² + k2·r⁴ + k3·r⁶) + p1·(r² + 2y²) + 2·p2·x·y

    El término radial (k1, k2, k3) es un polinomio PAR en r: la distorsión
    solo depende de la distancia al eje óptico (una lente es redonda). El
    tangencial (p1, p2) modela el des-alineado lente↔sensor. OJO: el modelo
    va de ideal→distorsionado; INVERTIRLO no tiene forma cerrada (se hace
    por iteración — cv2.undistortPoints la hace por nosotros).
    """
    k1, k2, p1, p2, k3 = DIST
    x, y = xy_norm[:, 0], xy_norm[:, 1]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([xd, yd], axis=1)


def proyectar_con_distorsion(X_cam: np.ndarray) -> np.ndarray:
    """Pinhole (nivel 02) + distorsión a mano: el modelo COMPLETO de cámara."""
    xy = X_cam[:, :2] / X_cam[:, 2:3]
    xyd = distorsionar(xy)
    return np.stack([FX * xyd[:, 0] + CX, FY * xyd[:, 1] + CY], axis=1)


def textura_tablero(px_por_cuadro: int = 60) -> np.ndarray:
    """El tablero con un margen blanco de un cuadro (lo exige el detector)."""
    h = (ROWS_SQ + 2) * px_por_cuadro
    w = (COLS_SQ + 2) * px_por_cuadro
    tex = np.full((h, w), 255, np.uint8)
    for i in range(ROWS_SQ):
        for j in range(COLS_SQ):
            if (i + j) % 2 == 0:
                y0 = (i + 1) * px_por_cuadro
                x0 = (j + 1) * px_por_cuadro
                tex[y0:y0 + px_por_cuadro, x0:x0 + px_por_cuadro] = 20
    return tex


def esquinas_texture_y_board() -> tuple[np.ndarray, np.ndarray]:
    """Las 4 esquinas EXTERIORES del quad de la textura, en px de textura y
    en metros del plano del tablero (incluyendo el margen de 1 cuadro)."""
    px = 60
    tex_pts = np.array([[0, 0], [(COLS_SQ + 2) * px, 0],
                        [(COLS_SQ + 2) * px, (ROWS_SQ + 2) * px],
                        [0, (ROWS_SQ + 2) * px]], np.float64)
    board_pts = np.array([[-SQ, -SQ, 0], [(COLS_SQ + 1) * SQ, -SQ, 0],
                          [(COLS_SQ + 1) * SQ, (ROWS_SQ + 1) * SQ, 0],
                          [-SQ, (ROWS_SQ + 1) * SQ, 0]], np.float64)
    return tex_pts, board_pts


def mapas_de_distorsion() -> tuple[np.ndarray, np.ndarray]:
    """Mapas para remap: imagen_real(ud, vd) = imagen_ideal(undistort(ud, vd)).

    La camara fisica DISTORSIONA; para sintetizar su imagen desde el render
    ideal hay que saber, para cada pixel real, de que pixel ideal proviene:
    eso es UNdistorsionar el pixel, que cv2.undistortPoints resuelve por
    iteracion (P=K lo devuelve en pixeles).
    """
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    pts = np.stack([uu.ravel(), vv.ravel()], axis=1).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, DIST, P=K).reshape(H, W, 2)
    return und[..., 0].astype(np.float32), und[..., 1].astype(np.float32)


def pose_tablero(rng: np.random.Generator) -> np.ndarray:
    """Una pose T_cam_board aleatoria y razonable (inclinada, a 0.45-0.85 m)."""
    ax, ay = rng.uniform(-0.45, 0.45, 2)              # ~ +-25 grados
    ca, sa = np.cos(ax), np.sin(ax)
    cb, sb = np.cos(ay), np.sin(ay)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    T = np.eye(4)
    T[:3, :3] = Ry @ Rx
    # centrar el tablero (su centro fisico) delante de la camara, con offsets
    centro_board = np.array([COLS_SQ * SQ / 2, ROWS_SQ * SQ / 2, 0.0])
    z = rng.uniform(0.45, 0.85)
    offset = np.array([rng.uniform(-0.16, 0.16), rng.uniform(-0.12, 0.12), z])
    T[:3, 3] = offset - T[:3, :3] @ centro_board
    return T


def main() -> int:
    out = AQUI / "data" / "tablero"
    out.mkdir(parents=True, exist_ok=True)

    tex = textura_tablero()
    tex_pts, board_pts = esquinas_texture_y_board()
    map_x, map_y = mapas_de_distorsion()
    rng = np.random.default_rng(4)

    # Las esquinas internas en el plano del tablero (para validar la fisica).
    objp = np.array([[(j + 1) * SQ, (i + 1) * SQ, 0.0]
                     for i in range(ROWS_SQ - 1) for j in range(COLS_SQ - 1)])

    n_ok = 0
    while n_ok < 14:
        T = pose_tablero(rng)
        # ¿el tablero completo (con margen) cae dentro de la imagen REAL?
        quad_cam = (board_pts @ T[:3, :3].T) + T[:3, 3]
        if np.any(quad_cam[:, 2] < 0.2):
            continue
        quad_px = proyectar_con_distorsion(quad_cam)
        if (quad_px.min() < 8 or quad_px[:, 0].max() > W - 8
                or quad_px[:, 1].max() > H - 8):
            continue

        # render ideal (homografia textura -> imagen pinhole SIN distorsion)
        quad_ideal = np.stack([FX * quad_cam[:, 0] / quad_cam[:, 2] + CX,
                               FY * quad_cam[:, 1] / quad_cam[:, 2] + CY], axis=1)
        H_mat = cv2.getPerspectiveTransform(tex_pts.astype(np.float32),
                                            quad_ideal.astype(np.float32))
        ideal = cv2.warpPerspective(tex, H_mat, (W, H), borderValue=96,
                                    borderMode=cv2.BORDER_CONSTANT)
        # ... y la lente: remap con la inversa de la distorsion
        real = cv2.remap(ideal, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=96)

        cv2.imwrite(str(out / f"vista_{n_ok:02d}.png"), real)
        n_ok += 1

    (out / "gt_calibracion.txt").write_text(
        "# La VERDAD de la camara simulada (K y distorsion de TUM freiburg1).\n"
        "# fx fy cx cy / k1 k2 p1 p2 k3 / cuadro_m\n"
        f"{FX} {FY} {CX} {CY}\n"
        f"{DIST[0]} {DIST[1]} {DIST[2]} {DIST[3]} {DIST[4]}\n"
        f"{SQ}\n", encoding="utf-8")

    print(f"OK: 14 vistas del tablero en {out}")
    print(f"    verdad en {out / 'gt_calibracion.txt'} (no mires antes de calibrar)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
