#!/usr/bin/env python3
"""Genera el CORREDOR: una secuencia de ida y vuelta, con ground truth exacto.

La cámara recorre un pasillo de carteles y VUELVE al punto de partida. Al
volver, re-visita zonas que ya mapeó: es el escenario que exige cierre de
bucle (nivel 12), ahora dentro de un SLAM de verdad.

─── Por qué CARTELES DISJUNTOS y no una pared ───
La primera versión de esta escena (en el repo padre) era una pared de fondo
que se veía desde TODAS partes. Resultado: todo era co-visible con todo, y
los "cierres de bucle" disparaban a mitad de camino sin significado — el
sistema creía estar volviendo cuando sólo miraba la misma pared de lejos.
Con carteles DISJUNTOS a lo largo del recorrido, cada uno sólo es visible en
un tramo: la co-visibilidad entre el inicio y la mitad del pasillo es NULA, y
re-visitar el inicio es un bucle GENUINO. (Es la lección 15 del repo padre:
la escena de prueba es parte del experimento, no decorado.)

Salida (en data/corredor/):
    images/000000.png ...   la secuencia (200 frames, 640x480)
    calib.txt               fx fy cx cy width height
    groundtruth.txt         poses reales en formato TUM

Uso:
    python genera_datos.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent


def rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Shepperd (nivel 03)."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw, qx = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        qy, qz = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw, qx = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        qy, qz = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw, qx = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        qy, qz = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw, qx = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        qy, qz = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return np.array([qx, qy, qz, qw])


def guardar_tum(items, path: Path) -> None:
    lineas = []
    for t, T in items:
        q = rotation_to_quaternion(T[:3, :3])
        tx, ty, tz = T[:3, 3]
        lineas.append(f"{t:.6f} {tx:.6f} {ty:.6f} {tz:.6f} "
                      f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
    path.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def make_texture(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Ruido multi-escala: rico en esquinas a varias frecuencias (nivel 05)."""
    tex = np.zeros((height, width), dtype=np.float64)
    for cell, weight in [(160, 1.0), (60, 0.8), (24, 0.6), (8, 0.4)]:
        octave = rng.uniform(0, 1, (height // cell + 2, width // cell + 2))
        tex += weight * cv2.resize(octave, (width, height), interpolation=cv2.INTER_CUBIC)
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    return (30 + tex * 195).astype(np.uint8)


@dataclass
class Cartel:
    """Un cartel plano frontal en (cx, cy, depth)."""
    center_x: float
    center_y: float
    depth: float
    half_x: float
    half_y: float
    texture: np.ndarray

    def esquinas(self) -> np.ndarray:
        cx, cy, z = self.center_x, self.center_y, self.depth
        hx, hy = self.half_x, self.half_y
        return np.array([[cx - hx, cy - hy, z], [cx + hx, cy - hy, z],
                         [cx + hx, cy + hy, z], [cx - hx, cy + hy, z]])


def render(canvas, cartel: Cartel, K, R_w_c, C) -> None:
    """Homografía exacta textura -> imagen (la del nivel 07/08/09)."""
    esq_cam = (R_w_c.T @ (cartel.esquinas() - C).T).T
    if np.any(esq_cam[:, 2] < 0.2):
        return
    Z = esq_cam[:, 2]
    img_pts = np.stack([K[0, 0] * esq_cam[:, 0] / Z + K[0, 2],
                        K[1, 1] * esq_cam[:, 1] / Z + K[1, 2]], axis=1).astype(np.float32)
    th, tw = cartel.texture.shape
    tex_pts = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(tex_pts, img_pts)
    size = (canvas.shape[1], canvas.shape[0])
    warp = cv2.warpPerspective(cartel.texture, H, size, flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((th, tw), 255, np.uint8), H, size,
                               flags=cv2.INTER_NEAREST)
    canvas[mask > 127] = warp[mask > 127]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", default=str(AQUI / "data" / "corredor"))
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    W, H = 640, 480
    fx = fy = 450.0
    K = np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]])

    # El corredor: carteles disjuntos a lo largo del recorrido (ver docstring).
    carteles = []
    for i, bx in enumerate(np.arange(-2.0, 11.5, 1.9)):
        depth = [3.8, 5.2, 4.4, 6.0][i % 4]
        carteles.append(Cartel(float(bx), 0.0, depth, 1.4, 2.6,
                               make_texture(520, 900, rng)))

    out = Path(args.output)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    gt = []
    for k in range(args.frames):
        # IDA Y VUELTA: x(0) = x(N-1) = 0. La camara siempre mira al frente
        # (el mundo sintetico solo existe ahi), pero el punto de partida se
        # RE-VISITA al final: eso es lo que el cierre de bucle debe detectar.
        fase = np.pi * k / (args.frames - 1)
        C = np.array([7.0 * np.sin(fase), 0.0, 0.0])
        R_w_c = rot_y(0.0)
        T = np.eye(4)
        T[:3, :3] = R_w_c
        T[:3, 3] = C
        gt.append((k / 30.0, T))

        canvas = np.full((H, W), 15, np.uint8)
        for c in carteles:
            render(canvas, c, K, R_w_c, C)
        cv2.imwrite(str(img_dir / f"{k:06d}.png"), canvas)

    (out / "calib.txt").write_text(
        f"# fx fy cx cy width height\n{fx} {fy} {W/2} {H/2} {W} {H}\n",
        encoding="utf-8")
    guardar_tum(gt, out / "groundtruth.txt")
    print(f"OK: {args.frames} frames del corredor en {img_dir}")
    print(f"    la camara va de x=0 a x=7 y VUELVE a x=0 (bucle genuino)")
    print(f"    ground truth (TUM): {out / 'groundtruth.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
