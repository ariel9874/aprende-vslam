#!/usr/bin/env python3
"""
Nivel 02 — La cámara pinhole
============================

Tres experimentos:

    1. proyectar puntos conocidos con la K real de TUM fr1 (a mano)
    2. el renderizador alambrico: un cubo girando, en numpy puro
    3. el campo de vision es la focal: mismo cubo con fx/2, fx, 2fx

Uso:
    python 02_pinhole.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent

# La camara REAL de TUM freiburg1 (la calibraras tu en el nivel 04 y la
# usaras en el SLAM de los niveles 8+). Imagen de 640x480.
FX, FY, CX, CY = 517.306408, 516.469215, 318.643040, 255.313989
W, H = 640, 480


class PinholeCamera:
    """La cámara ideal como función: 3D (marco de la cámara) -> píxeles.

    ─── La matemática: proyección por triángulos semejantes ───
    Con el centro óptico en el origen y el plano imagen a distancia focal f,
    el rayo del punto (X, Y, Z) cruza el plano en (f·X/Z, f·Y/Z): triángulos
    semejantes, nada más. En píxeles (f en unidades de fotositos, origen en
    la esquina superior izquierda):

        u = fx·X/Z + cx        v = fy·Y/Z + cy

    En forma matricial con coordenadas homogéneas, [u, v, 1]ᵀ ~ K·X/Z. La
    división por Z es NO lineal (por eso las homogéneas: vuelven lineal lo
    proyectivo) y DESTRUYE la profundidad: todos los puntos del rayo
    λ·(X, Y, Z) caen en el mismo píxel. Una imagen es un haz de rayos.

    Ejes convención OpenCV: +Z delante, +Y ABAJO, +X derecha (por eso cy
    crece hacia abajo en la imagen). Todo el curso usa estos ejes.
    """

    def __init__(self, fx: float, fy: float, cx: float, cy: float) -> None:
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def project(self, X: np.ndarray) -> np.ndarray:
        """(N, 3) en el marco de la cámara -> (N, 2) en píxeles."""
        X = np.atleast_2d(X).astype(np.float64)
        u = self.fx * X[:, 0] / X[:, 2] + self.cx
        v = self.fy * X[:, 1] / X[:, 2] + self.cy
        return np.stack([u, v], axis=1)

    def backproject(self, uv: np.ndarray, z: np.ndarray) -> np.ndarray:
        """El rayo del píxel, cortado a profundidad z: la INVERSA si conoces Z.

        x = (u - cx)/fx es el rayo unitario-en-Z; multiplicar por z da el
        punto. Sin z solo hay rayo: la ambigüedad fundamental del monocular.
        """
        uv = np.atleast_2d(uv).astype(np.float64)
        z = np.atleast_1d(z).astype(np.float64)
        x = (uv[:, 0] - self.cx) / self.fx * z
        y = (uv[:, 1] - self.cy) / self.fy * z
        return np.stack([x, y, z], axis=1)


# ─────────────────────── el cubo alámbrico ───────────────────────────────────

# 8 vertices de un cubo unitario centrado en el origen, y sus 12 aristas.
VERTICES = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
                    dtype=np.float64) * 0.5
ARISTAS = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7),
           (0, 4), (1, 5), (2, 6), (3, 7)]


def rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def render_cubo(cam: PinholeCamera, theta: float, distancia: float = 3.0,
                color=(0, 255, 0)) -> np.ndarray:
    """Renderiza el cubo girado theta, a `distancia` frente a la cámara.

    OJO: aquí quien se mueve es el CUBO (rotación de modelo); la cámara está
    fija en el origen. Mover la CÁMARA por el mundo — extrínsecos, poses —
    es exactamente el nivel 03. Un renderizador es: transformar vértices al
    marco de la cámara, proyectar con K, unir con líneas. No hay más.
    """
    R = rot_y(theta) @ rot_x(0.4)                  # inclinado para verlo 3D
    X_cam = VERTICES @ R.T + np.array([0.0, 0.0, distancia])
    uv = cam.project(X_cam).astype(int)

    img = np.zeros((H, W, 3), np.uint8)
    for i, j in ARISTAS:
        cv2.line(img, tuple(uv[i]), tuple(uv[j]), color, 2, cv2.LINE_AA)
    for u, v in uv:
        cv2.circle(img, (u, v), 4, (0, 128, 255), -1)
    return img


def main() -> int:
    salida = AQUI / "salida"
    salida.mkdir(exist_ok=True)
    cam = PinholeCamera(FX, FY, CX, CY)

    # ── 1 · Proyecciones a mano, verificables con la cabeza ──────────────────
    print("1. Proyecciones con la K de TUM fr1 (fx=517.3, cx=318.6...):")
    ejemplos = np.array([[0.0, 0.0, 2.0],     # sobre el eje optico
                         [1.0, 0.0, 2.0],     # 1 m a la derecha, a 2 m
                         [1.0, 0.0, 4.0]])    # lo mismo pero al doble de Z
    for X, uv in zip(ejemplos, cam.project(ejemplos)):
        print(f"   {X} m  ->  ({uv[0]:6.1f}, {uv[1]:6.1f}) px")
    print("   El eje optico cae en (cx, cy). Doblar Z parte a la mitad el")
    print("   desplazamiento desde el centro: u - cx = fx*X/Z.")

    # Ida y vuelta: si conoces Z, la proyeccion se invierte EXACTA.
    uv = cam.project(ejemplos)
    rec = cam.backproject(uv, ejemplos[:, 2])
    print(f"   ida y vuelta (con Z conocida): error max "
          f"{np.abs(rec - ejemplos).max():.2e} m")

    # Lo que se pierde: dos puntos del MISMO rayo, mismo pixel.
    rayo = np.array([[0.3, 0.2, 1.5], [0.6, 0.4, 3.0]])   # el doble = mismo rayo
    duv = np.abs(np.diff(cam.project(rayo), axis=0)).max()
    print(f"   dos puntos del mismo rayo (Z=1.5 y Z=3.0): mismos pixeles "
          f"(dif {duv:.2e} px)\n   -> la profundidad NO es observable en un pixel")

    # ── 2 · El cubo girando (mosaico + video) ─────────────────────────────────
    n_frames = 48
    frames = [render_cubo(cam, 2 * np.pi * k / n_frames) for k in range(n_frames)]
    mosaico = np.vstack([np.hstack(frames[0:24:6]), np.hstack(frames[24:48:6])])
    cv2.imwrite(str(salida / "cubo_giro.png"), mosaico)

    vw = cv2.VideoWriter(str(salida / "cubo.avi"),
                         cv2.VideoWriter_fourcc(*"MJPG"), 24, (W, H))
    escribio_video = vw.isOpened()
    if escribio_video:
        for f in frames:
            vw.write(f)
        vw.release()
    print(f"\n2. Cubo alambrico: {n_frames} frames renderizados en numpy puro.")
    print(f"   mosaico: {salida / 'cubo_giro.png'}"
          + (f"\n   video:   {salida / 'cubo.avi'}" if escribio_video else ""))

    # ── 3 · La focal ES el campo de vision ────────────────────────────────────
    print("\n3. Mismo cubo, tres focales (el 'zoom' es multiplicar fx):")
    paneles = []
    anchos = {}
    for factor in (0.5, 1.0, 2.0):
        c = PinholeCamera(FX * factor, FY * factor, CX, CY)
        img = render_cubo(c, 0.6)
        cv2.putText(img, f"fx = {FX*factor:.0f}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        paneles.append(img)
        # ancho proyectado del cubo (para medir la proporcionalidad)
        R = rot_y(0.6) @ rot_x(0.4)
        uv = c.project(VERTICES @ R.T + np.array([0, 0, 3.0]))
        anchos[factor] = float(uv[:, 0].max() - uv[:, 0].min())
        print(f"   fx x{factor:.1f}: cubo de {anchos[factor]:6.1f} px de ancho")
    print(f"   ratio x2/x0.5 = {anchos[2.0]/anchos[0.5]:.3f} (teoria: 4.000 - "
          "el tamano en pixeles es LINEAL en fx)")
    cv2.imwrite(str(salida / "fov_vs_fx.png"), np.hstack(paneles))

    print(f"\nGuardado en {salida}. Ahora corre `python verificacion.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
