#!/usr/bin/env python3
"""
Nivel 07 — Geometría epipolar
=============================

Recupera el movimiento de la cámara entre dos vistas y lo verifica contra
ground truth EXACTO (por eso este nivel corre sobre la secuencia sintética):

    1. matches -> matriz esencial (RANSAC) -> recoverPose -> R, t
       comparados contra GT: error de rotacion y de direccion de t
    2. lineas epipolares dibujadas sobre ambas vistas
    3. LA TRAMPA de recoverPose (leccion 1 del repo padre): con paso pequeño
       los inliers colapsan si no se sube distanceThresh

Uso:
    python 07_geometria_epipolar.py [--par 0 6]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent
DATOS = AQUI / "data" / "secuencia"

RATIO = 0.75
CHEIRALITY_DIST_THRESH = 2000.0


# ─────────────────────────── carga y utilidades ──────────────────────────────

def cargar_frame(idx: int) -> np.ndarray:
    ruta = DATOS / "images" / f"{idx:06d}.png"
    if not ruta.exists():
        raise SystemExit(f"No existe {ruta}. Corre `python genera_datos.py` primero.")
    return cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)


def leer_calibracion() -> np.ndarray:
    for line in (DATOS / "calib.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            fx, fy, cx, cy = [float(v) for v in line.split()[:4]]
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    raise SystemExit("calib.txt vacio")


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """Cuaternión (qx, qy, qz, qw) -> matriz de rotación 3x3.

    ─── La matemática ───
    Es la fórmula de Rodrigues escrita en términos del cuaternión: con
    q = (x, y, z, w), la matriz sale de R = I + 2w·[v]ₓ + 2·[v]ₓ² (v = xyz).
    Desarrollada término a término da la matriz de abajo — la inversa exacta
    de la conversión de Shepperd que usa genera_datos.py.
    """
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def leer_gt() -> list[np.ndarray]:
    """Lee groundtruth.txt (formato TUM) como lista de T_w_c (4x4)."""
    poses = []
    for line in (DATOS / "groundtruth.txt").read_text(encoding="utf-8").splitlines():
        v = [float(x) for x in line.split()]
        T = np.eye(4)
        T[:3, :3] = quaternion_to_rotation(np.array(v[4:8]))
        T[:3, 3] = v[1:4]
        poses.append(T)
    return poses


def invert_se3(T: np.ndarray) -> np.ndarray:
    """T⁻¹ = [[Rᵀ, −Rᵀ·t], [0, 1]] (nivel 03: la inversa cerrada de SE(3))."""
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def angulo_entre_rotaciones(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Ángulo (grados) de la rotación residual Raᵀ·Rb.

    ─── La matemática ───
    Si Ra y Rb fueran iguales, Raᵀ·Rb = I. Lo que se desvíe de I es una
    rotación de cierto ángulo θ — el ERROR angular — que se lee de la traza
    (identidad de Rodrigues): tr(R) = 1 + 2·cos(θ). Es la métrica estándar
    de error rotacional: un solo número, invariante al marco de referencia.
    """
    cos_t = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))


def angulo_entre_vectores(a: np.ndarray, b: np.ndarray) -> float:
    """Ángulo (grados) entre dos direcciones (la escala no cuenta)."""
    cos_t = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))


# ─────────────────────────── el estimador ────────────────────────────────────

def emparejar(gray_a: np.ndarray, gray_b: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
    """ORB + ratio test (niveles 05-06). Devuelve (pts_a, pts_b) en píxeles."""
    orb = cv2.ORB_create(nfeatures=2000)
    kps_a, desc_a = orb.detectAndCompute(gray_a, None)
    kps_b, desc_b = orb.detectAndCompute(gray_b, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    pares = bf.knnMatch(desc_a, desc_b, k=2)
    buenos = [m for m, n in pares if m.distance < RATIO * n.distance]
    pts_a = np.float64([kps_a[m.queryIdx].pt for m in buenos])
    pts_b = np.float64([kps_b[m.trainIdx].pt for m in buenos])
    return pts_a, pts_b


def estimar_pose(pts_a: np.ndarray, pts_b: np.ndarray, K: np.ndarray,
                 usar_dist_thresh: bool = True) -> dict:
    """E por RANSAC + recoverPose. Devuelve R, t (T_b<-a), inliers y E.

    ─── La matemática: la restricción epipolar ───
    Sea X un punto 3D visto por ambas cámaras, relacionadas por (R, t):
    X_b = R·X_a + t. Con rayos normalizados x̂ = K⁻¹·[u, v, 1]ᵀ, los vectores
    x̂_b, R·x̂_a y t son COPLANARES (los dos rayos y el baseline forman el
    plano epipolar que contiene a X y a ambos centros). Coplanaridad =
    triple producto nulo:

        x̂_b · (t × R·x̂_a) = 0   ⇔   x̂_bᵀ·[t]ₓ·R·x̂_a = 0 ,   E ≜ [t]ₓ·R

    E tiene 5 gdl (3 de R + 2 de la direccion de t: la ecuacion es homogenea,
    la escala no cuenta) → bastan 5 correspondencias (solver de Nister).
    RANSAC vota contra los outliers: si w es la fraccion de inliers, una
    muestra de 5 sale limpia con probabilidad w⁵ y bastan
    N = log(1−p)/log(1−w⁵) iteraciones (w=0.5, p=0.999 → N≈218).

    ─── La matemática: de E a (R, t) y la TRAMPA ───
    La SVD E = U·diag(1,1,0)·Vᵀ da CUATRO factorizaciones (dos rotaciones ×
    dos signos de t). Solo una deja los puntos triangulados DELANTE de ambas
    camaras: el test de QUIRALIDAD de recoverPose. Y ||t|| = 1 por convencion
    (E es homogenea en t: monocular no mide escala).

    TRAMPA (medida en el repo padre, leccion 1): la sobrecarga basica de
    recoverPose solo acepta inliers triangulados a < 50 unidades — y como
    ||t||=1, son MULTIPLOS DEL BASELINE. Si la escena esta lejos respecto al
    paso (profundidad/baseline > 50), TODO parece "en el infinito" y los
    inliers colapsan con geometria perfecta. distanceThresh sube el umbral.
    """
    E, mask = cv2.findEssentialMat(pts_a, pts_b, K, method=cv2.RANSAC,
                                   prob=0.999, threshold=1.0)
    if usar_dist_thresh:
        n_inl, R, t, mask_pose, _ = cv2.recoverPose(
            E, pts_a, pts_b, K, distanceThresh=CHEIRALITY_DIST_THRESH, mask=mask)
    else:
        n_inl, R, t, mask_pose = cv2.recoverPose(E, pts_a, pts_b, K, mask=mask)
    return {"E": E, "R": R, "t": t.ravel(), "n_inliers": int(n_inl),
            "mask": mask_pose.ravel().astype(bool),
            "n_ransac": int(mask.sum())}


# ───────────────────────── líneas epipolares ─────────────────────────────────

def dibujar_epipolares(gray_a, gray_b, pts_a, pts_b, E, K, path: Path,
                       n: int = 12) -> None:
    """Dibuja n puntos en A y sus líneas epipolares en B (y viceversa).

    La línea epipolar del punto x̂_a en la imagen b es l_b = E·x̂_a (todos los
    x̂_b compatibles cumplen x̂_bᵀ·l_b = 0: el rayo de a, visto desde b, es
    una recta). En PIXELES se usa la matriz fundamental F = K⁻ᵀ·E·K⁻¹.
    """
    F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
    idx = np.linspace(0, len(pts_a) - 1, n).astype(int)
    pa, pb = pts_a[idx], pts_b[idx]

    la = cv2.computeCorrespondEpilines(pb.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    lb = cv2.computeCorrespondEpilines(pa.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

    def pintar(gray, lineas, puntos):
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        h, w = gray.shape
        rng = np.random.default_rng(3)
        for (a, b, c), (x, y) in zip(lineas, puntos):
            color = tuple(int(v) for v in rng.integers(64, 255, 3))
            # la recta a·x + b·y + c = 0, recortada a los bordes de la imagen
            x0, y0 = 0, int(-c / b) if abs(b) > 1e-9 else 0
            x1, y1 = w, int(-(c + a * w) / b) if abs(b) > 1e-9 else h
            cv2.line(vis, (x0, y0), (x1, y1), color, 1)
            cv2.circle(vis, (int(x), int(y)), 5, color, -1)
        return vis

    vis = np.hstack([pintar(gray_a, la, pa), pintar(gray_b, lb, pb)])
    cv2.imwrite(str(path), vis)


def distancia_a_epipolares(pts_a, pts_b, E, K) -> float:
    """Mediana de la distancia (px) de cada punto b a la epipolar de su a.

    d(x_b, l) = |lᵀ·x_b| / sqrt(l1² + l2²) — la métrica que RANSAC umbraliza.
    """
    F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
    pa = np.hstack([pts_a, np.ones((len(pts_a), 1))])
    pb = np.hstack([pts_b, np.ones((len(pts_b), 1))])
    l = pa @ F.T                                     # lineas en b
    d = np.abs(np.sum(l * pb, axis=1)) / np.linalg.norm(l[:, :2], axis=1)
    return float(np.median(d))


# ─────────────────────────────── programa ────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Nivel 07: geometria epipolar")
    parser.add_argument("--par", type=int, nargs=2, default=[0, 6],
                        help="indices de los dos frames")
    args = parser.parse_args()

    ia, ib = args.par
    gray_a, gray_b = cargar_frame(ia), cargar_frame(ib)
    K = leer_calibracion()
    gt = leer_gt()
    salida = AQUI / "salida"
    salida.mkdir(exist_ok=True)

    # ── 1 · Recuperar la pose y compararla con el GT exacto ──────────────────
    pts_a, pts_b = emparejar(gray_a, gray_b)
    r = estimar_pose(pts_a, pts_b, K)
    print(f"Par {ia} -> {ib}: {len(pts_a)} matches, {r['n_ransac']} inliers "
          f"RANSAC, {r['n_inliers']} pasan quiralidad")

    # GT relativo en el marco de la camara: T_b<-a = (T_w_b)⁻¹ · T_w_a
    # (los subindices se cancelan — nivel 03). recoverPose devuelve lo mismo.
    T_b_a = invert_se3(gt[ib]) @ gt[ia]
    err_R = angulo_entre_rotaciones(r["R"], T_b_a[:3, :3])
    err_t = angulo_entre_vectores(r["t"], T_b_a[:3, 3])
    baseline = float(np.linalg.norm(T_b_a[:3, 3]))
    print(f"\n1. Contra ground truth (baseline real {100*baseline:.1f} cm):")
    print(f"   error de rotacion:        {err_R:.3f} grados  (<1 es exito)")
    print(f"   error de direccion de t:  {err_t:.3f} grados  (<5 es exito)")
    print(f"   ||t|| estimada = {np.linalg.norm(r['t']):.3f}  <- SIEMPRE 1: "
          "la escala es inobservable (la leccion del monocular)")

    # ── 2 · Las líneas epipolares, dibujadas ──────────────────────────────────
    inl_a, inl_b = pts_a[r["mask"]], pts_b[r["mask"]]
    if len(inl_a) < 12:                     # quiralidad estricta: usar RANSAC
        inl_a, inl_b = pts_a, pts_b
    dibujar_epipolares(gray_a, gray_b, inl_a, inl_b, r["E"], K,
                       salida / "epipolares.png")
    d_med = distancia_a_epipolares(inl_a, inl_b, r["E"], K)
    print(f"\n2. Lineas epipolares: {salida / 'epipolares.png'}")
    print(f"   distancia mediana punto-epipolar: {d_med:.2f} px "
          "(el umbral de RANSAC fue 1.0)")

    # ── 3 · La trampa de recoverPose, reproducida ─────────────────────────────
    # Con paso PEQUENO (frames consecutivos: baseline 5 cm, escena a 4-14 m)
    # la profundidad/baseline es 80-280 >> 50: la sobrecarga basica descarta
    # casi todo como "infinito".
    ga, gb = cargar_frame(0), cargar_frame(1)
    qa, qb = emparejar(ga, gb)
    con = estimar_pose(qa, qb, K, usar_dist_thresh=True)
    sin = estimar_pose(qa, qb, K, usar_dist_thresh=False)
    print(f"\n3. La trampa (frames 0->1, profundidad/baseline ~80-280):")
    print(f"   recoverPose basico:            {sin['n_inliers']:4d} inliers")
    print(f"   con distanceThresh={CHEIRALITY_DIST_THRESH:.0f}:      "
          f"{con['n_inliers']:4d} inliers")
    print("   Misma E, misma geometria: solo cambio el umbral de quiralidad.")
    print("   (Leccion 1 del repo padre; le costo una tarde de depuracion.)")

    print("\nAhora corre `python verificacion.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
