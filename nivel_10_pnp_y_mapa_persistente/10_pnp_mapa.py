#!/usr/bin/env python3
"""
Nivel 10 — PnP y mapa persistente
=================================

El salto de ODOMETRÍA a SLAM: en vez de re-estimar la geometría entre cada
par de frames (nivel 08), construimos un MAPA que persiste y localizamos la
cámara CONTRA él.

    INIT   : dos vistas -> matriz esencial -> triangular -> mapa (gauge)
    TRACK  : matchear el frame contra el MAPA -> PnP -> pose
    KF     : cada cierto tiempo, triangular puntos NUEVOS y ampliar el mapa

Al final compara, sobre la MISMA secuencia y con el mismo frontend, la
odometría 2D-2D del nivel 08 contra este tracker. El unico cambio es
arquitectonico — y el ATE se parte por dos.

Uso:
    python 10_pnp_mapa.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from evaluacion import ate, load_tum_positions, umeyama_alignment

AQUI = Path(__file__).resolve().parent
DATOS = AQUI / "data" / "secuencia"


# ─────────────────────────── utilidades (niveles 03, 07, 09) ─────────────────

def invert_se3(T: np.ndarray) -> np.ndarray:
    """T⁻¹ = [[Rᵀ, −Rᵀ·t], [0, 1]] (nivel 03)."""
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


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


def leer_calibracion(path: Path) -> np.ndarray:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            fx, fy, cx, cy = [float(v) for v in line.split()[:4]]
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    raise SystemExit(f"Calibracion vacia: {path}")


def proyectar(K, T_w_c, pts_w):
    """Puntos del mundo -> (uv, Z) en la vista T_w_c (nivel 09)."""
    T_c_w = invert_se3(T_w_c)
    pc = (T_c_w[:3, :3] @ pts_w.T).T + T_c_w[:3, 3]
    Z = pc[:, 2]
    uv = np.full((len(pts_w), 2), np.nan)
    ok = Z > 1e-6
    uv[ok] = np.stack([K[0, 0] * pc[ok, 0] / Z[ok] + K[0, 2],
                       K[1, 1] * pc[ok, 1] / Z[ok] + K[1, 2]], axis=1)
    return uv, Z


def triangular(K, T0, T1, p0, p1, reproj_px=2.0, min_par_deg=0.3):
    """DLT + quiralidad + reproyeccion + paralaje (nivel 09, resumido)."""
    P0 = K @ invert_se3(T0)[:3]
    P1 = K @ invert_se3(T1)[:3]
    Xh = cv2.triangulatePoints(P0, P1, p0.T, p1.T)
    w = np.where(np.abs(Xh[3]) < 1e-12, 1e-12, Xh[3])
    pts = (Xh[:3] / w).T

    uv0, Z0 = proyectar(K, T0, pts)
    uv1, Z1 = proyectar(K, T1, pts)
    err = np.maximum(np.linalg.norm(uv0 - p0, axis=1),
                     np.linalg.norm(uv1 - p1, axis=1))
    ok = (Z0 > 1e-6) & (Z1 > 1e-6) & (np.nan_to_num(err, nan=np.inf) < reproj_px)

    r0, r1 = pts - T0[:3, 3], pts - T1[:3, 3]
    cos = np.einsum("ij,ij->i", r0, r1) / (
        np.linalg.norm(r0, axis=1) * np.linalg.norm(r1, axis=1) + 1e-12)
    ok &= np.degrees(np.arccos(np.clip(cos, -1, 1))) > min_par_deg
    return pts, ok


# ────────────────────────────── el mapa ──────────────────────────────────────

class Mapa:
    """El mapa disperso: puntos 3D + su descriptor ORB.

    Lo MÍNIMO que hace falta para localizar contra él. (Un SLAM de verdad
    guarda además qué keyframe vio cada punto — la "covisibilidad" — y eso
    llega en el nivel 13; aquí el mapa es una bolsa de puntos.)
    """

    def __init__(self) -> None:
        self.puntos = np.zeros((0, 3))
        self.descriptores = np.zeros((0, 32), np.uint8)

    def añadir(self, pts: np.ndarray, desc: np.ndarray) -> None:
        self.puntos = np.vstack([self.puntos, pts])
        self.descriptores = np.vstack([self.descriptores, desc])

    def __len__(self) -> int:
        return len(self.puntos)


# ──────────────────────────── el tracker PnP ─────────────────────────────────

class TrackerPnP:
    """Tracking 3D-2D contra un mapa persistente."""

    # Umbrales didácticos (en el repo padre son perillas calibradas por barrido).
    MIN_INIT_FLOW_PX = 20.0    # antes de esto, no hay baseline para triangular
    MIN_MAP_MATCHES = 30       # menos que esto: no me fio del matching
    MIN_PNP_INLIERS = 15       # menos que esto: no me fio de la pose
    KF_MIN_GAP = 3             # no insertar keyframes pegados (sin paralaje)
    KF_MAX_GAP = 15            # ... pero tampoco morir de hambre
    KF_INLIER_RATIO = 0.6      # si quedan pocos inliers, el mapa se acaba: KF
    KF_MIN_INLIERS = 45        # PISO DE SALUD: nunca crear mapa desde pose
                               # incierta. El repo padre midio el precio de
                               # ignorarlo: un KF con 26 inliers creo 584
                               # puntos basura y el tracking se teletransporto.
    RATIO = 0.75
    PNP_REPROJ_PX = 3.0

    def __init__(self, K: np.ndarray) -> None:
        self.K = K
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.mapa = Mapa()

        self.T_w_c = np.eye(4)
        self.T_rel = np.eye(4)          # ultimo movimiento (modelo de velocidad)
        self.estado = "INIT"
        self._buffer = None             # (kps, desc) del frame de referencia de init
        self._kf = None                 # (T_w_c, kps, desc) del ultimo keyframe
        self._frames_desde_kf = 0
        self.n_keyframes = 0
        self.n_puntos_init = 0          # cuantos puntos nacieron en la INIT
                                        # (son los unicos con el gauge mediana=1;
                                        #  los demas HEREDAN la escala, no la fijan)

    def _match(self, da, db):
        pares = self.bf.knnMatch(da, db, k=2)
        return [m for m, n in pares if m.distance < self.RATIO * n.distance]

    # ── INIT ─────────────────────────────────────────────────────────────────
    def _inicializar(self, kps, desc, info) -> None:
        """Dos vistas con suficiente baseline -> E -> triangular -> gauge.

        ─── La matemática: el GAUGE monocular ───
        La matriz esencial da ‖t‖ = 1: la escala del mundo es libre (nivel
        07). Hay que ELEGIRLA, y la convención de este curso (y del repo
        padre) es: la profundidad MEDIANA del mapa inicial vale 1.0. Es
        arbitrario y da igual cuál sea — lo que importa es que a partir de
        aquí TODO (poses y puntos nuevos) hereda esa misma escala en vez de
        re-inventar una por frame. Eso es lo que mata la deriva de escala
        del nivel 08.
        """
        if self._buffer is None:
            self._buffer = (kps, desc)
            return

        kps0, desc0 = self._buffer
        matches = self._match(desc0, desc)
        if len(matches) < self.MIN_MAP_MATCHES:
            self._buffer = (kps, desc)
            return

        p0 = np.float64([kps0[m.queryIdx].pt for m in matches])
        p1 = np.float64([kps[m.trainIdx].pt for m in matches])

        # ¿hay baseline? Sin flujo, la triangulacion es basura (nivel 09).
        if np.median(np.linalg.norm(p1 - p0, axis=1)) < self.MIN_INIT_FLOW_PX:
            return                       # esperar: aun no nos hemos movido

        E, mask = cv2.findEssentialMat(p0, p1, self.K, method=cv2.RANSAC,
                                       prob=0.999, threshold=1.0)
        if E is None or E.shape != (3, 3):
            self._buffer = (kps, desc)
            return
        n_inl, R, t, pose_mask, _ = cv2.recoverPose(
            E, p0, p1, self.K, distanceThresh=2000.0, mask=mask)
        if n_inl < self.MIN_PNP_INLIERS:
            self._buffer = (kps, desc)
            return

        T_c1_c0 = np.eye(4)
        T_c1_c0[:3, :3], T_c1_c0[:3, 3] = R, t.ravel()
        T_w_c0 = np.eye(4)                       # la primera vista ES el mundo
        T_w_c1 = invert_se3(T_c1_c0)

        ok = pose_mask.ravel().astype(bool)
        pts, val = triangular(self.K, T_w_c0, T_w_c1, p0[ok], p1[ok])
        if val.sum() < self.MIN_PNP_INLIERS:
            self._buffer = (kps, desc)
            return

        # GAUGE: profundidad mediana = 1.0 (ver el docstring)
        _, Z = proyectar(self.K, T_w_c0, pts[val])
        escala = 1.0 / float(np.median(Z))
        pts_esc = pts[val] * escala
        T_w_c1[:3, 3] *= escala

        idx = np.where(ok)[0][val]
        self.mapa.añadir(pts_esc, np.array([desc[matches[i].trainIdx] for i in idx]))
        self.n_puntos_init = len(pts_esc)

        self.T_w_c = T_w_c1
        self.T_rel = T_w_c1              # el primer "movimiento"
        self._kf = (T_w_c1, kps, desc)
        self._frames_desde_kf = 0
        self.n_keyframes = 2
        self.estado = "TRACK"
        info.update(estado="INIT-OK", n_mapa=len(self.mapa))

    # ── TRACK ────────────────────────────────────────────────────────────────
    def _trackear(self, kps, desc, info) -> None:
        """Matchear contra el MAPA (3D-2D) y resolver PnP.

        ─── La matemática: PnP ───
            T* = argmin_T  Σ ‖ π(K, T_c_w·X_i) − u_i ‖²
        cv2.solvePnPRansac usa EPnP como solver interno (expresa los X_i como
        combinación de 4 puntos de control → sistema lineal O(n)) dentro de
        RANSAC para separar inliers, y luego refinamos con Levenberg-Marquardt
        SOLO con los inliers (el óptimo geométrico de verdad).

        OpenCV devuelve la pose MUNDO→CÁMARA (T_c_w) y parametriza R como
        rvec (eje-ángulo, Rodrigues): hay que invertir para nuestra T_w_c.
        """
        matches = self._match(desc, self.mapa.descriptores)
        info["n_matches"] = len(matches)
        if len(matches) < self.MIN_MAP_MATCHES:
            return self._coast(info)

        pix = np.float64([kps[m.queryIdx].pt for m in matches])
        pts = self.mapa.puntos[[m.trainIdx for m in matches]]

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts, pix, self.K, None, iterationsCount=200,
            reprojectionError=self.PNP_REPROJ_PX, confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers) < self.MIN_PNP_INLIERS:
            return self._coast(info)

        inl = inliers.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(pts[inl], pix[inl], self.K, None,
                                          rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        T_c_w = np.eye(4)
        T_c_w[:3, :3], T_c_w[:3, 3] = R, tvec.ravel()
        T_nueva = invert_se3(T_c_w)

        self.T_rel = invert_se3(self.T_w_c) @ T_nueva
        self.T_w_c = T_nueva
        info.update(estado="TRACK", n_inliers=len(inl))

        # ¿toca keyframe? (politica: ni pegados ni con hambre)
        self._frames_desde_kf += 1
        ratio = len(inl) / max(len(matches), 1)
        hambre = (ratio < self.KF_INLIER_RATIO
                  or self._frames_desde_kf >= self.KF_MAX_GAP)
        if hambre and self._frames_desde_kf >= self.KF_MIN_GAP:
            self._insertar_keyframe(kps, desc, len(inl), info)

    def _insertar_keyframe(self, kps, desc, n_inliers, info) -> None:
        """Triangula puntos NUEVOS contra el keyframe anterior y amplía el mapa."""
        # PISO DE SALUD (la leccion cara del repo padre): con una pose dudosa,
        # los puntos nuevos nacen en el sitio equivocado y envenenan el mapa
        # para siempre. Mas vale un mapa pequeno que un mapa mentiroso.
        if n_inliers < self.KF_MIN_INLIERS:
            return

        T_kf, kps_kf, desc_kf = self._kf
        matches = self._match(desc_kf, desc)
        if len(matches) < self.MIN_MAP_MATCHES:
            return
        p0 = np.float64([kps_kf[m.queryIdx].pt for m in matches])
        p1 = np.float64([kps[m.trainIdx].pt for m in matches])

        pts, val = triangular(self.K, T_kf, self.T_w_c, p0, p1)
        if val.sum() == 0:
            return
        idx = np.where(val)[0]
        self.mapa.añadir(pts[val],
                         np.array([desc[matches[i].trainIdx] for i in idx]))

        self._kf = (self.T_w_c.copy(), kps, desc)
        self._frames_desde_kf = 0
        self.n_keyframes += 1
        info.update(kf=True, n_mapa=len(self.mapa))

    def _coast(self, info) -> None:
        """Sin pose fiable: aplicar el ultimo movimiento (velocidad constante).
        Un sistema real intentaria RELOCALIZAR (nivel 13)."""
        self.T_w_c = self.T_w_c @ self.T_rel
        info["estado"] = "COAST"

    def procesar(self, gray: np.ndarray) -> tuple[np.ndarray, dict]:
        info = {"estado": self.estado, "n_matches": 0, "n_inliers": 0,
                "kf": False, "n_mapa": len(self.mapa)}
        kps, desc = self.orb.detectAndCompute(gray, None)
        if desc is None or len(kps) < 10:
            self._coast(info)
            return self.T_w_c, info

        if self.estado == "INIT":
            self._inicializar(kps, desc, info)
        else:
            self._trackear(kps, desc, info)
        return self.T_w_c, info


# ─────────────────── la odometría del nivel 08 (para comparar) ───────────────

class VO2D2D:
    """La odometría 2D-2D del nivel 08, tal cual, para medir el salto."""

    RATIO, MIN_MATCHES, MIN_INLIERS = 0.75, 30, 15

    def __init__(self, K):
        self.K = K
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.T_w_c, self.T_rel, self._prev = np.eye(4), np.eye(4), None

    def procesar(self, gray):
        kps, desc = self.orb.detectAndCompute(gray, None)
        if self._prev is None:
            self._prev = (kps, desc)
            return self.T_w_c
        kps0, desc0 = self._prev
        pares = self.bf.knnMatch(desc0, desc, k=2)
        matches = [m for m, n in pares if m.distance < self.RATIO * n.distance]
        if len(matches) < self.MIN_MATCHES:
            self.T_w_c = self.T_w_c @ self.T_rel
            self._prev = (kps, desc)
            return self.T_w_c
        p0 = np.float64([kps0[m.queryIdx].pt for m in matches])
        p1 = np.float64([kps[m.trainIdx].pt for m in matches])
        E, mask = cv2.findEssentialMat(p0, p1, self.K, method=cv2.RANSAC,
                                       prob=0.999, threshold=1.0)
        if E is None or E.shape != (3, 3):
            self.T_w_c = self.T_w_c @ self.T_rel
            self._prev = (kps, desc)
            return self.T_w_c
        n, R, t, _, _ = cv2.recoverPose(E, p0, p1, self.K,
                                        distanceThresh=2000.0, mask=mask)
        if n >= self.MIN_INLIERS:
            T_c1_c0 = np.eye(4)
            T_c1_c0[:3, :3], T_c1_c0[:3, 3] = R, t.ravel()
            self.T_rel = invert_se3(T_c1_c0)
        self.T_w_c = self.T_w_c @ self.T_rel
        self._prev = (kps, desc)
        return self.T_w_c


# ─────────────────────────────── programa ────────────────────────────────────

def guardar_ply(pts: np.ndarray, path: Path) -> None:
    cab = ["ply", "format ascii 1.0", f"element vertex {len(pts)}",
           "property float x", "property float y", "property float z",
           "end_header"]
    cab += [f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in pts]
    path.write_text("\n".join(cab) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Nivel 10: PnP + mapa persistente")
    parser.add_argument("--images", default=str(DATOS / "images"))
    parser.add_argument("--calib", default=str(DATOS / "calib.txt"))
    parser.add_argument("--gt", default=str(DATOS / "groundtruth.txt"))
    parser.add_argument("--output", default=str(AQUI / "salida"))
    args = parser.parse_args()

    rutas = sorted(Path(args.images).glob("*.png"))
    if not rutas:
        raise SystemExit(f"No hay imagenes en {args.images}. "
                         "Corre `python genera_datos.py` primero.")
    K = leer_calibracion(Path(args.calib))
    grises = [cv2.imread(str(r), cv2.IMREAD_GRAYSCALE) for r in rutas]
    print(f"Secuencia: {len(grises)} imagenes\n")

    # ── El tracker PnP ────────────────────────────────────────────────────────
    tr = TrackerPnP(K)
    poses, estados = [], []
    for i, gray in enumerate(grises):
        T, info = tr.procesar(gray)
        poses.append(T.copy())
        estados.append(info["estado"])
        if i % 15 == 0 or info["kf"]:
            print(f"frame {i:3d} | {info['estado']:7s} | inliers "
                  f"{info['n_inliers']:4d} | mapa {info['n_mapa']:5d} pts"
                  + ("  <- keyframe" if info["kf"] else ""))

    print(f"\nMapa final: {len(tr.mapa)} puntos en {tr.n_keyframes} keyframes")
    n_coast = sum(1 for e in estados if e == "COAST")
    print(f"Frames en COAST (sin pose fiable): {n_coast}")

    # ── La comparacion: el mismo frontend, otra arquitectura ─────────────────
    vo = VO2D2D(K)
    poses_vo = [vo.procesar(g).copy() for g in grises]

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    lineas = []
    for i, T in enumerate(poses):
        q = rotation_to_quaternion(T[:3, :3])
        tx, ty, tz = T[:3, 3]
        lineas.append(f"{i/30.0:.6f} {tx:.6f} {ty:.6f} {tz:.6f} "
                      f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
    (out / "trayectoria.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    guardar_ply(tr.mapa.puntos, out / "mapa.ply")

    gt = load_tum_positions(args.gt)
    est_pnp = np.stack([T[:3, 3] for T in poses])
    est_vo = np.stack([T[:3, 3] for T in poses_vo])
    m_pnp, m_vo = ate(est_pnp, gt), ate(est_vo, gt)

    print("\n" + "=" * 62)
    print("EL SALTO DEL NIVEL (misma secuencia, mismo ORB, mismo ratio test):")
    print(f"  VO 2D-2D (nivel 08):        ATE {100*m_vo['rmse']:5.1f} cm")
    print(f"  PnP contra mapa (este):     ATE {100*m_pnp['rmse']:5.1f} cm")
    print(f"  mejora: {100*(1 - m_pnp['rmse']/m_vo['rmse']):.0f}%")
    print("=" * 62)
    print("No tocaste una sola perilla: cambiaste la ARQUITECTURA. La pose ya")
    print("no se apoya en la anterior, sino en un mapa que persiste.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6.5))
        ax.plot(gt[:, 0], gt[:, 2], "k--", lw=1.5, label="ground truth")
        for est, m, nombre, c in [(est_vo, m_vo, "VO 2D-2D (nivel 08)", "tab:orange"),
                                  (est_pnp, m_pnp, "PnP + mapa (nivel 10)", "tab:blue")]:
            s, R, t = umeyama_alignment(est, gt)
            al = (s * (R @ est.T)).T + t
            ax.plot(al[:, 0], al[:, 2], "-", color=c, lw=1.6,
                    label=f"{nombre}: {100*m['rmse']:.1f} cm")
        ax.set_xlabel("x [m]"), ax.set_ylabel("z [m]")
        ax.set_title("El mismo frontend, dos arquitecturas")
        ax.axis("equal"), ax.grid(True, alpha=0.3), ax.legend()
        fig.savefig(out / "comparacion.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        print("[aviso] matplotlib no instalado: se omite comparacion.png")

    print(f"\nResultados en {out}. Ahora corre `python verificacion.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
