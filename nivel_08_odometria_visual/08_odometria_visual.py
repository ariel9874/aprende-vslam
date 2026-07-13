#!/usr/bin/env python3
"""
Nivel 08 — Odometría Visual Monocular
=====================================

El primer SISTEMA completo del curso, en un solo archivo legible de arriba
a abajo:

    imágenes ─▶ características ORB ─▶ matching (ratio test)
             ─▶ matriz esencial (RANSAC) ─▶ pose relativa (R, t)
             ─▶ composición de trayectoria (¡hasta escala!) ─▶ ATE vs GT

Es una adaptación autocontenida del examples/01 del repo padre.

Limitaciones DELIBERADAS (son la lección, no un descuido):

  * ESCALA: con una sola cámara, la traslación entre dos vistas solo se
    recupera en dirección, no en magnitud. Aquí asumimos ||t|| = 1 en cada
    paso: la trayectoria tiene la FORMA correcta solo si la velocidad real
    es ~constante. (Se arregla con un mapa triangulado — nivel 10 — o con
    sensores métricos — niveles 15/16.)
  * DERIVA: cada pose se compone sobre la anterior; los errores se acumulan
    sin límite porque no hay optimización (nivel 11) ni bucles (nivel 12).
  * 2D-2D SIEMPRE: re-estimamos la geometría desde cero en cada par de
    frames. El nivel 10 trackea 3D-2D (PnP) contra un mapa, que es más
    estable.

Uso:
    python 08_odometria_visual.py                # datos del nivel (genera_datos.py)
    python 08_odometria_visual.py --images <dir> --calib <calib.txt> [--gt <tum.txt>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from evaluacion import ate, load_tum_positions, umeyama_alignment

AQUI = Path(__file__).resolve().parent
DATOS = AQUI / "data" / "secuencia"


# ─────────────────────────── utilidades SE(3) ────────────────────────────────

def invert_se3(T: np.ndarray) -> np.ndarray:
    """Inversa cerrada de una transformación rígida 4x4.

    ─── La matemática: el grupo SE(3) ───
    Una pose es T = [[R, t], [0, 1]] con R ∈ SO(3) (RᵀR = I, det R = +1) y
    t ∈ ℝ³. Actúa sobre puntos como X' = R·X + t, y componer dos poses es
    multiplicar sus matrices (¡el orden importa: SE(3) no es conmutativo!).

    Para invertir, despeja X de X' = R·X + t:
        X = Rᵀ·X' − Rᵀ·t    ⇒    T⁻¹ = [[Rᵀ, −Rᵀ·t], [0, 1]]
    La forma cerrada es más barata que np.linalg.inv y garantiza que el
    resultado siga siendo exactamente rígido.

    Notación del curso (nivel 03): T_a_b lleva puntos del frame b al frame a.
    Los subíndices se encadenan "cancelándose", como unidades:
        T_w_c2 = T_w_c1 · T_c1_c2      (w←c1 por c1←c2 da w←c2)
    """
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Matriz 3x3 -> cuaternión (qx, qy, qz, qw), método de Shepperd.
    (La derivación completa está en genera_datos.py, junto al ground truth.)"""
    R = np.asarray(R, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
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
    """Lee 'fx fy cx cy [w h]' (comentarios con #) y devuelve la K 3x3."""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            fx, fy, cx, cy = [float(v) for v in line.split()[:4]]
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    raise SystemExit(f"Calibracion vacia: {path}")


# ──────────────────────── el estimador de odometría ──────────────────────────

class MonocularVO:
    """Odometría visual monocular 2D-2D mínima.

    Mantiene la pose acumulada T_w_c (convención del curso: transforma puntos
    de cámara a mundo; el primer frame define el origen del mundo).
    """

    # Umbrales didácticos: en un sistema real serían configuración.
    MIN_MATCHES = 30          # menos que esto -> el matching no es fiable
    MIN_INLIERS = 15          # menos que esto -> la geometría no es fiable
    RATIO = 0.75              # ratio test de Lowe (nivel 06)
    RANSAC_PROB = 0.999
    RANSAC_THRESHOLD_PX = 1.0
    # Profundidad máxima aceptada en el test de quiralidad, en múltiplos del
    # baseline entre frames (ver la nota "TRAMPA CLÁSICA" en process_frame).
    CHEIRALITY_DIST_THRESH = 2000.0

    def __init__(self, K: np.ndarray) -> None:
        self.K = K
        # ORB (nivel 05): esquinas FAST en pirámide de escalas + descriptor
        # binario de 256 bits. BFMatcher con Hamming (nivel 06): el descriptor
        # es binario, la distancia natural es contar bits distintos.
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.T_w_c = np.eye(4)        # pose actual (mundo <- cámara)
        self.T_prev_rel = np.eye(4)   # último movimiento (para el coasting)
        self._prev = None             # (keypoints, descriptores) del frame anterior

    def _match(self, desc_a, desc_b) -> list:
        """Matching con ratio test de Lowe (nivel 06): para cada descriptor
        se buscan los DOS vecinos más cercanos y se acepta solo si el mejor
        es claramente mejor que el segundo (dist1 < RATIO * dist2). Descarta
        correspondencias ambiguas (texturas repetitivas): matches limpios
        ahora = RANSAC más barato y fiable después."""
        pares = self.bf.knnMatch(desc_a, desc_b, k=2)
        return [m for m, n in pares if m.distance < self.RATIO * n.distance]

    def process_frame(self, gray: np.ndarray) -> tuple[np.ndarray, dict]:
        """Procesa un frame y devuelve (T_w_c actualizada, info de diagnóstico)."""
        info = {"n_kps": 0, "n_matches": 0, "n_inliers": 0, "tracked": False}

        # ── PASO 1 · Extracción de características (nivel 05) ────────────────
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        info["n_kps"] = len(keypoints)

        if self._prev is None:
            # Primer frame: fija el origen del mundo. No hay geometría aún.
            self._prev = (keypoints, descriptors)
            info["tracked"] = True
            return self.T_w_c, info

        prev_kps, prev_desc = self._prev

        # ── PASO 2 · Matching (nivel 06) ──────────────────────────────────────
        matches = self._match(prev_desc, descriptors)
        info["n_matches"] = len(matches)

        if len(matches) < self.MIN_MATCHES:
            return self._coast(keypoints, descriptors, info)

        # Coordenadas de píxel de cada correspondencia (prev -> curr).
        pts_prev = np.float64([prev_kps[m.queryIdx].pt for m in matches])
        pts_curr = np.float64([keypoints[m.trainIdx].pt for m in matches])

        # ── PASO 3 · Geometría epipolar (nivel 07) ────────────────────────────
        #
        # ─── La matemática: la restricción epipolar ───
        # Sea X un punto 3D visto por ambas cámaras, relacionadas por la pose
        # relativa (R, t):  X_curr = R·X_prev + t. Trabajamos con RAYOS en
        # coordenadas normalizadas x̂ = K⁻¹·[u, v, 1]ᵀ (los píxeles fuera).
        #
        # Los vectores x̂_curr, R·x̂_prev y t son COPLANARES: los dos rayos y el
        # baseline forman el "plano epipolar" que contiene a X y a ambos
        # centros ópticos. Coplanaridad = triple producto escalar nulo:
        #
        #     x̂_curr · (t × R·x̂_prev) = 0
        #  ⇔  x̂_currᵀ · [t]_× · R · x̂_prev = 0 ,       E ≜ [t]_× · R
        #
        # E (matriz ESENCIAL) empaqueta la rotación y la DIRECCIÓN de t en una
        # sola 3x3. Tiene 5 grados de libertad (3 de R + 2 de dirección de t:
        # la ecuación es homogénea, la escala no cuenta) → bastan 5
        # correspondencias: el solver de 5 puntos de Nistér.
        #
        # ─── La matemática: RANSAC ───
        # El ratio test limpió mucho, pero UN solo outlier arruina un ajuste
        # por mínimos cuadrados. RANSAC itera: muestrear 5 matches al azar →
        # resolver E → contar cuántos matches la satisfacen (distancia
        # epipolar < threshold px) → quedarse con el E de mayor consenso. Si
        # w es la fracción de inliers, una muestra de 5 sale toda-inlier con
        # probabilidad w⁵, así que para acertar con probabilidad p bastan
        #     N = log(1 − p) / log(1 − w⁵)   iteraciones
        # (con w = 0.5 y p = 0.999, N ≈ 218: por eso corre en tiempo real).
        E, inlier_mask = cv2.findEssentialMat(
            pts_prev, pts_curr, self.K,
            method=cv2.RANSAC, prob=self.RANSAC_PROB,
            threshold=self.RANSAC_THRESHOLD_PX,
        )
        if E is None or E.shape != (3, 3):
            return self._coast(keypoints, descriptors, info)

        # ─── La matemática: de E a (R, t) ───
        # Con la SVD  E = U·diag(1, 1, 0)·Vᵀ  existen CUATRO factorizaciones
        # E = [t]_×·R (dos rotaciones × dos signos de t, el "twisted pair").
        # Solo UNA deja los puntos triangulados con profundidad positiva en
        # ambas cámaras: ese es el test de QUIRALIDAD que recoverPose hace
        # por nosotros. Devuelve T_curr<-prev.
        #
        # ¿Por qué ||t|| = 1? Porque E = [t]_×·R es homogénea en t: si
        # (R, t, {X_i}) explica las imágenes, (R, s·t, {s·X_i}) las explica
        # EXACTAMENTE igual para todo s > 0. Una cámara monocular no puede
        # medir la escala del mundo; se fija ||t|| = 1 por convención.
        #
        # TRAMPA CLÁSICA de OpenCV (medida en el repo padre, lección 1): la
        # sobrecarga básica de recoverPose solo acepta como inliers puntos
        # triangulados a menos de 50 unidades — y como ||t||=1, esas unidades
        # son MÚLTIPLOS DEL BASELINE. Si la cámara se mueve poco respecto a
        # la profundidad de la escena (depth/baseline > 50: lo normal en
        # KITTI, drones o cualquier avance suave), TODOS los puntos parecen
        # "en el infinito" y los inliers caen a ~0 aunque la geometría sea
        # perfecta. La sobrecarga con distanceThresh permite subir el umbral.
        n_inliers, R, t, pose_mask, _tri = cv2.recoverPose(
            E, pts_prev, pts_curr, self.K,
            distanceThresh=self.CHEIRALITY_DIST_THRESH, mask=inlier_mask,
        )
        info["n_inliers"] = int(n_inliers)

        if n_inliers < self.MIN_INLIERS:
            return self._coast(keypoints, descriptors, info)

        # CASO DEGENERADO que debes conocer: si la cámara solo ROTA, el
        # baseline t → 0 y con él E → 0: la restricción se satisface
        # trivialmente y la dirección de t es puro ruido. (Sin paralaje no
        # hay traslación observable.) Los sistemas serios eligen por consenso
        # entre homografía y esencial — así inicializa ORB-SLAM.

        # ── PASO 4 · Composición de la trayectoria (nivel 03) ────────────────
        # recoverPose dio T_curr<-prev; para acumular necesitamos el
        # movimiento en el mundo: T_w<-curr = T_w<-prev · (T_curr<-prev)⁻¹.
        T_curr_prev = np.eye(4)
        T_curr_prev[:3, :3] = R
        T_curr_prev[:3, 3] = t.ravel()  # ||t|| = 1: escala arbitraria

        T_rel = invert_se3(T_curr_prev)          # prev <- curr
        self.T_w_c = self.T_w_c @ T_rel          # componer sobre la anterior
        self.T_prev_rel = T_rel                  # recordar para el coasting

        info["tracked"] = True
        self._prev = (keypoints, descriptors)
        return self.T_w_c, info

    def _coast(self, keypoints, descriptors, info) -> tuple[np.ndarray, dict]:
        """Fallo de tracking: aplica el último movimiento conocido.

        Es el "modelo de velocidad constante", la mitigación más simple. Un
        sistema real intentaría RELOCALIZAR contra el mapa (nivel 13)."""
        self.T_w_c = self.T_w_c @ self.T_prev_rel
        self._prev = (keypoints, descriptors)
        return self.T_w_c, info


# ───────────────────────────── visualización ─────────────────────────────────

def guardar_grafico(est: np.ndarray, gt: np.ndarray | None, path: Path) -> None:
    """Vista cenital (X-Z). Si hay GT, alinea la estimación (Umeyama) para
    poder compararlas en el mismo marco."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[aviso] matplotlib no instalado: se omite trayectoria.png")
        return
    fig, ax = plt.subplots(figsize=(6.5, 6))
    if gt is not None:
        s, R, t = umeyama_alignment(est, gt)
        est = (s * (R @ est.T)).T + t
        ax.plot(gt[:, 0], gt[:, 2], "k--", lw=1.2, label="ground truth")
    ax.plot(est[:, 0], est[:, 2], "-", lw=1.5, label="estimada (alineada)")
    ax.plot(est[0, 0], est[0, 2], "go", label="inicio")
    ax.plot(est[-1, 0], est[-1, 2], "rs", label="fin")
    ax.set_xlabel("x [m]"), ax.set_ylabel("z [m]")
    ax.set_title("Odometria visual monocular - vista cenital")
    ax.axis("equal"), ax.grid(True, alpha=0.3), ax.legend()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────── programa ────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Nivel 08: odometria visual monocular",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--images", default=str(DATOS / "images"))
    parser.add_argument("--calib", default=str(DATOS / "calib.txt"))
    parser.add_argument("--gt", default=str(DATOS / "groundtruth.txt"),
                        help="'' para correr sin ground truth")
    parser.add_argument("--output", default=str(AQUI / "salida"))
    parser.add_argument("--max-frames", type=int, default=0, help="0 = todos")
    args = parser.parse_args()

    img_dir = Path(args.images)
    rutas = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    if not rutas:
        raise SystemExit(f"No hay imagenes en {img_dir}. "
                         "Corre `python genera_datos.py` primero.")
    if args.max_frames:
        rutas = rutas[:args.max_frames]

    K = leer_calibracion(Path(args.calib))
    print(f"Secuencia: {len(rutas)} imagenes | K: fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
          f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    vo = MonocularVO(K)
    poses: list[tuple[float, np.ndarray]] = []

    for i, ruta in enumerate(rutas):
        gray = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
        T_w_c, info = vo.process_frame(gray)
        poses.append((i / 30.0, T_w_c.copy()))

        if i % 20 == 0 or not info["tracked"]:
            x, y, z = T_w_c[:3, 3]
            estado = "ok" if info["tracked"] else "COASTING (tracking debil)"
            print(f"frame {i:5d} | inliers {info['n_inliers']:4d} | "
                  f"pos [{x:+7.2f} {y:+7.2f} {z:+7.2f}] | {estado}")

    # ── Resultados ────────────────────────────────────────────────────────────
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    lines = []
    for ts, T in poses:
        q = rotation_to_quaternion(T[:3, :3])
        tx, ty, tz = T[:3, 3]
        lines.append(f"{ts:.6f} {tx:.6f} {ty:.6f} {tz:.6f} "
                     f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
    (out / "trayectoria.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nTrayectoria: {len(poses)} poses -> {out / 'trayectoria.txt'} (formato TUM)")

    est = np.stack([T[:3, 3] for _, T in poses])
    gt = None
    if args.gt and Path(args.gt).exists():
        gt = load_tum_positions(args.gt)
        # La VO emite una pose por frame: GT y estimacion van 1 a 1.
        m = ate(est, gt)
        print(f"ATE (alineacion de similitud): rmse {m['rmse']*100:.1f} cm | "
              f"media {m['mean']*100:.1f} | max {m['max']*100:.1f} | "
              f"{m['rmse_pct']:.1f}% del recorrido")
        print("Referencia del nivel: ~13 cm (la medicion de v0.1 del repo padre;")
        print("8-20 cm es normal, RANSAC es aleatorio).")

    guardar_grafico(est, gt, out / "trayectoria.png")
    print(f"Grafico: {out / 'trayectoria.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
