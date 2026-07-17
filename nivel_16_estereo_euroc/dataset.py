"""Carga de secuencias EuRoC MAV (formato ASL) y el RIG estéreo.

EuRoC es el segundo dataset real del curso, y trae dos novedades:

  1. DOS cámaras calibradas (cam0 izquierda, cam1 derecha) montadas en un
     dron: el rig estéreo. La "cámara derecha virtual" del nivel 15 se
     vuelve REAL.
  2. El ground truth vive en el frame del CUERPO (IMU), no de la cámara:
     hay que corregir el brazo de palanca (ver leer_gt_euroc).

Estructura en disco:
    <sec>/mav0/cam0/{data.csv, data/*.png, sensor.yaml}
    <sec>/mav0/cam1/{...}
    <sec>/mav0/state_groundtruth_estimate0/data.csv

─── La matemática: por qué RECTIFICAR ────────────────────────────────────────
Dos cámaras cualesquiera ven un punto sobre su recta epipolar (nivel 07):
buscar la correspondencia es una búsqueda 2D. La RECTIFICACIÓN reproyecta
ambas imágenes a un par virtual con ejes ópticos paralelos y planos de imagen
coplanares → las rectas epipolares se vuelven FILAS horizontales y la
correspondencia colapsa a una búsqueda 1D: el mismo punto aparece en (u_L, v)
y (u_R, v) con la MISMA v. La diferencia d = u_L − u_R es la DISPARIDAD, y la
geometría del par da la profundidad:

    z = fx · b / d          (b = baseline; fx·b ≡ bf)

`cv2.stereoRectify` calcula las rotaciones R1, R2 y las nuevas matrices de
proyección P1, P2 desde los intrínsecos + la pose relativa cam0<-cam1. Tras
rectificar, la cámara izquierda es un pinhole SIN distorsión (P1), y
bf = −P2[0,3]: P2 codifica el baseline como −fx·b en su columna de traslación.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, List, Tuple

import cv2
import numpy as np


def _lista_yaml(texto: str, clave: str) -> List[float]:
    """Extrae la lista `clave: [a, b, ...]` de un sensor.yaml de EuRoC.

    Parser mínimo sin dependencia de PyYAML: soporta listas multilínea (la
    matriz T_BS.data) e ignora comentarios tras el `]`. Es cuadrado para lo
    que necesitamos: intrinsics, distortion_coefficients, resolution y T_BS.
    """
    m = re.search(rf"(?m)^\s*{re.escape(clave)}\s*:\s*\[(.*?)\]",
                  texto, re.DOTALL)
    if not m:
        raise ValueError(f"clave '{clave}' no encontrada en el sensor.yaml")
    return [float(x) for x in m.group(1).replace("\n", " ").split(",")
            if x.strip()]


def camara_euroc(root: str | Path, cam: str = "cam0"):
    """(K 3x3, dist (5,), (ancho, alto)) desde `mav0/<cam>/sensor.yaml`.

    EuRoC da 4 coeficientes (k1, k2, p1, p2); k3 = 0 en Brown-Conrady.
    """
    texto = (Path(root) / "mav0" / cam / "sensor.yaml").read_text(
        encoding="utf-8")
    fx, fy, cx, cy = _lista_yaml(texto, "intrinsics")
    k1, k2, p1, p2 = _lista_yaml(texto, "distortion_coefficients")[:4]
    w, h = (int(round(v)) for v in _lista_yaml(texto, "resolution"))
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return K, np.array([k1, k2, p1, p2, 0.0]), (w, h)


def _T_BS(root: str | Path, cam: str) -> np.ndarray:
    """Extrínseco T_BS (cuerpo<-sensor, 4x4) de `mav0/<cam>/sensor.yaml`."""
    texto = (Path(root) / "mav0" / cam / "sensor.yaml").read_text(
        encoding="utf-8")
    return np.array(_lista_yaml(texto, "data")).reshape(4, 4)


def _invert_se3(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


class CargadorEuRoC:
    """Itera (timestamp_seg, imagen_gris) sobre una cámara de EuRoC.

    Los timestamps del CSV están en NANOsegundos → se pasan a segundos
    (÷1e9) para casar con el ground truth y con el resto del curso.
    """

    def __init__(self, root: str | Path, cam: str = "cam0") -> None:
        self.root = Path(root)
        self.cam_dir = self.root / "mav0" / cam
        csv = self.cam_dir / "data.csv"
        if not csv.is_file():
            raise FileNotFoundError(f"No existe {csv} (¿es una secuencia "
                                    "EuRoC? corre descarga_datos.py)")
        self.entradas: List[Tuple[float, str]] = []
        for linea in csv.read_text(encoding="utf-8").splitlines():
            linea = linea.strip()
            if not linea or linea.startswith("#"):
                continue
            ts, fn = linea.split(",")[:2]
            self.entradas.append((int(ts) * 1e-9, fn.strip()))
        if not self.entradas:
            raise FileNotFoundError(f"data.csv sin entradas en {self.cam_dir}")

    @property
    def timestamps(self) -> np.ndarray:
        return np.array([t for t, _ in self.entradas], dtype=np.float64)

    def __len__(self) -> int:
        return len(self.entradas)

    def __iter__(self) -> Iterator[Tuple[float, np.ndarray]]:
        for ts, fn in self.entradas:
            img = cv2.imread(str(self.cam_dir / "data" / fn),
                             cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError(f"No se pudo leer: {self.cam_dir / 'data' / fn}")
            yield ts, img


class RigEstereo:
    """Rectificación del par EuRoC (cam0 izquierda, cam1 derecha).

    Tras construirlo:
      - `rectify(izq, der)` devuelve el par con epipolares = filas;
      - `K` es la cámara izquierda RECTIFICADA (pinhole sin distorsión);
      - `bf` = fx·b en px·m — el MISMO número que el BA usa en u_R = u − bf/z,
        solo que aquí no es una convención: lo mide la calibración del rig.
    """

    def __init__(self, root: str | Path, izq: str = "cam0",
                 der: str = "cam1") -> None:
        K_L, dist_L, (w, h) = camara_euroc(root, izq)
        K_R, dist_R, _ = camara_euroc(root, der)
        # Pose relativa: X_der = T_R_L · X_izq, con T_R_L = T_B_R⁻¹ · T_B_L
        # (ambos sensor.yaml dan cuerpo<-sensor). stereoRectify pide justo la
        # transformacion de la camara 1 (izq) a la 2 (der).
        T_R_L = _invert_se3(_T_BS(root, der)) @ _T_BS(root, izq)
        R1, R2, P1, P2, self.Q, _, _ = cv2.stereoRectify(
            K_L, dist_L, K_R, dist_R, (w, h),
            T_R_L[:3, :3], T_R_L[:3, 3],
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        self.K = np.array([[P1[0, 0], 0.0, P1[0, 2]],
                           [0.0, P1[1, 1], P1[1, 2]],
                           [0.0, 0.0, 1.0]])
        self.ancho, self.alto = w, h
        self.baseline = float(-P2[0, 3] / P2[0, 0])   # metros
        self.bf = float(-P2[0, 3])                    # fx·b (px·m)
        self._map_izq = cv2.initUndistortRectifyMap(
            K_L, dist_L, R1, P1, (w, h), cv2.CV_32FC1)
        self._map_der = cv2.initUndistortRectifyMap(
            K_R, dist_R, R2, P2, (w, h), cv2.CV_32FC1)

    def rectify(self, gris_izq: np.ndarray, gris_der: np.ndarray):
        """(izq, der) rectificadas: filas = rectas epipolares (búsqueda 1D)."""
        return (cv2.remap(gris_izq, *self._map_izq, cv2.INTER_LINEAR),
                cv2.remap(gris_der, *self._map_der, cv2.INTER_LINEAR))


class CargadorEstereo:
    """Itera (ts, izquierda_rectificada, profundidad) — la MISMA firma que la
    SecuenciaTUM con profundidad del nivel 15, para que el tracker métrico
    funcione SIN CAMBIOS. La profundidad no viene de un sensor: se triangula
    por DISPARIDAD densa (cv2.StereoSGBM) sobre el par rectificado.

        depth = bf / disparidad;  = 0 donde la disparidad es inválida o cae
        fuera de [min_depth, max_depth] (mismo convenio '0 = sin dato').

    ─── La simetría bonita (lección 37 del padre) ───
    El ruido de la profundidad estéreo crece con z² (∂z/∂d = −bf/d²)... y el
    peso del residuo u_R = u − bf/z en el BA decae exactamente con z²
    (∂u_R/∂z = bf/z²). La geometría compensa el ruido en la dirección
    correcta — la misma cancelación que en el Kinect del nivel 15.
    """

    def __init__(self, root: str | Path, rig: RigEstereo | None = None,
                 num_disparities: int = 96, block_size: int = 7,
                 min_depth: float = 0.5, max_depth: float = 40.0) -> None:
        self.rig = rig or RigEstereo(root)
        self._izq = CargadorEuRoC(root, "cam0")
        self._der = CargadorEuRoC(root, "cam1")
        self.min_depth, self.max_depth = min_depth, max_depth
        # SGBM: el matcher denso estandar. P1/P2 penalizan saltos de
        # disparidad (suavidad), escalados con el bloque como recomienda
        # OpenCV. uniquenessRatio y el filtro de speckle matan los falsos.
        self._sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * block_size ** 2, P2=32 * block_size ** 2,
            uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
            disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    @property
    def timestamps(self) -> np.ndarray:
        return self._izq.timestamps

    @property
    def stereo_bf(self) -> float:
        return self.rig.bf

    def __len__(self) -> int:
        return min(len(self._izq), len(self._der))

    def __iter__(self) -> Iterator[Tuple[float, np.ndarray, np.ndarray]]:
        for (ts, gl), (_, gr) in zip(self._izq, self._der):
            L, R = self.rig.rectify(gl, gr)
            disp = self._sgbm.compute(L, R).astype(np.float32) / 16.0
            prof = np.zeros_like(disp)
            valida = disp > 0.0
            prof[valida] = self.rig.bf / disp[valida]
            prof[(prof < self.min_depth) | (prof > self.max_depth)] = 0.0
            yield ts, L, prof


def _quat_wxyz_a_R(q: np.ndarray) -> np.ndarray:
    """Rotación 3x3 desde un cuaternión [w, x, y, z] (la convención del GT
    de EuRoC: q_RS = orientación del cuerpo en el frame de referencia)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def leer_gt_euroc(root: str | Path, cam: str = "cam0"):
    """(timestamps_seg, posiciones de la CÁMARA en el mundo) del GT de EuRoC.

    ─── La matemática: el GT vive en el frame del CUERPO (IMU) ───
    EuRoC entrega T_world_body (posición + cuaternión del CUERPO). La cámara
    está desplazada del cuerpo por el extrínseco T_BS. Su posición es:

        p_cam_world = R_world_body · t_BS + p_world_body

    Sin esta corrección, comparar la trayectoria de la cámara (estimada)
    contra la del cuerpo (GT) mete un error de BRAZO DE PALANCA que ROTA con
    la pose — no lo absorbe ninguna alineación global del ATE. En EuRoC el
    brazo es de ~7 cm: pequeño, pero es justo el orden del ATE que queremos
    medir. La trampa perfecta.
    """
    gt_csv = Path(root) / "mav0" / "state_groundtruth_estimate0" / "data.csv"
    data = np.loadtxt(gt_csv, delimiter=",")
    if data.ndim == 1:
        data = data[None, :]
    ts = data[:, 0] * 1e-9
    p_cuerpo = data[:, 1:4]
    q = data[:, 4:8]                                  # [w, x, y, z]
    t_bs = _T_BS(root, cam)[:3, 3]
    pos = np.array([_quat_wxyz_a_R(q[i]) @ t_bs + p_cuerpo[i]
                    for i in range(len(ts))])
    return ts, pos


def asociar_por_timestamp(ts_consulta: np.ndarray, ts_ref: np.ndarray,
                          max_dt: float = 0.02) -> np.ndarray:
    """Para cada t de `ts_consulta`, el índice del más cercano en `ts_ref`
    (o -1 si dista más de `max_dt` s). La asociación del nivel 14."""
    ref = np.asarray(ts_ref, dtype=np.float64)
    orden = np.argsort(ref)
    ref_ord = ref[orden]
    idx = np.searchsorted(ref_ord, ts_consulta)
    out = np.full(len(ts_consulta), -1, dtype=int)
    for i, (q, j) in enumerate(zip(ts_consulta, idx)):
        cands = [k for k in (j - 1, j) if 0 <= k < len(ref_ord)]
        mejor = min(cands, key=lambda k: abs(ref_ord[k] - q), default=None)
        if mejor is not None and abs(ref_ord[mejor] - q) <= max_dt:
            out[i] = orden[mejor]
    return out
