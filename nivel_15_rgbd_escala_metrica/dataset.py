"""Carga de secuencias TUM RGB-D: el primer loader de datos REALES del curso.

Hasta el nivel 13 los datos eran sintéticos: geometría exacta, sin distorsión,
timestamps perfectos (frame i = instante i). Un dataset real trae sus propias
convenciones, y este módulo las resuelve:

  1. CALIBRACIÓN publicada: cada cámara Freiburg tiene su K y su distorsión
     Brown-Conrady (medidas por los autores del dataset, como en el nivel 04).
  2. TIMESTAMPS reales: la cámara no captura a fps constante. El archivo
     rgb.txt dice EN QUÉ instante se capturó cada imagen.
  3. GROUND TRUTH a otra frecuencia: la verdad viene de un sistema de captura
     de movimiento (mocap) a ~100 Hz, el RGB va a ~30 Hz. Para comparar hay
     que ASOCIAR cada frame con la medida de mocap más cercana en el tiempo.

─── La matemática: asociación por timestamp ──────────────────────────────────
Dos relojes muestreando el mismo recorrido a frecuencias distintas. Para cada
t del RGB se busca el t' del mocap con |t − t'| mínimo (búsqueda binaria sobre
los timestamps ordenados) y se acepta sólo si |t − t'| <= max_dt. Con mocap a
100 Hz casi todo frame tiene pareja a < 10 ms; los frames sin pareja se
excluyen SOLO de la evaluación, no del tracking.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Tuple

import cv2
import numpy as np

# Intrínsecos + distorsión Brown-Conrady de las cámaras de TUM RGB-D
# (https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats).
# Formato: (fx, fy, cx, cy, k1, k2, p1, p2, k3). Todas son 640x480.
# fr3 se distribuye ya rectificada (dist = 0): es la calibración "ROS default"
# — y parte del porqué fr3_long deriva en el repo padre (su lección 28).
INTRINSECOS_TUM = {
    "freiburg1": (517.306408, 516.469215, 318.643040, 255.313989,
                  0.262383, -0.953104, -0.005358, 0.002628, 1.163314),
    "freiburg2": (520.908620, 521.007327, 325.141442, 249.701764,
                  0.231222, -0.784899, -0.003257, -0.000105, 0.917205),
    "freiburg3": (535.4, 539.2, 320.1, 247.6, 0.0, 0.0, 0.0, 0.0, 0.0),
}


def camara_tum(nombre_secuencia: str) -> Tuple[np.ndarray, np.ndarray]:
    """(K 3x3, dist (5,)) de una secuencia TUM a partir de su nombre.

    Busca 'freiburgN' (o su abreviatura 'frN') en el nombre de la carpeta:
    'rgbd_dataset_freiburg2_xyz' -> la cámara de freiburg2.
    """
    for clave, p in INTRINSECOS_TUM.items():
        if clave in nombre_secuencia or clave.replace("freiburg", "fr") in nombre_secuencia:
            K = np.array([[p[0], 0.0, p[2]],
                          [0.0, p[1], p[3]],
                          [0.0, 0.0, 1.0]])
            return K, np.array(p[4:9])
    raise ValueError(f"No reconozco la camara TUM de '{nombre_secuencia}' "
                     f"(esperaba freiburg1/2/3 en el nombre)")


def _leer_indice_tum(path: Path) -> List[Tuple[float, str]]:
    """Lee un archivo 'timestamp ruta_relativa' de TUM (rgb.txt). Ignora #."""
    out = []
    for linea in path.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#"):
            continue
        ts, rel = linea.split()
        out.append((float(ts), rel))
    return out


class SecuenciaTUM:
    """Itera (timestamp, gris, profundidad_en_metros) sobre una secuencia TUM.

    A diferencia de una carpeta de imágenes numeradas, aquí manda rgb.txt:
    da el ORDEN y el INSTANTE real de captura de cada frame.

    ─── La profundidad del Kinect (lo nuevo de este nivel) ───
    depth.txt indexa PNGs de 16 bits SIN signo con la convención TUM:
    valor / 5000 = METROS (5000 = 1 m). El valor 0 significa SIN DATO — las
    sombras del proyector infrarrojo, superficies especulares o negras, y
    todo lo que caiga fuera del rango útil del sensor (~0.3-8 m).

    Y el detalle que muerde: RGB y profundidad son SENSORES DISTINTOS, no
    sincronizados. Cada frame RGB se asocia al mapa de profundidad más
    cercano en el tiempo (asociar_por_timestamp, max_dt = 50 ms); si no hay
    pareja, el frame se emite con profundidad None — el llamador decide (el
    driver de este nivel NO inicializa hasta el primer frame CON profundidad:
    el bug del mapa mixto, ver el README).
    """

    FACTOR_PROFUNDIDAD = 5000.0     # convención TUM: uint16 / 5000 = metros

    def __init__(self, root: str | Path, con_profundidad: bool = False,
                 max_dt_depth: float = 0.05) -> None:
        self.root = Path(root)
        rgb_txt = self.root / "rgb.txt"
        if not rgb_txt.is_file():
            raise FileNotFoundError(f"No existe {rgb_txt} (no parece una "
                                    "secuencia TUM; corre descarga_datos.py)")
        self.entradas = _leer_indice_tum(rgb_txt)
        if not self.entradas:
            raise FileNotFoundError(f"rgb.txt sin entradas en {self.root}")
        self.con_profundidad = con_profundidad
        self._depth_de: List = [None] * len(self.entradas)
        if con_profundidad:
            depth_entradas = _leer_indice_tum(self.root / "depth.txt")
            d_ts = np.array([t for t, _ in depth_entradas])
            asoc = asociar_por_timestamp(self.timestamps, d_ts,
                                         max_dt=max_dt_depth)
            for i, j in enumerate(asoc):
                if j >= 0:
                    self._depth_de[i] = depth_entradas[j][1]

    @property
    def timestamps(self) -> np.ndarray:
        return np.array([t for t, _ in self.entradas], dtype=np.float64)

    def __len__(self) -> int:
        return len(self.entradas)

    def __iter__(self) -> Iterator[Tuple]:
        for i, (ts, rel) in enumerate(self.entradas):
            gris = cv2.imread(str(self.root / rel), cv2.IMREAD_GRAYSCALE)
            if gris is None:
                raise IOError(f"No se pudo leer la imagen: {self.root / rel}")
            if not self.con_profundidad:
                yield ts, gris
                continue
            prof = None
            if self._depth_de[i] is not None:
                # IMREAD_UNCHANGED: sin él, OpenCV degradaría el PNG de 16
                # bits a 8 y la profundidad perdería toda su resolución.
                crudo = cv2.imread(str(self.root / self._depth_de[i]),
                                   cv2.IMREAD_UNCHANGED)
                if crudo is not None:
                    prof = crudo.astype(np.float32) / self.FACTOR_PROFUNDIDAD
            yield ts, gris, prof


def leer_trayectoria_tum(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Lee groundtruth.txt (t tx ty tz qx qy qz qw) -> (timestamps, posiciones)."""
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    return data[:, 0].astype(np.float64), data[:, 1:4].astype(np.float64)


def asociar_por_timestamp(ts_consulta: np.ndarray, ts_ref: np.ndarray,
                          max_dt: float = 0.02) -> np.ndarray:
    """Para cada t de `ts_consulta`, el índice del más cercano en `ts_ref`
    (o -1 si el más cercano dista más de `max_dt` segundos).

    Es la asociación estándar de TUM (rgb <-> mocap): teoría en la cabecera.
    """
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
