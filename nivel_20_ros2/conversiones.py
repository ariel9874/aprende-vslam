"""La FRONTERA núcleo ↔ ROS: conversiones de convención (numpy puro).

La regla de oro de este nivel (la regla 4 del repo padre): la cáscara ROS no
importa nada del núcleo y el núcleo no sabe que ROS existe. TODO cambio de
convención vive aquí, en la frontera — nunca dentro del tracker.

─── La matemática: cambio de convención de ejes (POR CONJUGACIÓN) ────────────
El núcleo usa los ejes ÓPTICOS de OpenCV (nivel 02): +Z delante, +X derecha,
+Y abajo. ROS usa REP-103 para el cuerpo: +X delante, +Y izquierda, +Z
arriba. El cambio de base es la rotación fija

        R_bo = [[ 0,  0,  1],      x_body =  z_opt   (delante)
                [-1,  0,  0],      y_body = -x_opt   (izquierda)
                [ 0, -1,  0]]      z_body = -y_opt   (arriba)

Una pose T_w_c (óptica EN AMBOS LADOS: mundo óptico ← cámara óptica) se
convierte CONJUGANDO — rotando el frame del mundo Y el del cuerpo:

        T_map_base = R̃_bo · T_w_c · R̃_bo⁻¹

Conjugar preserva la estructura de GRUPO: las composiciones y los deltas se
convierten igual (conv(A·B) = conv(A)·conv(B)). Si solo rotaras un lado, los
ejes del mundo y del cuerpo quedarían inconsistentes — la trayectoria sale
"DE LADO" en RViz (el suelo en una pared). Es el bug clásico de todo primer
puente a ROS, y el examen de este nivel lo comete a propósito y lo MIDE.

─── La matemática: REP-105 (map → odom → base_link) ──────────────────────────
REP-105 impone una CADENA de frames, y TF exige que cada frame tenga UN solo
padre. La odometría publica odom→base_link (continua, deriva); el SLAM sabe
la verdad map→base_link... pero NO puede publicarla directa (base_link ya
tiene padre). Publica la CORRECCIÓN:

        T_map_odom = T_map_base · T_odom_base⁻¹

de modo que la cadena componga T_map_odom · T_odom_base = T_map_base. Sin
deriva, map→odom = identidad; cuando el SLAM corrige (bucle), lo que salta
es map→odom — y la pose odom→base_link sigue CONTINUA (los planificadores
locales viven de esa continuidad).
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np

# Óptico (OpenCV) → cuerpo (REP-103), embebida en SE(3).
R_BO = np.array([[0.0, 0.0, 1.0],
                 [-1.0, 0.0, 0.0],
                 [0.0, -1.0, 0.0]])
_T_BO = np.eye(4)
_T_BO[:3, :3] = R_BO
_T_OB = _T_BO.T                          # inversa (rotación pura)


def optico_a_rep103(T_w_c: np.ndarray) -> np.ndarray:
    """Pose óptica (núcleo) → REP-103 (ROS), por conjugación (teoría arriba)."""
    return _T_BO @ T_w_c @ _T_OB


def optico_a_rep103_MAL(T_w_c: np.ndarray) -> np.ndarray:
    """El BUG a propósito: rotar SOLO el lado del mundo. Existe únicamente
    para que el examen MIDA la inconsistencia (no lo uses jamás)."""
    return _T_BO @ T_w_c


def t_map_odom(T_map_base: np.ndarray, T_odom_base: np.ndarray) -> np.ndarray:
    """La corrección que publica el SLAM (REP-105, teoría arriba)."""
    return T_map_base @ np.linalg.inv(T_odom_base)


def rot_a_quat_xyzw(R: np.ndarray):
    """Rotación → cuaternión (x, y, z, w) — el ORDEN de geometry_msgs (¡no
    el (w,x,y,z) del nivel 03!). Fórmula de la traza con las 4 ramas
    numéricamente estables: la rama de la traza falla cerca de θ = π."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        y, z = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        y, z = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return float(x), float(y), float(z), float(w)


def quat_xyzw_a_rot(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = np.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
