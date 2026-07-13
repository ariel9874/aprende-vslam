#!/usr/bin/env python3
"""Examen del nivel 03: SE(3) sin fisuras.

Si todo pasa: NIVEL 03: VERIFICADO.

Uso:
    python verificacion.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("n3", AQUI / "03_poses.py")
n3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n3)

fallos = []


def check(nombre: str, ok: bool, detalle: str = "") -> None:
    estado = "OK " if ok else "FALLO"
    print(f"  [{estado}] {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def main() -> int:
    print("Verificando el toolbox SE(3)\n")
    rng = np.random.default_rng(3)

    # Poses aleatorias legitimas (via cuaternion aleatorio normalizado).
    def pose_azar():
        T = np.eye(4)
        T[:3, :3] = n3.quaternion_to_rotation(rng.normal(size=4))
        T[:3, 3] = rng.normal(size=3)
        return T

    # 1. La inversa cerrada es exacta y rigida.
    errs = []
    for _ in range(20):
        T = pose_azar()
        errs.append(np.abs(T @ n3.invert_se3(T) - np.eye(4)).max())
    check("T @ invert_se3(T) == I (20 poses al azar)", max(errs) < 1e-12,
          f"err max {max(errs):.2e}")

    # 2. La cadena de subindices cierra: T_w_c2 == T_w_c1 @ (T_w_c1^-1 @ T_w_c2).
    Ta, Tb = pose_azar(), pose_azar()
    err = np.abs(Ta @ (n3.invert_se3(Ta) @ Tb) - Tb).max()
    check("la cadena T_w_c1 @ T_c1_c2 reconstruye T_w_c2", err < 1e-12,
          f"err {err:.2e}")

    # 3. SE(3) NO conmuta (si esto 'pasa', el alumno rompio algo).
    diff = np.abs(Ta @ Tb - Tb @ Ta).max()
    check("componer no conmuta (Ta@Tb != Tb@Ta)", diff > 1e-6,
          f"dif {diff:.3f}")

    # 4. R <-> cuaternion cierra exacto (incluye casos cerca de theta=pi,
    #    donde las formulas ingenuas cancelan — por eso Shepperd).
    peor = 0.0
    for _ in range(50):
        R = n3.quaternion_to_rotation(rng.normal(size=4))
        R2 = n3.quaternion_to_rotation(n3.rotation_to_quaternion(R))
        peor = max(peor, np.abs(R2 - R).max())
    casi_pi = n3.quaternion_to_rotation(np.array([0.9999, 0.0, 0.0, 0.0141]))
    peor = max(peor, np.abs(
        n3.quaternion_to_rotation(n3.rotation_to_quaternion(casi_pi)) - casi_pi).max())
    check("R -> q -> R exacto (50 al azar + theta~pi)", peor < 1e-9,
          f"err max {peor:.2e}")

    # 5. mirar_a produce rotaciones legitimas que de verdad miran al objetivo.
    T = n3.mirar_a(np.array([3.0, -0.5, 1.0]), np.zeros(3))
    R = T[:3, :3]
    orto = np.abs(R.T @ R - np.eye(3)).max()
    det = np.linalg.det(R)
    # el objetivo, visto desde la camara, debe caer sobre el eje +Z
    obj_cam = (n3.invert_se3(T) @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
    centrado = np.abs(obj_cam[:2]).max() < 1e-9 and obj_cam[2] > 0
    check("mirar_a: R ortonormal, det +1, objetivo sobre +Z",
          orto < 1e-12 and abs(det - 1) < 1e-12 and centrado,
          f"orto {orto:.1e}, det {det:.6f}, obj_cam {np.round(obj_cam, 6)}")

    # 6. TUM: guardar y releer reproduce las poses.
    salida = AQUI / "salida"
    salida.mkdir(exist_ok=True)
    tray = [(k * 0.1, pose_azar()) for k in range(10)]
    n3.guardar_tum(tray, salida / "_test_tum.txt")
    releida = n3.leer_tum(salida / "_test_tum.txt")
    err_io = max(np.abs(Ta - Tb).max() for (_, Ta), (_, Tb) in zip(tray, releida))
    (salida / "_test_tum.txt").unlink()
    check("guardar_tum -> leer_tum reproduce las poses", err_io < 1e-5,
          f"err max {err_io:.2e} (limitado por los 6 decimales del formato)")

    print()
    if fallos:
        print(f"NIVEL 03: {len(fallos)} fallo(s): {', '.join(fallos)}")
        return 1
    print("NIVEL 03: VERIFICADO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
