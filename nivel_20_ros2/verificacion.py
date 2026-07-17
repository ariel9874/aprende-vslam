#!/usr/bin/env python3
"""Examen del nivel 20: la matemática de la frontera, SIN ROS.

La demo viva (RViz + nodos en Docker) es tuya y de tu pantalla; lo que un
examen SÍ puede verificar en cualquier máquina es la matemática que la hace
funcionar — y el bug clásico, cometido a propósito y MEDIDO:

  1. R_BO es una rotación propia y manda cada eje a donde REP-103 dice.
  2. La CONJUGACIÓN preserva la estructura de grupo: los deltas relativos
     de una trayectoria se convierten igual que las poses.
  3. EL BUG: rotar SOLO un lado deja al robot montado "de lado" (120° de
     error de actitud constante — la trayectoria en la pared de RViz).
  4. Cuaterniones xyzw: ida y vuelta por las 4 ramas (incluida θ ≈ π), y la
     trampa del ORDEN (geometry_msgs es xyzw; el nivel 03 usaba wxyz).
  5. REP-105: T_map_odom · T_odom_base == T_map_base, exacto; sin deriva,
     la corrección es la identidad.
  6. (opcional) Si el daemon de Docker está corriendo, comprueba que la
     imagen del nivel construye. Si no, se salta con instrucciones.

Uso:
    python verificacion.py [--docker]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

from conversiones import (R_BO, optico_a_rep103, optico_a_rep103_MAL,
                          quat_xyzw_a_rot, rot_a_quat_xyzw, t_map_odom)

fallos = []


def check(nombre: str, ok: bool, detalle: str = "") -> None:
    estado = "OK " if ok else "FALLO"
    print(f"  [{estado}] {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def rot_aleatoria(rng) -> np.ndarray:
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def pose(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker", action="store_true",
                        help="incluir el chequeo de build de la imagen")
    args = parser.parse_args()
    rng = np.random.default_rng(20)
    print("Verificando la frontera nucleo <-> ROS (sin ROS)\n")

    # ── Acto 1: R_BO ─────────────────────────────────────────────────────────
    print("[1/5] R_BO: la rotacion del cambio de convencion...")
    check("R_BO es ortonormal con det +1 (rotacion propia)",
          np.allclose(R_BO.T @ R_BO, np.eye(3)) and
          np.isclose(np.linalg.det(R_BO), 1.0), "RtR = I, det = +1")
    check("manda los ejes a REP-103: z_opt->x, -x_opt->y, -y_opt->z",
          np.allclose(R_BO @ [0, 0, 1], [1, 0, 0]) and
          np.allclose(R_BO @ [-1, 0, 0], [0, 1, 0]) and
          np.allclose(R_BO @ [0, -1, 0], [0, 0, 1]),
          "delante/izquierda/arriba del cuerpo")

    # ── Acto 2: la conjugacion preserva el grupo ─────────────────────────────
    print("\n[2/5] La conjugacion preserva la estructura de grupo...")
    tray = [pose(rot_aleatoria(rng), rng.normal(size=3)) for _ in range(20)]
    peor = 0.0
    for i in range(len(tray) - 1):
        delta_opt = np.linalg.inv(tray[i]) @ tray[i + 1]
        a = np.linalg.inv(optico_a_rep103(tray[i])) @ optico_a_rep103(tray[i + 1])
        b = optico_a_rep103(delta_opt)
        peor = max(peor, np.abs(a - b).max())
    check("los deltas relativos se convierten IGUAL que las poses",
          peor < 1e-12, f"dif maxima {peor:.1e} en 19 deltas "
          "(conv(A)^-1·conv(B) == conv(A^-1·B))")

    # ── Acto 3: el bug, medido ───────────────────────────────────────────────
    print("\n[3/5] El bug a proposito: rotar SOLO un lado...")
    # Una camara que mira al frente (R = I optica) y avanza por su eje +Z
    # con el suelo plano: en REP-103 el robot debe avanzar en +X con
    # actitud IDENTIDAD (ni roll ni pitch).
    recta = [pose(np.eye(3), [0, 0, float(z)]) for z in range(5)]
    bien = [optico_a_rep103(T) for T in recta]
    mal = [optico_a_rep103_MAL(T) for T in recta]
    ang_bien = max(np.degrees(np.arccos(np.clip(
        (np.trace(T[:3, :3]) - 1) / 2, -1, 1))) for T in bien)
    ang_mal = min(np.degrees(np.arccos(np.clip(
        (np.trace(T[:3, :3]) - 1) / 2, -1, 1))) for T in mal)
    check("bien conjugado: el robot avanza en +X con actitud identidad",
          ang_bien < 1e-9 and np.allclose(bien[-1][:3, 3], [4, 0, 0]),
          f"actitud {ang_bien:.1e} grados; posicion final {bien[-1][:3, 3]}")
    check("rotando UN solo lado: 120 grados de actitud FALSA constante",
          abs(ang_mal - 120.0) < 1e-6,
          f"{ang_mal:.1f} grados — el robot 'de lado' en RViz; la posicion "
          "parece bien y la actitud miente: el bug que no se ve en un plot 2D")

    # ── Acto 4: cuaterniones xyzw ────────────────────────────────────────────
    print("\n[4/5] Cuaterniones en orden xyzw (geometry_msgs)...")
    peor = 0.0
    for _ in range(50):
        R = rot_aleatoria(rng)
        peor = max(peor, np.abs(quat_xyzw_a_rot(*rot_a_quat_xyzw(R)) - R).max())
    # las ramas de theta ~ pi (la traza falla ahi):
    for eje in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                np.array([0, 0, 1.0])):
        c, s = np.cos(np.pi - 1e-7), np.sin(np.pi - 1e-7)
        x, y, z = eje
        Kx = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        R = np.eye(3) + s * Kx + (1 - c) * (Kx @ Kx)
        peor = max(peor, np.abs(quat_xyzw_a_rot(*rot_a_quat_xyzw(R)) - R).max())
    check("R -> q_xyzw -> R exacto (50 al azar + las 3 ramas de theta~pi)",
          peor < 1e-9, f"dif maxima {peor:.1e}")
    check("la trampa del ORDEN: la identidad es (0,0,0,1) en xyzw",
          rot_a_quat_xyzw(np.eye(3)) == (0.0, 0.0, 0.0, 1.0),
          "en wxyz (nivel 03) era (1,0,0,0): mezclarlos rota 180 grados")

    # ── Acto 5: REP-105 ──────────────────────────────────────────────────────
    print("\n[5/5] REP-105: quien publica que...")
    T_mb = pose(rot_aleatoria(rng), rng.normal(size=3))
    T_ob = pose(rot_aleatoria(rng), rng.normal(size=3))
    compuesta = t_map_odom(T_mb, T_ob) @ T_ob
    check("T_map_odom . T_odom_base == T_map_base (la cadena compone)",
          np.abs(compuesta - T_mb).max() < 1e-12,
          f"dif {np.abs(compuesta - T_mb).max():.1e} — el SLAM publica la "
          "CORRECCION, nunca map->base_link directo (base_link ya tiene padre)")
    check("sin deriva, la correccion es la identidad",
          np.abs(t_map_odom(T_mb, T_mb) - np.eye(4)).max() < 1e-12,
          "map->odom = I hasta que el SLAM corrige (y entonces SALTA el, "
          "no odom->base_link)")

    # ── Acto 6 (opcional): el contenedor ─────────────────────────────────────
    if args.docker:
        print("\n[6/6] Docker: construyendo la imagen del nivel...")
        r = subprocess.run(["docker", "build", "-t", "aprende-vslam-ros2", "."],
                           cwd=AQUI, capture_output=True, text=True)
        check("la imagen construye", r.returncode == 0,
              (r.stderr or r.stdout).strip().splitlines()[-1][:70]
              if (r.stderr or r.stdout) else "")
    else:
        print("\n[6/6] SALTADO: el chequeo del contenedor (pasa --docker con "
              "el daemon corriendo).\n      La demo viva: docker compose up, "
              "y rviz2 dentro (ver README).")

    print()
    if fallos:
        print(f"NIVEL 20: {len(fallos)} fallo(s): {', '.join(fallos)}")
        return 1
    print("NIVEL 20: VERIFICADO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
