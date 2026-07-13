#!/usr/bin/env python3
"""
Nivel 09 — Triangulación
========================

Los matches del nivel 06, con las poses del nivel 07, se vuelven PUNTOS 3D:

    1. DLT: dos rayos que (casi) se cortan
    2. los tres filtros de calidad (quiralidad, reproyeccion, paralaje)
       y CUANTO tira cada uno
    3. el precio del paralaje: la incertidumbre en profundidad explota
       cuando el baseline es pequeno (medido, no contado)
    4. exportar el mapa a PLY

Uso:
    python 09_triangulacion.py [--par 0 6]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent
DATOS = AQUI / "data" / "secuencia"

RATIO = 0.75
REPROJ_THRESH_PX = 2.0
MIN_PARALLAX_DEG = 0.3


# ─────────────────────── carga (igual que el nivel 07) ───────────────────────

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
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def leer_gt() -> list[np.ndarray]:
    poses = []
    for line in (DATOS / "groundtruth.txt").read_text(encoding="utf-8").splitlines():
        v = [float(x) for x in line.split()]
        T = np.eye(4)
        T[:3, :3] = quaternion_to_rotation(np.array(v[4:8]))
        T[:3, 3] = v[1:4]
        poses.append(T)
    return poses


def invert_se3(T: np.ndarray) -> np.ndarray:
    """T⁻¹ = [[Rᵀ, −Rᵀ·t], [0, 1]] (nivel 03)."""
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def emparejar(gray_a, gray_b) -> tuple[np.ndarray, np.ndarray]:
    """ORB + ratio test (niveles 05-06)."""
    orb = cv2.ORB_create(nfeatures=2000)
    kps_a, desc_a = orb.detectAndCompute(gray_a, None)
    kps_b, desc_b = orb.detectAndCompute(gray_b, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    pares = bf.knnMatch(desc_a, desc_b, k=2)
    buenos = [m for m, n in pares if m.distance < RATIO * n.distance]
    return (np.float64([kps_a[m.queryIdx].pt for m in buenos]),
            np.float64([kps_b[m.trainIdx].pt for m in buenos]))


# ─────────────────────────── la triangulación ────────────────────────────────

def proyectar(K, T_w_c, pts_w) -> tuple[np.ndarray, np.ndarray]:
    """Proyecta puntos del mundo en una vista. Devuelve (uv, profundidad Z)."""
    T_c_w = invert_se3(T_w_c)
    pts_c = (T_c_w[:3, :3] @ pts_w.T).T + T_c_w[:3, 3]
    Z = pts_c[:, 2]
    uv = np.full((len(pts_w), 2), np.nan)
    ok = Z > 1e-6
    uv[ok] = np.stack([K[0, 0] * pts_c[ok, 0] / Z[ok] + K[0, 2],
                       K[1, 1] * pts_c[ok, 1] / Z[ok] + K[1, 2]], axis=1)
    return uv, Z


def triangular(K, T_w_c0, T_w_c1, pts0, pts1) -> tuple[np.ndarray, dict]:
    """DLT + los tres filtros. Devuelve (puntos_mundo, mascaras de cada filtro).

    ─── La matemática: DLT (Direct Linear Transform) ───
    Cada observación cumple  λ·x̂ = P·X̄  (X̄ homogéneo, P = K·T_c_w[:3]).
    El factor λ (la profundidad, desconocida) estorba: se elimina con el
    producto vectorial, porque x̂ × (P·X̄) = 0 si son paralelos:

        [x̂]ₓ · P · X̄ = 0

    De las 3 filas solo 2 son independientes. Con las DOS vistas se apilan
    4 ecuaciones homogéneas en las 4 incógnitas de X̄:

        A · X̄ = 0,   A ∈ R^{4x4}

    La solución no trivial es el vector singular de MENOR valor singular de
    A (el "casi núcleo": la dirección que A menos estira). Eso es lo que
    hace cv2.triangulatePoints. Es un mínimo cuadrático ALGEBRAICO, no
    geométrico — por eso hay que filtrar con la reproyección después.

    ─── La matemática: por qué el paralaje manda ───
    Deriva Z = f·B/d (disparidad d, baseline B): ∂Z/∂d = −f·B/d² = −Z²/(f·B).
    Un error de ε píxeles en la imagen mueve el punto ε·Z²/(f·B) metros: la
    incertidumbre crece con el CUADRADO de la profundidad e inversamente con
    el baseline. Sin baseline no hay 3D — sólo un rayo (nivel 02).
    """
    P0 = K @ invert_se3(T_w_c0)[:3]
    P1 = K @ invert_se3(T_w_c1)[:3]
    X_h = cv2.triangulatePoints(P0, P1, pts0.T, pts1.T)      # (4, N) homogéneo
    w = np.where(np.abs(X_h[3]) < 1e-12, 1e-12, X_h[3])
    pts_w = (X_h[:3] / w).T

    uv0, Z0 = proyectar(K, T_w_c0, pts_w)
    uv1, Z1 = proyectar(K, T_w_c1, pts_w)

    # Filtro 1 — QUIRALIDAD: delante de ambas camaras (la proyeccion pinhole
    # ni siquiera esta definida con Z <= 0).
    quiral = (Z0 > 1e-6) & (Z1 > 1e-6)

    # Filtro 2 — REPROYECCION: el punto debe EXPLICAR sus dos observaciones.
    err0 = np.linalg.norm(uv0 - pts0, axis=1)
    err1 = np.linalg.norm(uv1 - pts1, axis=1)
    reproj = np.nan_to_num(np.maximum(err0, err1), nan=np.inf) < REPROJ_THRESH_PX

    # Filtro 3 — PARALAJE: angulo entre los dos rayos de observacion.
    C0, C1 = T_w_c0[:3, 3], T_w_c1[:3, 3]
    r0, r1 = pts_w - C0, pts_w - C1
    cos = np.einsum("ij,ij->i", r0, r1) / (
        np.linalg.norm(r0, axis=1) * np.linalg.norm(r1, axis=1) + 1e-12)
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    paralaje = ang > MIN_PARALLAX_DEG

    return pts_w, {"quiralidad": quiral, "reproyeccion": reproj,
                   "paralaje": paralaje, "error_reproj": np.maximum(err0, err1),
                   "angulo": ang,
                   "valido": quiral & reproj & paralaje}


def guardar_ply(pts: np.ndarray, path: Path, colores: np.ndarray | None = None) -> None:
    """Nube de puntos en PLY ASCII (MeshLab, CloudCompare, el visor de Windows).

    El formato es tan simple que se escribe a mano: una cabecera que declara
    cuantos vertices y que propiedades trae cada uno, y luego una linea por
    punto. Nada de magia.
    """
    n = len(pts)
    cab = ["ply", "format ascii 1.0", f"element vertex {n}",
           "property float x", "property float y", "property float z"]
    if colores is not None:
        cab += ["property uchar red", "property uchar green", "property uchar blue"]
    cab.append("end_header")
    lineas = list(cab)
    for i in range(n):
        x, y, z = pts[i]
        if colores is None:
            lineas.append(f"{x:.6f} {y:.6f} {z:.6f}")
        else:
            c = colores[i]
            lineas.append(f"{x:.6f} {y:.6f} {z:.6f} {c[0]} {c[1]} {c[2]}")
    path.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Nivel 09: triangulacion")
    parser.add_argument("--par", type=int, nargs=2, default=[0, 6])
    args = parser.parse_args()

    ia, ib = args.par
    K = leer_calibracion()
    gt = leer_gt()
    salida = AQUI / "salida"
    salida.mkdir(exist_ok=True)

    gray_a, gray_b = cargar_frame(ia), cargar_frame(ib)
    pts0, pts1 = emparejar(gray_a, gray_b)

    # Usamos las poses del GROUND TRUTH: este nivel aisla la triangulacion.
    # (En el nivel 10 las poses vendran de tu propia estimacion, con su ruido.)
    T0, T1 = gt[ia], gt[ib]
    pts_w, m = triangular(K, T0, T1, pts0, pts1)

    print(f"1. DLT sobre el par {ia}->{ib}: {len(pts0)} matches triangulados")
    print(f"   (poses del ground truth: este nivel mide la TRIANGULACION sola)")

    # ── 2 · Los filtros, uno a uno ────────────────────────────────────────────
    print("\n2. Los tres filtros de calidad (cuanto descarta cada uno):")
    n = len(pts_w)
    for nombre in ("quiralidad", "reproyeccion", "paralaje"):
        pasan = int(m[nombre].sum())
        print(f"   {nombre:13s}: pasan {pasan:4d}/{n}  "
              f"(descarta {100*(1-pasan/n):4.1f}%)")
    n_val = int(m["valido"].sum())
    print(f"   {'TODOS':13s}: pasan {n_val:4d}/{n}  "
          f"(el mapa final: {100*n_val/n:.1f}%)")

    err = m["error_reproj"][m["valido"]]
    print(f"\n   error de reproyeccion de los supervivientes: "
          f"media {err.mean():.3f} px, max {err.max():.3f} px")
    print(f"   paralaje de los supervivientes: mediana "
          f"{np.median(m['angulo'][m['valido']]):.2f} grados")

    # ── 3 · El precio del paralaje: baseline pequeno = profundidad incierta ──
    #
    # No comparamos contra un "GT por punto" (no lo tenemos): medimos la
    # SENSIBILIDAD, que es lo que predice la formula. Re-triangulamos los
    # mismos matches con 0.5 px de ruido en la segunda vista y vemos cuanto
    # se mueve el punto 3D. dZ = eps * Z^2 / (f * B): debe caer con el
    # baseline B, y en la tabla se ve exactamente eso.
    print("\n3. El precio del paralaje: cuanto se mueve el punto 3D si mueves")
    print("   su pixel medio pixel (misma vista base, baselines crecientes):")
    print(f"   {'par':>9s} {'baseline':>9s} {'paralaje':>9s} {'desplaz. 3D':>12s}"
          f" {'filtro paralaje':>16s}")
    rng = np.random.default_rng(0)
    for jb in (ia + 1, ia + 2, ia + 6, ia + 15, ia + 30):
        if jb >= len(gt):
            continue
        qa, qb = emparejar(cargar_frame(ia), cargar_frame(jb))
        if len(qa) < 30:
            continue
        p_w, mm = triangular(K, gt[ia], gt[jb], qa, qb)
        base = float(np.linalg.norm(gt[jb][:3, 3] - gt[ia][:3, 3]))
        val = mm["valido"]
        # cuanto tira el filtro de paralaje EN ESTE par (su razon de ser)
        tirados = int((~mm["paralaje"]).sum())
        pct = 100.0 * tirados / len(qa)
        if val.sum() < 10:
            print(f"   {ia:3d}->{jb:3d} {100*base:8.1f}cm  (menos de 10 validos)")
            continue
        p_w2, _ = triangular(K, gt[ia], gt[jb], qa,
                             qb + rng.normal(0, 0.5, qb.shape))
        despl = np.linalg.norm(p_w2[val] - p_w[val], axis=1)
        print(f"   {ia:3d}->{jb:3d} {100*base:8.1f}cm "
              f"{np.median(mm['angulo'][val]):8.2f}d "
              f"{100*np.median(despl):9.1f} cm "
              f"{tirados:6d} ({pct:4.1f}%)")
    print("   Con 5 cm de baseline, MEDIO PIXEL son medio metro de error en")
    print("   profundidad; con 1.5 m de baseline, 3 cm. Y fijate en la ultima")
    print("   columna: el filtro de paralaje solo muerde en los pares cortos,")
    print("   que es donde nacen los puntos 'en el infinito' que hay que tirar.")

    # ── 4 · El mapa: PLY + planta ─────────────────────────────────────────────
    mapa = pts_w[m["valido"]]
    # color por profundidad, para que el PLY se lea bonito
    z = mapa[:, 2]
    t = np.clip((z - z.min()) / (z.ptp() + 1e-9), 0, 1)
    col = np.stack([(255 * t).astype(int), (80 + 100 * t).astype(int),
                    (255 * (1 - t)).astype(int)], axis=1)
    guardar_ply(mapa, salida / "mapa.ply", col)
    print(f"\n4. Mapa: {len(mapa)} puntos -> {salida / 'mapa.ply'}")
    print("   (abrelo en MeshLab / CloudCompare / el visor 3D de Windows)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 6))
        sc = ax.scatter(mapa[:, 0], mapa[:, 2], c=mapa[:, 2], s=6, cmap="viridis")
        ax.plot([T0[0, 3], T1[0, 3]], [T0[2, 3], T1[2, 3]], "r-o", ms=6,
                label="las dos camaras")
        ax.set_xlabel("x [m]"), ax.set_ylabel("z [m]")
        ax.set_title("El mapa en planta (los 3 planos de la escena)")
        ax.legend(), ax.grid(True, alpha=0.3), ax.axis("equal")
        fig.colorbar(sc, ax=ax, label="profundidad z [m]")
        fig.savefig(salida / "mapa_planta.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        print("[aviso] matplotlib no instalado: se omite mapa_planta.png")

    print("\nAhora corre `python verificacion.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
