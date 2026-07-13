#!/usr/bin/env python3
"""
Nivel 03 — Poses y transformaciones
===================================

La cámara del nivel 02 por fin SE MUEVE: vuela en círculo alrededor del
cubo, mirándolo siempre. Tres experimentos:

    1. componer e invertir T_w_c (con la gimnasia de subindices)
    2. renderizar la orbita: los puntos del MUNDO al marco de la camara
    3. exportar la trayectoria en formato TUM (R <-> cuaternion, exacto)

Uso:
    python 03_poses.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent

# La misma camara del nivel 02 (TUM fr1).
FX, FY, CX, CY = 517.306408, 516.469215, 318.643040, 255.313989
W, H = 640, 480
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])

# El cubo del nivel 02, ahora QUIETO en el mundo (en el origen).
VERTICES = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
                    dtype=np.float64) * 0.5
ARISTAS = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7),
           (0, 4), (1, 5), (2, 6), (3, 7)]


# ─────────────────────────── el toolbox SE(3) ────────────────────────────────

def invert_se3(T: np.ndarray) -> np.ndarray:
    """Inversa cerrada de una transformación rígida 4x4.

    ─── La matemática: el grupo SE(3) ───
    Una pose es T = [[R, t], [0, 1]] con R ∈ SO(3) (RᵀR = I, det R = +1) y
    t ∈ ℝ³. Actúa sobre puntos como X' = R·X + t, y componer dos poses es
    multiplicar sus matrices (¡el orden importa: SE(3) no es conmutativo!).

    Para invertir, despeja X de X' = R·X + t:
        X = Rᵀ·X' − Rᵀ·t    ⇒    T⁻¹ = [[Rᵀ, −Rᵀ·t], [0, 1]]
    La forma cerrada es más barata que np.linalg.inv y garantiza que el
    resultado siga siendo exactamente rígido (Rᵀ es rotación perfecta;
    la inversa numérica genérica solo lo sería aproximadamente).

    Notación del curso: T_a_b lleva puntos del frame b al frame a. Los
    subíndices se encadenan "cancelándose", como unidades:
        T_w_c2 = T_w_c1 · T_c1_c2      (w←c1 por c1←c2 da w←c2)
    """
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def mirar_a(centro_camara: np.ndarray, objetivo: np.ndarray) -> np.ndarray:
    """Construye T_w_c de una cámara en `centro_camara` mirando a `objetivo`.

    ─── La matemática: una rotación es una BASE ortonormal ───
    Las columnas de R (en T_w_c) son los ejes de la cámara EXPRESADOS en el
    mundo. Basta construirlos: +Z apunta al objetivo (así es OpenCV: Z
    delante), +X se elige perpendicular a Z y al "arriba" del mundo (via
    producto cruz), +Y completa la terna (Y = Z x X: hacia abajo, OpenCV).
    Ortonormalidad por construcción — no hay ángulos de Euler que despejar.
    """
    z = objetivo - centro_camara
    z = z / np.linalg.norm(z)
    arriba_mundo = np.array([0.0, -1.0, 0.0])       # -Y del mundo = "cielo"
    x = np.cross(arriba_mundo, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], axis=1)          # columnas = ejes
    T[:3, 3] = centro_camara
    return T


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Matriz 3x3 -> cuaternión (qx, qy, qz, qw), método de Shepperd.

    ─── La matemática ───
    q = (v·sin(θ/2), cos(θ/2)) codifica girar θ alrededor del eje unitario v
    (q y −q son la MISMA rotación: doble cobertura de SO(3)). La conversión
    invierte dos identidades de Rodrigues: tr(R) = 1 + 2·cos(θ) fija el
    ángulo y R − Rᵀ = 2·sin(θ)·[v]ₓ fija el eje. Shepperd elige la rama que
    calcula primero la componente MAYOR de q (según la diagonal dominante)
    para no dividir nunca por algo pequeño: estable hasta en θ ≈ 0 y θ ≈ π.
    """
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


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """Cuaternión (qx, qy, qz, qw) -> matriz 3x3 (Rodrigues en términos de q)."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def guardar_tum(items: list[tuple[float, np.ndarray]], path: Path) -> None:
    """timestamp tx ty tz qx qy qz qw — una pose T_w_c por línea."""
    lines = []
    for t, T in items:
        q = rotation_to_quaternion(T[:3, :3])
        tx, ty, tz = T[:3, 3]
        lines.append(f"{t:.6f} {tx:.6f} {ty:.6f} {tz:.6f} "
                     f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def leer_tum(path: Path) -> list[tuple[float, np.ndarray]]:
    """El inverso exacto de guardar_tum."""
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        v = [float(x) for x in line.split()]
        T = np.eye(4)
        T[:3, :3] = quaternion_to_rotation(np.array(v[4:8]))
        T[:3, 3] = v[1:4]
        items.append((v[0], T))
    return items


# ────────────────────────── render con extrínsecos ───────────────────────────

def render_desde(T_w_c: np.ndarray) -> np.ndarray:
    """Renderiza el cubo DEL MUNDO visto desde la pose T_w_c.

    La cadena completa de un renderizador (y de un SLAM, leída al revés):
        X_c = T_c_w · X_w = invert_se3(T_w_c) · X_w      (mundo -> cámara)
        [u, v, 1]ᵀ ~ K · X_c / Z_c                        (cámara -> píxeles)
    Los subíndices cancelan: c←w por w = c. Si escribieras T_w_c · X_w
    (error clásico), estarías llevando puntos del marco CÁMARA al mundo.
    """
    T_c_w = invert_se3(T_w_c)
    Xw_h = np.hstack([VERTICES, np.ones((len(VERTICES), 1))])
    Xc = (Xw_h @ T_c_w.T)[:, :3]
    uv = np.stack([FX * Xc[:, 0] / Xc[:, 2] + CX,
                   FY * Xc[:, 1] / Xc[:, 2] + CY], axis=1).astype(int)
    img = np.zeros((H, W, 3), np.uint8)
    for i, j in ARISTAS:
        cv2.line(img, tuple(uv[i]), tuple(uv[j]), (0, 255, 0), 2, cv2.LINE_AA)
    return img


def main() -> int:
    salida = AQUI / "salida"
    salida.mkdir(exist_ok=True)

    # ── 1 · La gimnasia de subindices, con numeros ────────────────────────────
    print("1. Componer e invertir:")
    T_w_c1 = mirar_a(np.array([3.0, -0.5, 0.0]), np.zeros(3))
    T_w_c2 = mirar_a(np.array([0.0, -0.5, 3.0]), np.zeros(3))
    # El movimiento RELATIVO entre las dos vistas: c1 <- w por w <- c2.
    T_c1_c2 = invert_se3(T_w_c1) @ T_w_c2
    # ... y la cadena cierra: T_w_c2 reconstruida desde c1 y el relativo.
    err = np.abs(T_w_c1 @ T_c1_c2 - T_w_c2).max()
    print(f"   T_w_c2 == T_w_c1 @ T_c1_c2: error max {err:.2e}")
    ida_vuelta = np.abs(T_w_c1 @ invert_se3(T_w_c1) - np.eye(4)).max()
    print(f"   T @ T^-1 == I:              error max {ida_vuelta:.2e}")

    # ── 2 · La orbita: 60 poses mirando al cubo ───────────────────────────────
    n = 60
    trayectoria = []
    frames = []
    for k in range(n):
        ang = 2 * np.pi * k / n
        centro = np.array([3.0 * np.sin(ang), -0.8, 3.0 * np.cos(ang)])
        T_w_c = mirar_a(centro, np.zeros(3))
        trayectoria.append((k / 30.0, T_w_c))
        frames.append(render_desde(T_w_c))

    mosaico = np.vstack([np.hstack(frames[0:30:10] + [frames[30]]),
                         np.hstack(frames[30:60:10] + [frames[0]])])
    cv2.imwrite(str(salida / "orbita.png"), mosaico)
    vw = cv2.VideoWriter(str(salida / "orbita.avi"),
                         cv2.VideoWriter_fourcc(*"MJPG"), 24, (W, H))
    if vw.isOpened():
        for f in frames:
            vw.write(f)
        vw.release()
    print(f"\n2. Orbita de {n} poses renderizada (el cubo esta QUIETO: quien")
    print(f"   gira es la camara). {salida / 'orbita.avi'}")

    # La vuelta entera compuesta de relativos debe cerrar en la identidad.
    T_acum = np.eye(4)
    for k in range(n):
        T_a = trayectoria[k][1]
        T_b = trayectoria[(k + 1) % n][1]
        T_acum = T_acum @ (invert_se3(T_a) @ T_b)    # T_ck_ck+1 encadenadas
    err_bucle = np.abs(T_acum - np.eye(4)).max()
    print(f"   la vuelta entera compuesta de 60 relativos cierra en I: "
          f"error {err_bucle:.2e}")
    print("   (En el nivel 08, con poses ESTIMADAS, este mismo bucle NO va a")
    print("    cerrar: esa diferencia ES la deriva.)")

    # ── 3 · Formato TUM: guardar, releer, verificar ───────────────────────────
    guardar_tum(trayectoria, salida / "trayectoria.txt")
    releida = leer_tum(salida / "trayectoria.txt")
    err_io = max(np.abs(Ta - Tb).max()
                 for (_, Ta), (_, Tb) in zip(trayectoria, releida))
    print(f"\n3. Trayectoria TUM: {salida / 'trayectoria.txt'}")
    print(f"   guardar -> releer (R via cuaternion): error max {err_io:.2e}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        p = np.stack([T[:3, 3] for _, T in trayectoria])
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.plot(p[:, 0], p[:, 2], "-o", ms=3)
        ax.plot(0, 0, "ks", label="cubo")
        ax.set_xlabel("x [m]"), ax.set_ylabel("z [m]")
        ax.set_title("La orbita en planta")
        ax.axis("equal"), ax.grid(True, alpha=0.3), ax.legend()
        fig.savefig(salida / "trayectoria_planta.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        print("[aviso] matplotlib no instalado: se omite trayectoria_planta.png")

    print("\nAhora corre `python verificacion.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
