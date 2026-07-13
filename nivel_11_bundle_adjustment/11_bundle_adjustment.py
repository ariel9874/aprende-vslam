#!/usr/bin/env python3
"""
Nivel 11 — Bundle Adjustment
============================

Cinco experimentos sobre geometría sintética EXACTA (sabemos la verdad, así
que podemos medir el error de verdad, no sólo la reproyección):

    1. el BA converge: poses y puntos ruidosos -> la verdad
    2. EL GAUGE tiene 7 gdl: con 1 ancla la escala queda LIBRE; con 2, no
    3. el AGUJERO DE COSTO: omitir los puntos de detras ensena a hacer trampa
    4. HUBER no es un rechazador: degrada los outliers, no los anula
    5. un punto con UNA observacion se desliza por su rayo (C_p singular)

Uso:
    python 11_bundle_adjustment.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bundle_adjustment import (bundle_adjustment, invert_se3,
                               residual_and_jacobians, se3_exp)

AQUI = Path(__file__).resolve().parent

# La camara de siempre (TUM fr1).
K = np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]])


# ───────────────────────── la escena sintética ───────────────────────────────

def escena(n_kf: int = 5, n_pts: int = 60, seed: int = 7):
    """Cámaras en arco mirando a una nube de puntos. Verdad conocida.

    Devuelve (poses_gt, puntos_gt, observaciones) SIN ruido de píxel (el
    ruido se añade donde cada experimento lo necesite).
    """
    rng = np.random.default_rng(seed)
    pts = {i: np.array([rng.uniform(-1.2, 1.2), rng.uniform(-0.9, 0.9),
                        rng.uniform(4.0, 7.0)]) for i in range(n_pts)}

    poses = {}
    for k in range(n_kf):
        ang = -0.25 + 0.5 * k / max(n_kf - 1, 1)      # arco de ~30 grados
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.array([1.6 * np.sin(ang), 0.0, 0.3 * k])
        poses[k] = T

    obs = []
    for k, T in poses.items():
        T_c_w = invert_se3(T)
        for p, X in pts.items():
            Xc = T_c_w[:3, :3] @ X + T_c_w[:3, 3]
            if Xc[2] < 0.5:
                continue
            u = K[0, 0] * Xc[0] / Xc[2] + K[0, 2]
            v = K[1, 1] * Xc[1] / Xc[2] + K[1, 2]
            if 0 <= u < 640 and 0 <= v < 480:
                obs.append((k, p, np.array([u, v])))
    return poses, pts, obs


def semilla(gt_poses, gt_pts, rng, anclas=(0, 1), sigma_pose=0.02,
            sigma_pt=0.10):
    """La estimación inicial que un SLAM real le pasaría al BA.

    OJO con las ANCLAS: van en su valor VERDADERO, no perturbadas. En un SLAM
    real las poses de fuera de la ventana ya están optimizadas y se toman como
    referencia — el BA las cree. Si le clavas anclas MALAS, el BA no puede
    arreglarlo: reconstruye una escena consistente con un marco equivocado, y
    el error se amplifica por Z²/(f·B), la misma ley del nivel 09 (medido
    construyendo este nivel: 1 cm de error en las anclas -> 1 m en los puntos,
    con este baseline de 36 cm). Es el ejercicio 4.
    """
    poses = {k: (T.copy() if k in anclas else T @ se3_exp(rng.normal(0, sigma_pose, 6)))
             for k, T in gt_poses.items()}
    pts = {p: X + rng.normal(0, sigma_pt, 3) for p, X in gt_pts.items()}
    return poses, pts


def error_rmse(a: dict, b: dict) -> float:
    """RMSE entre dos diccionarios de posiciones (o de centros de cámara)."""
    d = [np.linalg.norm(np.asarray(a[k])[:3] - np.asarray(b[k])[:3]) for k in a]
    return float(np.sqrt(np.mean(np.square(d))))


def centros(poses: dict) -> dict:
    return {k: T[:3, 3] for k, T in poses.items()}


def reproy_media(K, poses, pts, obs) -> float:
    """Error de reproyección medio (px): la métrica que el BA minimiza."""
    errs = []
    for k, p, uv in obs:
        out = residual_and_jacobians(K, invert_se3(poses[k]), pts[p], uv)
        errs.append(np.linalg.norm(out[0]) if out is not None else 1e4)
    return float(np.mean(errs))


def escala_residual(x1: dict, gt_pts: dict) -> float:
    """Cuánto más grande es la nube estimada que la verdadera (mediana de
    razones de distancias entre pares: invariante a rotación y traslación)."""
    pares = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
    d_est = np.array([np.linalg.norm(x1[a] - x1[b]) for a, b in pares])
    d_gt = np.array([np.linalg.norm(gt_pts[a] - gt_pts[b]) for a, b in pares])
    return float(np.median(d_est / d_gt))


# ─────────────────────────── los experimentos ────────────────────────────────

def exp1_converge(salida: Path) -> tuple[float, float, float]:
    """El BA converge: baja la reproyeccion Y se acerca a la verdad."""
    salida.mkdir(exist_ok=True)     # el examen puede llamarnos sin main()
    rng = np.random.default_rng(1)
    gt_poses, gt_pts, obs = escena()
    obs_r = [(k, p, uv + rng.normal(0, 0.5, 2)) for k, p, uv in obs]  # 0.5 px
    poses0, pts0 = semilla(gt_poses, gt_pts, rng)

    hist: list[float] = []
    poses1, pts1 = bundle_adjustment(K, poses0, pts0, obs_r, fixed_kfs={0, 1},
                                     iterations=15, historial=hist)

    px0, px1 = reproy_media(K, poses0, pts0, obs_r), reproy_media(K, poses1, pts1, obs_r)
    ep0, ep1 = 100 * error_rmse(pts0, gt_pts), 100 * error_rmse(pts1, gt_pts)
    ec0 = 100 * error_rmse(centros(poses0), centros(gt_poses))
    ec1 = 100 * error_rmse(centros(poses1), centros(gt_poses))

    print(f"1. El BA converge (5 keyframes, 60 puntos, {len(obs)} observaciones,")
    print("   ruido de 0.5 px; las 2 anclas van en su valor verdadero):")
    print(f"   {'':22s} {'antes':>9s} {'despues':>9s}")
    print(f"   {'reproyeccion':22s} {px0:7.2f} px {px1:7.2f} px")
    print(f"   {'error de los puntos':22s} {ep0:7.1f} cm {ep1:7.1f} cm")
    print(f"   {'error de las camaras':22s} {ec0:7.1f} cm {ec1:7.1f} cm")
    print(f"   {'costo total':22s} {hist[0]:10.1f} {hist[-1]:10.1f}")
    print("   El residual de ~5 cm en los puntos NO es un fallo: es el suelo")
    print("   del ruido de pixel amplificado por Z^2/(f*B) (nivel 09). Con")
    print("   0.5 px y 36 cm de baseline, 5-7 cm es exactamente lo esperable.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.semilogy(hist, "o-")
        ax.set_xlabel("iteracion de LM"), ax.set_ylabel("costo (escala log)")
        ax.set_title("El BA converge")
        ax.grid(True, which="both", alpha=0.3)
        fig.savefig(salida / "convergencia.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass
    return px1, ep1, ec1


def exp2_gauge() -> tuple[float, float]:
    """EL experimento del nivel: el gauge monocular tiene 7 gdl, no 6.

    Sembramos la escena entera ESCALADA un 15% alrededor del centro de la
    cámara 0 — que es justo la dirección del espacio nulo. Las observaciones
    son perfectas (sin ruido), así que el único error posible es la escala.
    """
    gt_poses, gt_pts, obs = escena()
    C0 = gt_poses[0][:3, 3]
    s = 1.15

    poses0 = {}
    for k, T in gt_poses.items():
        T2 = T.copy()
        T2[:3, 3] = C0 + s * (T[:3, 3] - C0)          # escalar alrededor de C0
        poses0[k] = T2
    pts0 = {p: C0 + s * (X - C0) for p, X in gt_pts.items()}
    for k in (0, 1):                                   # las anclas, en la verdad
        poses0[k] = gt_poses[k].copy()

    print("\n2. El GAUGE (7 gdl): sembramos un error de ESCALA del 15% y")
    print("   dejamos que el BA lo arregle. Observaciones PERFECTAS.")
    print(f"   {'anclas':>10s} {'reproyeccion final':>20s} {'escala residual':>17s}")

    res = {}
    for nombre, fijas in [("1 camara", {0}), ("2 camaras", {0, 1})]:
        p1, x1 = bundle_adjustment(K, poses0, pts0, obs, fixed_kfs=fijas,
                                   iterations=30)
        esc = escala_residual(x1, gt_pts)
        px = reproy_media(K, p1, x1, obs)
        res[nombre] = esc
        print(f"   {nombre:>10s} {px:17.4f} px {esc:16.4f}")

    print("   Lee la tabla: con UNA ancla la reproyeccion es PERFECTA (0.0000")
    print("   px) y aun asi la escala se queda en 1.15: el error sembrado,")
    print("   INTACTO. Escalar la escena alrededor de la camara fija no mueve")
    print("   ni un pixel: el optimizador no tiene forma de saber que esta mal.")
    print("   Con DOS anclas, la distancia entre ellas fija el metro y la")
    print("   escala vuelve a 1.0000. Ese es el septimo grado de libertad.")
    return res["1 camara"], res["2 camaras"]


def exp3_agujero_de_costo() -> tuple[float, int, float, int]:
    """Omitir del costo los puntos de detras ENSENA al optimizador a esconderlos."""
    rng = np.random.default_rng(3)
    gt_poses, gt_pts, obs = escena()

    # Envenenamos 6 puntos con observaciones IMPOSIBLES: ningun punto 3D
    # explica esos pixeles. El optimizador tiene que decidir que hacer.
    obs2 = [(k, p, uv + rng.uniform(-150, 150, 2) if p < 6 else uv)
            for k, p, uv in obs]
    poses0, pts0 = semilla(gt_poses, gt_pts, rng, sigma_pose=0.01, sigma_pt=0.03)

    print("\n3. El AGUJERO DE COSTO (6 puntos con observaciones imposibles):")
    print(f"   {'los puntos de detras...':>26s} {'|X| max':>10s} {'detras':>8s} "
          f"{'err. del resto':>15s}")
    r = {}
    for nombre, penalizar in [("se OMITEN (el agujero)", False),
                              ("se PENALIZAN (correcto)", True)]:
        p1, x1 = bundle_adjustment(K, poses0, pts0, obs2, fixed_kfs={0, 1},
                                   iterations=15, penalizar_detras=penalizar)
        lejos = max(float(np.linalg.norm(X)) for X in x1.values())
        T_c_w = invert_se3(p1[0])
        detras = sum(1 for X in x1.values()
                     if (T_c_w[:3, :3] @ X + T_c_w[:3, 3])[2] < 0)
        # el dano colateral: cuanto empeoraron los puntos SANOS (los no
        # envenenados), que es lo que de verdad importa
        sanos = {p: x1[p] for p in x1 if p >= 6}
        err = 100 * error_rmse(sanos, {p: gt_pts[p] for p in sanos})
        r[nombre] = (lejos, detras, err)
        print(f"   {nombre:>26s} {lejos:8.0f} m {detras:8d} {err:12.1f} cm")

    print("   Los 6 puntos envenenados no tienen solucion posible, asi que")
    print("   vuelan lejos en AMBOS casos (ninguna posicion 3D explica sus")
    print("   pixeles). La diferencia esta en la columna 'detras': con el")
    print("   agujero, el optimizador DESCUBRE que empujarlos detras de la")
    print("   camara BORRA su residuo (no se evalua) y 'baja' el costo, asi")
    print("   que los esconde ahi. El repo padre midio puntos volando a 15 000")
    print("   unidades por esto. Un costo con agujeros no es un costo: es una")
    print("   trampa, y el optimizador SIEMPRE encuentra la trampa.")
    a = r["se OMITEN (el agujero)"]
    b = r["se PENALIZAN (correcto)"]
    return a[0], a[1], b[0], b[1]


def exp4_huber() -> tuple[float, float]:
    """Huber DEGRADA los outliers, no los anula: queda un sesgo proporcional."""
    rng = np.random.default_rng(4)
    gt_poses, gt_pts, obs = escena()

    # 10% de observaciones corrompidas en direccion ALEATORIA. (Un sesgo
    # CONSISTENTE desplazaria el minimo legitimamente y no probaria nada:
    # el detalle de diseno de test que el repo padre aprendio por las malas.)
    obs2 = []
    for k, p, uv in obs:
        if rng.random() < 0.10:
            d = rng.normal(size=2)
            uv = uv + 40.0 * d / np.linalg.norm(d)
        obs2.append((k, p, uv))
    poses0, pts0 = semilla(gt_poses, gt_pts, rng, sigma_pose=0.01, sigma_pt=0.03)

    print("\n4. HUBER no es un rechazador (10% de outliers de 40 px):")
    err = {}
    for nombre, huber in [("cuadratico puro", 1e9), ("Huber (2.5 px)", 2.5)]:
        p1, x1 = bundle_adjustment(K, poses0, pts0, obs2, fixed_kfs={0, 1},
                                   iterations=15, huber_px=huber)
        e = 100 * error_rmse(x1, gt_pts)
        err[nombre] = e
        print(f"   {nombre:>18s}: error de los puntos {e:7.2f} cm")
    mejora = 100 * (1 - err["Huber (2.5 px)"] / err["cuadratico puro"])
    print(f"   Huber mejora un {mejora:.0f}%, pero NO llega al error sin outliers")
    print("   (~5 cm, experimento 1): los outliers siguen empujando LINEALMENTE")
    print("   (peso huber/e: decae, pero nunca se anula). Para eliminarlos de")
    print("   verdad hace falta rechazarlos (RANSAC, test chi2). Huber los amansa.")
    return err["cuadratico puro"], err["Huber (2.5 px)"]


def exp5_una_observacion() -> tuple[float, float]:
    """Un punto con UNA sola observacion se desliza libremente por su rayo."""
    rng = np.random.default_rng(5)
    gt_poses, gt_pts, obs = escena()

    # Fabricamos 5 puntos "huerfanos": solo los ve la camara 2. En un SLAM
    # real pasa constantemente (un punto entra en campo justo al final).
    huerfanos = set(range(5))
    obs2 = [(k, p, uv) for k, p, uv in obs
            if p not in huerfanos or k == 2]
    poses0, pts0 = semilla(gt_poses, gt_pts, rng, sigma_pose=0.01, sigma_pt=0.05)

    print("\n5. El punto con UNA observacion (5 puntos vistos solo por la cam 2):")
    print(f"   {'':28s} {'error de los huerfanos':>24s}")
    r = {}
    for nombre, min_obs in [("SIN filtro (min_obs=1)", 1), ("CON filtro (min_obs=2)", 2)]:
        _, x1 = bundle_adjustment(K, poses0, pts0, obs2, fixed_kfs={0, 1},
                                  iterations=15, min_obs=min_obs)
        e = 100 * error_rmse({p: x1[p] for p in huerfanos},
                             {p: gt_pts[p] for p in huerfanos})
        r[nombre] = e
        print(f"   {nombre:>28s} {e:20.1f} cm")

    print("   LA MATEMATICA:")
    print("   Con UNA observacion, J_punto es 2x3 (dos ecuaciones, tres")
    print("   incognitas) y su bloque normal C_p = J^T J (3x3) tiene RANGO 2:")
    print("   es SINGULAR. El punto puede moverse a lo largo de su rayo sin")
    print("   cambiar el residuo. No hay nada que fije su profundidad, y la")
    print("   regularizacion de LM lo empuja a donde le da la gana. Excluirlo")
    print("   no es una optimizacion: es CORRECCION. El BA del repo padre")
    print("   EMPEORABA el mapa hasta que lo aprendieron (su leccion 7).")
    return r["SIN filtro (min_obs=1)"], r["CON filtro (min_obs=2)"]


def main() -> int:
    salida = AQUI / "salida"
    salida.mkdir(exist_ok=True)

    exp1_converge(salida)
    exp2_gauge()
    exp3_agujero_de_costo()
    exp4_huber()
    exp5_una_observacion()

    print(f"\nGrafica en {salida / 'convergencia.png'}")
    print("Ahora corre `python verificacion.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
