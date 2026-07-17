"""Bundle Adjustment: refina poses de keyframes Y puntos 3D a la vez.

Es el refinador de oro de todo el SLAM geométrico. El PnP del nivel 10 estima
cada pose contra un mapa que da por bueno; la triangulación del nivel 09 crea
puntos dando las poses por buenas. El BA admite que TODO tiene ruido y busca
la configuración conjunta que mejor explica todas las observaciones.

─── La matemática: el problema ───────────────────────────────────────────────
    argmin_{T_k, X_p}  Σ_(k,p)  ρ( ‖ π(K, T_k⁻¹·X_p) − u_kp ‖² )

Observación (k, p): el keyframe k vio el punto p en el píxel u_kp. π es la
proyección pinhole (nivel 02) y ρ el kernel de Huber (los outliers empujan
linealmente en vez de cuadráticamente, así no secuestran la solución).

─── La matemática: jacobianos analíticos ─────────────────────────────────────
El residuo r = π(X_c) − u depende de la pose y del punto vía X_c = T_c_w·X_w.
Con la perturbación POR LA DERECHA, T_w_c ← T_w_c·Exp(δ), se tiene
T_c_w ← Exp(−δ)·T_c_w, y a primer orden X_c(δ) ≈ X_c − ρ − ω×X_c, es decir:

    ∂X_c/∂δ   = [ −I₃ | [X_c]_× ]              (3×6, con δ = [ρ, ω])
    ∂X_c/∂X_w = R_c_w                          (3×3)

    ∂π/∂X_c   = [[ fx/z,    0,  −fx·x/z² ],    (2×3; derivar u = fx·x/z + cx)
                 [    0, fy/z,  −fy·y/z² ]]

y por la regla de la cadena:
    J_pose  = ∂π · ∂X_c/∂δ    (2×6)
    J_punto = ∂π · R_c_w      (2×3)

─── La matemática: complemento de Schur ──────────────────────────────────────
Las ecuaciones normales tienen estructura de FLECHA:

    H·δ = g,   H = [[B, E], [Eᵀ, C]],   δ = [δ_cámaras, δ_puntos]

B (6K×6K) acopla cámaras entre sí; C es DIAGONAL POR BLOQUES 3×3 porque dos
puntos nunca comparten un residuo — los puntos sólo se hablan a través de las
cámaras. Eso permite marginalizar los puntos por casi nada:

    (B − E·C⁻¹·Eᵀ)·δ_c = g_c − E·C⁻¹·g_p       ← sistema reducido, ¡6K×6K!
    δ_p = C_p⁻¹·(g_p − E_pᵀ·δ_c)               ← retro-sustitución por punto

Este truco ES la razón de que el BA escale; g2o/Ceres/GTSAM viven de él.

─── La matemática: RGB-D como estéreo virtual (lo nuevo del nivel 15) ───────
El sensor mide z en cada píxel, pero meter z directo al costo mezcla unidades
(metros vs píxeles) y exigiría un σ_z aparte. El truco de ORB-SLAM2: convertir
la profundidad en la coordenada que MEDIRÍA una cámara derecha a baseline b:

    u_R = u − fx·b/z          (fx·b ≡ bf; la disparidad d = fx·b/z)

y extender el residuo de [u, v] a [u, v, u_R]. Todo queda en píxeles (misma
Huber, mismo Schur — sólo crecen las filas de los jacobianos), y el peso de
la profundidad decae con la distancia: ∂u_R/∂z = fx·b/z², exactamente el
inverso del ruido del sensor (que crece ~z² en el Kinect) — la física y la
geometría se cancelan en la dirección correcta. El residuo extra ancla la
ESTRUCTURA del mapa a la medición métrica EN CADA OBSERVACIÓN: sin él, el BA
sólo re-teje reproyecciones y la deriva métrica no tiene de dónde corregirse.
Una observación (3,) con u_R = NaN significa "este píxel no tenía z válida":
residuo 2D normal.

─── La matemática: el gauge monocular tiene 7 grados, no 6 ──────────────────
Fijar UNA cámara ancla la rotación y traslación globales (6 gdl), pero queda
un séptimo: la ESCALA. La familia X′ = s·X, C′_k = s·C_k (escalar la escena
alrededor de la cámara fija) deja TODOS los residuos intactos → H tiene un
espacio nulo de dimensión 1 y el optimizador se para en cualquier miembro de
la familia. La firma empírica es inconfundible: el error relativo residual
sale IDÉNTICO en poses y en puntos. Solución: fijar ≥ 2 cámaras con baseline
entre ellas (ORB-SLAM fija todas las de fuera de la ventana por esta razón).
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np

_EPS = 1e-8

Observacion = Tuple[int, int, np.ndarray]      # (kf_id, punto_id, pixel (2,)
                                               #  o (3,) = [u, v, u_R] RGB-D)


# ───────────────────── álgebra de Lie: lo justo que hace falta ───────────────

def hat(v: np.ndarray) -> np.ndarray:
    """Matriz antisimétrica [v]_× tal que [v]_×·u = v × u."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(omega: np.ndarray) -> np.ndarray:
    """Exp de SO(3): eje-ángulo (3,) -> matriz de rotación (Rodrigues).

    ─── La matemática ───
    Exp(ω) = I + sin θ·[k]_× + (1 − cos θ)·[k]_×²   con θ = ‖ω‖, k = ω/θ.
    Cerca de θ = 0 se usa la serie de Taylor (sin θ/θ → 1, (1−cos θ)/θ² → ½):
    la fórmula cerrada dividiría por cero.
    """
    theta = np.linalg.norm(omega)
    W = hat(omega)
    if theta < _EPS:
        return np.eye(3) + W + 0.5 * (W @ W)
    return (np.eye(3) + (np.sin(theta) / theta) * W
            + ((1.0 - np.cos(theta)) / theta ** 2) * (W @ W))


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Exp de SE(3): ξ = [ρ, ω] ∈ R⁶ -> matriz 4x4.

    ─── La matemática: por qué la traslación NO es sólo ρ ───
    Exp([ρ, ω]) = [[Exp(ω), V·ρ], [0, 1]]  con
        V = I + (1−cos θ)/θ²·[ω]_× + (θ − sin θ)/θ³·[ω]_×²
    V (el "jacobiano izquierdo") aparece porque al girar MIENTRAS avanzas, el
    camino recorrido se curva: V·ρ es el ARCO, no la cuerda. Con θ → 0, V → I
    y todo degenera suavemente al caso euclidiano.
    """
    rho, omega = np.asarray(xi[:3], float), np.asarray(xi[3:], float)
    theta = np.linalg.norm(omega)
    W = hat(omega)
    if theta < _EPS:
        V = np.eye(3) + 0.5 * W + (W @ W) / 6.0
    else:
        V = (np.eye(3) + ((1.0 - np.cos(theta)) / theta ** 2) * W
             + ((theta - np.sin(theta)) / theta ** 3) * (W @ W))
    T = np.eye(4)
    T[:3, :3] = so3_exp(omega)
    T[:3, 3] = V @ rho
    return T


def invert_se3(T: np.ndarray) -> np.ndarray:
    """T⁻¹ = [[Rᵀ, −Rᵀ·t], [0, 1]] (nivel 03)."""
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


# ────────────────────────── residuo y jacobianos ─────────────────────────────

def residual_and_jacobians(K, T_c_w, X_w, uv, bf=0.0):
    """Residuo y jacobianos de UNA observación: 2D monocular o 3D con la
    coordenada derecha virtual u_R (teoría en la cabecera; bf = fx·b).

    Devuelve None si el punto cae DETRÁS de la cámara (la proyección pinhole
    ni siquiera está definida con z <= 0).
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X_c = T_c_w[:3, :3] @ X_w + T_c_w[:3, 3]
    x, y, z = X_c
    if z < 1e-3:
        return None

    r = [fx * x / z + cx - uv[0], fy * y / z + cy - uv[1]]
    d_pi = [[fx / z, 0.0, -fx * x / z ** 2],
            [0.0, fy / z, -fy * y / z ** 2]]
    if bf > 0.0 and len(uv) == 3 and np.isfinite(uv[2]):
        # Fila esterea virtual: u_R = fx·x/z + cx − bf/z. Su derivada en z es
        # la de u MÁS bf/z² — el término que hace PESAR la profundidad.
        r.append(fx * x / z + cx - bf / z - uv[2])
        d_pi.append([fx / z, 0.0, (bf - fx * x) / z ** 2])
    r, d_pi = np.asarray(r), np.asarray(d_pi)
    J_pose = d_pi @ np.hstack([-np.eye(3), hat(X_c)])   # ∂X_c/∂δ = [−I | [X_c]ₓ]
    J_point = d_pi @ T_c_w[:3, :3]                      # ∂X_c/∂X_w = R_c_w
    return r, J_pose, J_point


# ──────────────────────────────── el BA ──────────────────────────────────────

def bundle_adjustment(
    K: np.ndarray,
    kf_poses: Dict[int, np.ndarray],
    points: Dict[int, np.ndarray],
    observations: List[Observacion],
    fixed_kfs: Set[int],
    iterations: int = 10,
    huber_px: float = 2.5,
    penalizar_detras: bool = True,
    min_obs: int = 2,
    historial: list | None = None,
    stereo_bf: float = 0.0,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Levenberg-Marquardt + Schur sobre poses y puntos (teoría en la cabecera).

    Args:
        kf_poses: {kf_id: T_w_c} — TODAS las poses (las fijas no se optimizan,
            pero sus observaciones SÍ restringen los puntos).
        points: {punto_id: (3,)} posiciones iniciales.
        observations: (kf_id, punto_id, pixel).
        fixed_kfs: ids anclados. En monocular deben ser >= 2 CON BASELINE
            entre ellas: con una sola, la escala queda libre (el gauge de
            7 gdl de la cabecera) y el optimizador deriva dentro de esa
            familia. El experimento 2 del script lo demuestra.
        penalizar_detras: si False, las observaciones de puntos detrás de la
            cámara se OMITEN del costo — el "agujero de costo" que le enseña
            al optimizador a hacer trampa (experimento 3). Ponerlo en False
            es didáctico, nunca correcto.
        min_obs: un punto con MENOS de min_obs observaciones se excluye del BA.
            No es una optimización: es CORRECCIÓN. Con una sola observación,
            J_punto (2×3) tiene rango 2 y su bloque C_p = JᵀJ (3×3) es
            SINGULAR — no hay nada que fije la profundidad y el punto se
            desliza libremente por su rayo (a donde lo lleve la regularización
            de LM: el experimento 5 lo mide). El repo padre lo descubrió
            porque su BA EMPEORABA el mapa hasta que lo añadió.
        historial: si se pasa una lista, se le añade el costo de cada iter.
        stereo_bf: fx·b de la cámara derecha VIRTUAL (teoría en la cabecera).
            0 = apagado (residuo 2D puro); con bf > 0, las observaciones (3,)
            con u_R finita añaden la fila métrica. Debe ser el MISMO bf con
            el que el tracker fabricó las u_R.
    """
    poses = {k: np.asarray(T, float).copy() for k, T in kf_poses.items()}
    pts = {p: np.asarray(x, float).copy() for p, x in points.items()}
    obs = [(k, p, np.asarray(uv, float)) for k, p, uv in observations
           if k in poses and p in pts]

    # Excluir los puntos poco observados (ver min_obs arriba). Se quedan con
    # su valor de entrada, sin tocar: mejor no optimizarlos que romperlos.
    if min_obs > 1:
        cuenta: Dict[int, int] = {}
        for k, p, _ in obs:
            cuenta[p] = cuenta.get(p, 0) + 1
        excluidos = {p for p in pts if cuenta.get(p, 0) < min_obs}
        if excluidos:
            pts_fuera = {p: pts[p] for p in excluidos}
            pts = {p: x for p, x in pts.items() if p not in excluidos}
            obs = [(k, p, uv) for k, p, uv in obs if p not in excluidos]
        else:
            pts_fuera = {}
    else:
        pts_fuera = {}

    free_cams = sorted(k for k in poses if k not in fixed_kfs)
    cam_idx = {k: 6 * n for n, k in enumerate(free_cams)}
    pt_list = sorted(pts)
    if not obs or not pt_list:
        return poses, {**pts, **pts_fuera}

    # AGUJERO DE COSTO (el bug que el repo padre cazó verificando este módulo,
    # en dos actos): (1) si las observaciones de un punto detrás de la cámara
    # se OMITEN del costo, el optimizador aprende la trampa — empujar un punto
    # conflictivo detrás de las cámaras BORRA sus residuos y "baja" el costo
    # (midieron puntos volando a 15 000 unidades). (2) Con una penalización
    # tímida (100 px) la trampa persiste refinada: un outlier de 113 px cuesta
    # 280 > 247, y sale rentable esconder el punto detrás de LA cámara del
    # outlier. Moraleja: la penalización debe superar el residuo físicamente
    # posible más grande (el orden de la diagonal de la imagen).
    PENALTY = huber_px * (2000.0 - 0.5 * huber_px)

    def costo(poses_, pts_) -> float:
        c = 0.0
        for k, p, uv in obs:
            out = residual_and_jacobians(K, invert_se3(poses_[k]), pts_[p], uv,
                                         stereo_bf)
            if out is None:
                c += PENALTY if penalizar_detras else 0.0    # <- el agujero
                continue
            e = float(np.linalg.norm(out[0]))
            # Costo de Huber: cuadrático hasta δ, LINEAL después.
            c += 0.5 * e ** 2 if e <= huber_px else huber_px * (e - 0.5 * huber_px)
        return c

    lam = 1e-4
    cost = costo(poses, pts)
    if historial is not None:
        historial.append(cost)
    n_c = 6 * len(free_cams)

    for _ in range(iterations):
        B = np.zeros((n_c, n_c))
        g_c = np.zeros(n_c)
        C: Dict[int, np.ndarray] = {p: np.zeros((3, 3)) for p in pt_list}
        g_p: Dict[int, np.ndarray] = {p: np.zeros(3) for p in pt_list}
        E: Dict[int, Dict[int, np.ndarray]] = {p: {} for p in pt_list}

        T_c_w = {k: invert_se3(T) for k, T in poses.items()}
        for k, p, uv in obs:
            out = residual_and_jacobians(K, T_c_w[k], pts[p], uv, stereo_bf)
            if out is None:
                continue
            r, J_pose, J_point = out
            # Peso IRLS de Huber: los outliers pesan menos, pero NO se anulan
            # (siguen empujando linealmente — ojo, Huber no es un rechazador).
            e = float(np.linalg.norm(r))
            w = 1.0 if e <= huber_px else huber_px / e

            C[p] += w * (J_point.T @ J_point)
            g_p[p] -= w * (J_point.T @ r)
            if k in cam_idx:
                i = cam_idx[k]
                B[i:i + 6, i:i + 6] += w * (J_pose.T @ J_pose)
                g_c[i:i + 6] -= w * (J_pose.T @ r)
                E[p].setdefault(k, np.zeros((6, 3)))
                E[p][k] += w * (J_pose.T @ J_point)

        # Amortiguación LM sobre ambas diagonales.
        B[np.diag_indices(n_c)] += lam * np.diag(B) + 1e-9
        C_inv = {p: np.linalg.inv(C[p] + lam * np.diag(np.diag(C[p]))
                                  + 1e-9 * np.eye(3)) for p in pt_list}

        # SCHUR: S = B − Σ_p E_p·C_p⁻¹·E_pᵀ. Fíjate en qué pares de cámaras
        # se tocan: sólo las que COMPARTEN un punto. La covisibilidad ES la
        # estructura de S (y es la razón de que el grafo de covisibilidad del
        # nivel 13 no sea un detalle de implementación, sino la geometría).
        S, rhs = B.copy(), g_c.copy()
        for p in pt_list:
            cams = list(E[p])
            for ka in cams:
                ia = cam_idx[ka]
                ECi = E[p][ka] @ C_inv[p]
                rhs[ia:ia + 6] -= ECi @ g_p[p]
                for kb in cams:
                    ib = cam_idx[kb]
                    S[ia:ia + 6, ib:ib + 6] -= ECi @ E[p][kb].T

        try:
            delta_c = np.linalg.solve(S, rhs) if n_c else np.zeros(0)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue

        # Retro-sustitución de los puntos + actualización de prueba.
        # OJO: las poses se actualizan POR LA DERECHA en la variedad,
        # T ← T·Exp(δ) — no se suman deltas a las matrices.
        trial_poses = {k: T.copy() for k, T in poses.items()}
        for k, i in cam_idx.items():
            trial_poses[k] = trial_poses[k] @ se3_exp(delta_c[i:i + 6])
        trial_pts = {}
        for p in pt_list:
            acc = g_p[p].copy()
            for k, blk in E[p].items():
                acc -= blk.T @ delta_c[cam_idx[k]:cam_idx[k] + 6]
            trial_pts[p] = pts[p] + C_inv[p] @ acc

        trial_cost = costo(trial_poses, trial_pts)
        if trial_cost < cost:      # paso aceptado: confiar mas (lam baja)
            poses, pts, cost = trial_poses, trial_pts, trial_cost
            lam = max(lam / 3.0, 1e-9)
        else:                      # paso rechazado: desconfiar (lam sube ->
            lam *= 5.0             # el paso se acorta y se parece al gradiente)
        if historial is not None:
            historial.append(cost)

    return poses, {**pts, **pts_fuera}     # los excluidos vuelven intactos
