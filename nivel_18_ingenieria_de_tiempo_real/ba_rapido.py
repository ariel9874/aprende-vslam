"""La GEMELA rápida del bundle adjustment: misma matemática, sin bucles.

El BA didáctico (bundle_adjustment.py, nivel 11) evalúa residuo y jacobianos
observación POR observación en un bucle de Python — legible, y es lo que el
perfil de este nivel señala como el punto caliente (~60% del tiempo, igual
que en el repo padre). Esta gemela hace EXACTAMENTE los mismos cálculos en
lotes de NumPy: todas las observaciones a la vez.

─── El método (la lección del nivel) ─────────────────────────────────────────
1. Se sustituye SOLO el punto caliente. La interfaz es idéntica a
   `bundle_adjustment(...)`: se enchufa sin tocar el tracker.
2. La GEMELA se verifica con un TEST DE EQUIVALENCIA (verificacion.py):
   mismo problema, mismos pasos LM, mismos resultados a tolerancia numérica.
   No "parece que funciona": DA LO MISMO. (No es bit a bit: sumar en otro
   orden cambia los últimos decimales del float — el test usa tolerancias.
   El repo padre llegó a verificar hasta la semántica de desempate de
   np.argmin entre su C++ y su Python: ese es el estándar.)
3. De dónde sale la velocidad: el intérprete de Python cuesta ~µs por
   operación; NumPy amortiza ese costo sobre miles de elementos por llamada.
   Mismos FLOPs, menos intérprete. (GTSAM/Ceres ganan otro orden de magnitud
   con C++ y factorización dispersa de verdad — ver el README.)
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np

from bundle_adjustment import invert_se3, se3_exp

Observacion = Tuple[int, int, np.ndarray]


def _hat_lote(v: np.ndarray) -> np.ndarray:
    """[v]_x para un lote (N, 3) -> (N, 3, 3)."""
    N = len(v)
    W = np.zeros((N, 3, 3))
    W[:, 0, 1], W[:, 0, 2] = -v[:, 2], v[:, 1]
    W[:, 1, 0], W[:, 1, 2] = v[:, 2], -v[:, 0]
    W[:, 2, 0], W[:, 2, 1] = -v[:, 1], v[:, 0]
    return W


def bundle_adjustment_rapido(
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
    """La misma optimización que `bundle_adjustment`, vectorizada.

    (stereo_bf se acepta por compatibilidad de firma pero esta gemela es la
    monocular: residuo 2D. La extensión métrica es el ejercicio 4.)
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    poses = {k: np.asarray(T, float).copy() for k, T in kf_poses.items()}
    pts = {p: np.asarray(x, float).copy() for p, x in points.items()}
    obs = [(k, p, np.asarray(uv[:2], float)) for k, p, uv in observations
           if k in poses and p in pts]

    # Exclusion de puntos poco observados (identica al didactico).
    if min_obs > 1:
        cuenta: Dict[int, int] = {}
        for k, p, _ in obs:
            cuenta[p] = cuenta.get(p, 0) + 1
        excluidos = {p for p in pts if cuenta.get(p, 0) < min_obs}
        pts_fuera = {p: pts[p] for p in excluidos}
        pts = {p: x for p, x in pts.items() if p not in excluidos}
        obs = [(k, p, uv) for k, p, uv in obs if p not in excluidos]
    else:
        pts_fuera = {}

    free_cams = sorted(k for k in poses if k not in fixed_kfs)
    cam_idx = {k: n for n, k in enumerate(free_cams)}
    pt_list = sorted(pts)
    pt_idx = {p: n for n, p in enumerate(pt_list)}
    if not obs or not pt_list:
        return poses, {**pts, **pts_fuera}

    todas_cams = sorted(poses)
    cam_pos = {k: n for n, k in enumerate(todas_cams)}

    # ── los datos, como ARRAYS de una vez y para siempre ──
    k_of = np.array([cam_pos[k] for k, _, _ in obs])          # (N,)
    esta_libre = np.array([k in cam_idx for k, _, _ in obs])  # (N,)
    kfree_of = np.array([cam_idx.get(k, -1) for k, _, _ in obs])
    p_of = np.array([pt_idx[p] for _, p, _ in obs])           # (N,)
    uv = np.stack([uvo for _, _, uvo in obs])                 # (N, 2)
    N = len(obs)

    PENALTY = huber_px * (2000.0 - 0.5 * huber_px)

    def preparar(poses_, pts_):
        """(N,3) puntos en camara + validez, TODO en lote."""
        Tcw = np.stack([invert_se3(poses_[k]) for k in todas_cams])
        R = Tcw[:, :3, :3][k_of]                              # (N, 3, 3)
        t = Tcw[:, :3, 3][k_of]                               # (N, 3)
        X = np.stack([pts_[p] for p in pt_list])[p_of]        # (N, 3)
        Xc = np.einsum("nij,nj->ni", R, X) + t
        return Xc, R, Xc[:, 2] > 1e-3

    def costo(poses_, pts_) -> float:
        Xc, _, ok = preparar(poses_, pts_)
        z = np.where(ok, Xc[:, 2], 1.0)
        r = np.stack([fx * Xc[:, 0] / z + cx - uv[:, 0],
                      fy * Xc[:, 1] / z + cy - uv[:, 1]], axis=1)
        e = np.linalg.norm(r, axis=1)
        c_h = np.where(e <= huber_px, 0.5 * e ** 2,
                       huber_px * (e - 0.5 * huber_px))
        pen = PENALTY if penalizar_detras else 0.0
        return float(np.where(ok, c_h, pen).sum())

    lam = 1e-4
    cost = costo(poses, pts)
    if historial is not None:
        historial.append(cost)
    n_c = 6 * len(free_cams)
    P = len(pt_list)

    for _ in range(iterations):
        Xc, Rcw, ok = preparar(poses, pts)
        z = np.where(ok, Xc[:, 2], 1.0)
        r = np.stack([fx * Xc[:, 0] / z + cx - uv[:, 0],
                      fy * Xc[:, 1] / z + cy - uv[:, 1]], axis=1)   # (N,2)
        d_pi = np.zeros((N, 2, 3))
        d_pi[:, 0, 0] = fx / z
        d_pi[:, 0, 2] = -fx * Xc[:, 0] / z ** 2
        d_pi[:, 1, 1] = fy / z
        d_pi[:, 1, 2] = -fy * Xc[:, 1] / z ** 2
        # ∂X_c/∂δ = [−I | [X_c]_x]  →  J_pose = d_pi @ (2,3)·(3,6)
        dXc = np.concatenate([-np.tile(np.eye(3), (N, 1, 1)),
                              _hat_lote(Xc)], axis=2)               # (N,3,6)
        J_pose = np.einsum("nij,njk->nik", d_pi, dXc)               # (N,2,6)
        J_pt = np.einsum("nij,njk->nik", d_pi, Rcw)                 # (N,2,3)

        e = np.linalg.norm(r, axis=1)
        w = np.where(e <= huber_px, 1.0, huber_px / np.maximum(e, 1e-12))
        w = np.where(ok, w, 0.0)                # detras de la camara: fuera

        # ── acumulacion por bloques con np.add.at (el nucleo del truco) ──
        wJp = J_pose * w[:, None, None]
        wJx = J_pt * w[:, None, None]
        C = np.zeros((P, 3, 3))
        g_p = np.zeros((P, 3))
        np.add.at(C, p_of, np.einsum("nij,nik->njk", wJx, J_pt))
        np.add.at(g_p, p_of, -np.einsum("nij,ni->nj", wJx, r))

        B = np.zeros((len(free_cams), 6, 6))
        g_c = np.zeros((len(free_cams), 6))
        m = esta_libre
        np.add.at(B, kfree_of[m], np.einsum("nij,nik->njk", wJp[m], J_pose[m]))
        np.add.at(g_c, kfree_of[m], -np.einsum("nij,ni->nj", wJp[m], r[m]))
        E_obs = np.einsum("nij,nik->njk", wJp, J_pt)                # (N,6,3)

        # Amortiguacion LM (identica al didactico).
        Bd = np.zeros((n_c, n_c))
        for n_, blk in enumerate(B):
            Bd[6 * n_:6 * n_ + 6, 6 * n_:6 * n_ + 6] = blk
        Bd[np.diag_indices(n_c)] += lam * np.diag(Bd) + 1e-9
        # lam * diag(C) por bloque, identico al didactico
        C_reg = C.copy()
        idx3 = np.arange(3)
        C_reg[:, idx3, idx3] += lam * C[:, idx3, idx3] + 1e-9
        C_inv = np.linalg.inv(C_reg)

        # ── SCHUR: agrupar las observaciones por punto ──
        orden = np.argsort(p_of, kind="stable")
        cortes = np.searchsorted(p_of[orden], np.arange(P + 1))
        S = Bd.copy()
        rhs = g_c.reshape(-1).copy()
        for n_p in range(P):
            grupo = orden[cortes[n_p]:cortes[n_p + 1]]
            if not len(grupo):
                continue
            libres = grupo[esta_libre[grupo]]
            if not len(libres):
                continue
            Ci = C_inv[n_p]
            Es = E_obs[libres]                       # (m, 6, 3)
            ks = kfree_of[libres]
            ECi = Es @ Ci                            # (m, 6, 3)
            aporta = np.einsum("mij,lkj->mlik", ECi, Es)   # (m, l, 6, 6)
            for a in range(len(libres)):
                ia = 6 * ks[a]
                rhs[ia:ia + 6] -= ECi[a] @ g_p[n_p]
                for b in range(len(libres)):
                    ib = 6 * ks[b]
                    S[ia:ia + 6, ib:ib + 6] -= aporta[a, b]

        try:
            delta_c = np.linalg.solve(S, rhs) if n_c else np.zeros(0)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue

        # Retro-sustitucion de puntos + actualizacion de prueba (por la
        # derecha en la variedad, identico al didactico).
        trial_poses = {k: T.copy() for k, T in poses.items()}
        for k, n_ in cam_idx.items():
            trial_poses[k] = trial_poses[k] @ se3_exp(delta_c[6 * n_:6 * n_ + 6])
        acc = g_p.copy()
        dC = delta_c.reshape(-1, 6) if n_c else np.zeros((0, 6))
        m = esta_libre
        np.add.at(acc, p_of[m],
                  -np.einsum("nji,nj->ni", E_obs[m], dC[kfree_of[m]]))
        nuevos = np.stack([pts[p] for p in pt_list]) \
            + np.einsum("pij,pj->pi", C_inv, acc)
        trial_pts = {p: nuevos[n_] for n_, p in enumerate(pt_list)}

        trial_cost = costo(trial_poses, trial_pts)
        if trial_cost < cost:
            poses, pts, cost = trial_poses, trial_pts, trial_cost
            lam = max(lam / 3.0, 1e-9)
        else:
            lam *= 5.0
        if historial is not None:
            historial.append(cost)

    return poses, {**pts, **pts_fuera}
