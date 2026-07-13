"""El SLAM completo: máquina de estados + mapa + backend.

Todo lo del curso, ensamblado. Es una versión RECORTADA del tracker del repo
padre (~850 líneas -> ~350): mismo esqueleto, sin las optimizaciones de
producción (matching guiado, BoW, hilo de mapeo, relocalización — están en el
padre y en los niveles 14/18).

─── La arquitectura (la de ORB-SLAM, en pequeño) ─────────────────────────────

    INIT   dos vistas con paralaje -> matriz esencial (nivel 07)
           -> triangular (nivel 09) -> gauge mediana = 1 (nivel 10)

    TRACK  matchear contra el MAPA LOCAL -> PnP (nivel 10)
           -> ¿toca keyframe? -> triangular puntos nuevos
                              -> BA de ventana (nivel 11)
                              -> ¿bucle? -> verificar -> grafo Sim(3) (nivel 12)

    LOST   sin pose fiable: velocidad constante ("coasting")

─── La matemática: por qué el mapa LOCAL ─────────────────────────────────────
Matchear contra TODO el mapa es caro Y peor: a escala real, un mapa de 10 000
puntos ORB tiene descriptores ambiguos y el matching global produce cientos de
correspondencias con CERO inliers geométricos (el repo padre lo midió: su
lección 22). El mapa local (los puntos de los últimos N keyframes) no es sólo
eficiencia: es CORRECCIÓN.

─── La matemática: el bucle se VERIFICA, no se cree ──────────────────────────
La lección del nivel 12: Huber no salva un falso positivo. Así que antes de
meter la arista al grafo:
  1. filtro temporal: sólo keyframes lejanos en el tiempo (si no, "cierras
     bucles" con tu propio vecino);
  2. matching de descriptores contra ese keyframe;
  3. VERIFICACIÓN GEOMÉTRICA: PnP contra sus puntos 3D. Si no hay ≥ 40
     inliers, no hay bucle. Punto.
Sólo lo que sobrevive a los tres pasos entra al grafo Sim(3).
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from bundle_adjustment import bundle_adjustment
from lie import se3_inv, sim3_inv
from pose_graph import GrafoDePoses


# ─────────────────────────── utilidades ──────────────────────────────────────

def proyectar(K, T_w_c, pts_w):
    """Puntos del mundo -> (uv, Z) en la vista T_w_c (nivel 09)."""
    T_c_w = se3_inv(T_w_c)
    pc = (T_c_w[:3, :3] @ pts_w.T).T + T_c_w[:3, 3]
    Z = pc[:, 2]
    uv = np.full((len(pts_w), 2), np.nan)
    ok = Z > 1e-6
    uv[ok] = np.stack([K[0, 0] * pc[ok, 0] / Z[ok] + K[0, 2],
                       K[1, 1] * pc[ok, 1] / Z[ok] + K[1, 2]], axis=1)
    return uv, Z


def triangular(K, T0, T1, p0, p1, reproj_px=2.0, min_par_deg=0.5):
    """DLT + los tres filtros del nivel 09."""
    P0 = K @ se3_inv(T0)[:3]
    P1 = K @ se3_inv(T1)[:3]
    Xh = cv2.triangulatePoints(P0, P1, p0.T, p1.T)
    w = np.where(np.abs(Xh[3]) < 1e-12, 1e-12, Xh[3])
    pts = (Xh[:3] / w).T

    uv0, Z0 = proyectar(K, T0, pts)
    uv1, Z1 = proyectar(K, T1, pts)
    err = np.maximum(np.linalg.norm(uv0 - p0, axis=1),
                     np.linalg.norm(uv1 - p1, axis=1))
    ok = (Z0 > 1e-6) & (Z1 > 1e-6) & (np.nan_to_num(err, nan=np.inf) < reproj_px)

    r0, r1 = pts - T0[:3, 3], pts - T1[:3, 3]
    cos = np.einsum("ij,ij->i", r0, r1) / (
        np.linalg.norm(r0, axis=1) * np.linalg.norm(r1, axis=1) + 1e-12)
    ok &= np.degrees(np.arccos(np.clip(cos, -1, 1))) > min_par_deg
    return pts, ok


def umeyama_escala(src: np.ndarray, dst: np.ndarray) -> float:
    """Escala de la similitud óptima src -> dst (sólo el factor s).

    La necesita el cierre de bucle: al re-visitar una zona, los puntos que el
    sistema triangula AHORA y los que trianguló ANTES son los mismos puntos
    físicos, pero con la escala derivada por el camino. Su razón es s_rel: lo
    que el grafo Sim(3) tiene que redistribuir (nivel 12).
    """
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    var = (sc ** 2).sum() / len(src)
    C = sc.T @ dc / len(src)
    U, D, Vt = np.linalg.svd(C)
    S = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return float(np.trace(np.diag(D) @ S) / max(var, 1e-12))


# ──────────────────────────────── el mapa ────────────────────────────────────

class Mapa:
    """Puntos 3D + descriptores + OBSERVACIONES (quién vio qué, y dónde).

    Las observaciones son lo que el BA necesita (nivel 11) y lo que define la
    covisibilidad. Un mapa sin observaciones es una nube de puntos, no un mapa.
    """

    def __init__(self) -> None:
        self.puntos: Dict[int, np.ndarray] = {}
        self.desc: Dict[int, np.ndarray] = {}
        self.obs: List[Tuple[int, int, np.ndarray]] = []   # (kf_id, pt_id, uv)
        self._next = 0

    def add_punto(self, X: np.ndarray, d: np.ndarray) -> int:
        pid = self._next
        self._next += 1
        self.puntos[pid] = np.asarray(X, float)
        self.desc[pid] = d
        return pid

    def add_obs(self, kf_id: int, pt_id: int, uv: np.ndarray) -> None:
        self.obs.append((kf_id, pt_id, np.asarray(uv, float)))

    def puntos_de_kfs(self, kf_ids: set) -> List[int]:
        """Los puntos vistos por ese conjunto de keyframes (el mapa LOCAL)."""
        return sorted({p for k, p, _ in self.obs if k in kf_ids})

    def aplicar_similitud(self, S: np.ndarray) -> None:
        """Re-ancla TODOS los puntos con una similitud (tras un bucle Sim(3)).

        x' = s·R·x + t. La escala vive en el MAPA, no en las poses: por eso
        tras corregir el grafo hay que arrastrar los puntos con la misma
        transformación, o el mapa y las poses dejan de hablar el mismo idioma.
        """
        for p, X in self.puntos.items():
            self.puntos[p] = S[:3, :3] @ X + S[:3, 3]

    def __len__(self) -> int:
        return len(self.puntos)


# ─────────────────────────────── el tracker ──────────────────────────────────

class SLAM:
    """Máquina de estados INIT / TRACK / LOST con mapa, BA y cierre de bucle."""

    # ── umbrales (didácticos: el repo padre los calibra por barrido medido) ──
    RATIO = 0.75
    MIN_INIT_FLOW_PX = 20.0    # sin paralaje no hay triangulacion (nivel 09)
    MIN_MAP_MATCHES = 30
    MIN_PNP_INLIERS = 15
    KF_MIN_GAP = 3             # ni pegados...
    KF_MAX_GAP = 12            # ...ni muertos de hambre (leccion 9 del padre)
    KF_INLIER_RATIO = 0.6
    KF_HEALTH_INLIERS = 45     # NUNCA crear mapa desde pose incierta (nivel 10)
    LOCAL_KFS = 5              # tamano del mapa local
    BA_WINDOW = 5              # keyframes que optimiza el BA de ventana
    BA_ITERS = 6
    # Cierre de bucle: los tres filtros (ver el docstring del modulo).
    # OJO a las UNIDADES: el gap se cuenta en KEYFRAMES, no en frames. La
    # secuencia tiene 200 frames pero solo ~18 keyframes; poner el gap en 20
    # (pensando en frames) hace que NINGUN candidato califique jamas y el
    # cierre de bucle no dispara nunca — en silencio. Pasa de verdad.
    LOOP_TEMPORAL_GAP = 8      # 1) no cerrar bucles con tu propio vecino
    LOOP_MIN_MATCHES = 60      # 2) matching de descriptores
    LOOP_MIN_INLIERS = 40      # 3) VERIFICACION GEOMETRICA (la que de verdad
                               #    protege: nivel 12, Huber no basta)
    LOOP_COOLDOWN = 5

    def __init__(self, K: np.ndarray, usar_ba: bool = True,
                 usar_bucle: bool = True) -> None:
        self.K = K
        self.usar_ba = usar_ba
        self.usar_bucle = usar_bucle

        self.orb = cv2.ORB_create(nfeatures=2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.mapa = Mapa()

        self.T_w_c = np.eye(4)
        self.T_rel = np.eye(4)
        self.estado = "INIT"

        self.kf_poses: Dict[int, np.ndarray] = {}
        self.kf_frame: Dict[int, int] = {}   # en QUE frame nacio cada keyframe
                                             # (lo necesita la metrica honesta:
                                             #  ver trayectoria_kfs)
        self._kf_db: List[dict] = []      # {id, kps, desc, pts (pid por kp)}
        self._buffer = None
        self._frames_desde_kf = 0
        self._ultimo_bucle = -999
        self._frame = -1                  # indice del frame actual
        self.eventos_bucle: List[Tuple[int, int]] = []
        self.n_perdidos = 0

    # ── matching ─────────────────────────────────────────────────────────────

    def _match(self, da, db) -> list:
        if da is None or db is None or len(da) < 2 or len(db) < 2:
            return []
        pares = self.bf.knnMatch(da, db, k=2)
        return [m for m, n in pares if m.distance < self.RATIO * n.distance]

    # ── INIT ─────────────────────────────────────────────────────────────────

    def _init(self, kps, desc, info) -> None:
        if self._buffer is None:
            self._buffer = (kps, desc)
            return
        kps0, desc0 = self._buffer
        matches = self._match(desc0, desc)
        if len(matches) < self.MIN_MAP_MATCHES:
            self._buffer = (kps, desc)
            return

        p0 = np.float64([kps0[m.queryIdx].pt for m in matches])
        p1 = np.float64([kps[m.trainIdx].pt for m in matches])
        if np.median(np.linalg.norm(p1 - p0, axis=1)) < self.MIN_INIT_FLOW_PX:
            return                                   # aun sin baseline: esperar

        E, mask = cv2.findEssentialMat(p0, p1, self.K, method=cv2.RANSAC,
                                       prob=0.999, threshold=1.0)
        if E is None or E.shape != (3, 3):
            self._buffer = (kps, desc)
            return
        n_inl, R, t, pose_mask, _ = cv2.recoverPose(
            E, p0, p1, self.K, distanceThresh=2000.0, mask=mask)
        if n_inl < self.MIN_PNP_INLIERS:
            self._buffer = (kps, desc)
            return

        T_c1_c0 = np.eye(4)
        T_c1_c0[:3, :3], T_c1_c0[:3, 3] = R, t.ravel()
        T0 = np.eye(4)                                # la 1a vista ES el mundo
        T1 = se3_inv(T_c1_c0)

        ok = pose_mask.ravel().astype(bool)
        pts, val = triangular(self.K, T0, T1, p0[ok], p1[ok])
        if val.sum() < self.MIN_PNP_INLIERS:
            self._buffer = (kps, desc)
            return

        # GAUGE monocular: profundidad mediana = 1.0 (nivel 10).
        _, Z = proyectar(self.K, T0, pts[val])
        s = 1.0 / float(np.median(Z))
        pts_esc = pts[val] * s
        T1[:3, 3] *= s

        # Registrar los dos keyframes fundacionales y sus observaciones.
        # OJO (leccion 7 del nivel 11): TODO punto nace con DOS observaciones
        # — ambos extremos de su triangulacion. Con una sola, el BA lo
        # deslizaria por su rayo.
        self.kf_poses[0], self.kf_poses[1] = T0, T1
        idx_ok = np.where(ok)[0][val]
        pid_por_kp0, pid_por_kp1 = {}, {}
        for X, i in zip(pts_esc, idx_ok):
            m = matches[i]
            pid = self.mapa.add_punto(X, desc[m.trainIdx])
            self.mapa.add_obs(0, pid, np.float64(kps0[m.queryIdx].pt))
            self.mapa.add_obs(1, pid, np.float64(kps[m.trainIdx].pt))
            pid_por_kp0[m.queryIdx] = pid
            pid_por_kp1[m.trainIdx] = pid

        self._kf_db = [
            {"id": 0, "kps": kps0, "desc": desc0, "pts": pid_por_kp0},
            {"id": 1, "kps": kps, "desc": desc, "pts": pid_por_kp1},
        ]
        self.kf_frame[0] = max(self._frame - 1, 0)   # el buffer era el anterior
        self.kf_frame[1] = self._frame
        self.T_w_c, self.T_rel = T1, T1
        self._frames_desde_kf = 0
        self.estado = "TRACK"
        info.update(estado="INIT-OK", n_mapa=len(self.mapa))

    # ── mapa local ───────────────────────────────────────────────────────────

    def _kfs_locales(self) -> set:
        """Los últimos N keyframes (mapa local por RECENCIA).

        El repo padre usa recencia UNION covisibilidad, que es estrictamente
        mejor (su lección 14: la recencia sola, al re-visitar una zona,
        EXCLUYE los puntos originales y el sistema crea duplicados desplazados
        por la deriva -> dos modos coherentes -> biestabilidad del PnP). Aquí
        se deja la recencia para que el código se lea, y la covisibilidad es
        el ejercicio estrella del nivel.
        """
        ids = sorted(self.kf_poses)[-self.LOCAL_KFS:]
        return set(ids)

    # ── TRACK ────────────────────────────────────────────────────────────────

    def _track(self, kps, desc, info) -> None:
        pids = self.mapa.puntos_de_kfs(self._kfs_locales())
        if len(pids) < self.MIN_MAP_MATCHES:
            return self._coast(info)

        desc_local = np.array([self.mapa.desc[p] for p in pids])
        matches = self._match(desc, desc_local)
        info["n_matches"] = len(matches)
        if len(matches) < self.MIN_MAP_MATCHES:
            return self._coast(info)

        pix = np.float64([kps[m.queryIdx].pt for m in matches])
        pts = np.array([self.mapa.puntos[pids[m.trainIdx]] for m in matches])

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts, pix, self.K, None, iterationsCount=200, reprojectionError=3.0,
            confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers) < self.MIN_PNP_INLIERS:
            return self._coast(info)

        inl = inliers.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(pts[inl], pix[inl], self.K, None,
                                          rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        T_c_w = np.eye(4)
        T_c_w[:3, :3], T_c_w[:3, 3] = R, tvec.ravel()
        T_nueva = se3_inv(T_c_w)

        self.T_rel = se3_inv(self.T_w_c) @ T_nueva
        self.T_w_c = T_nueva
        self.estado = "TRACK"
        info.update(estado="TRACK", n_inliers=len(inl))

        self._frames_desde_kf += 1
        ratio = len(inl) / max(len(matches), 1)
        hambre = (ratio < self.KF_INLIER_RATIO
                  or self._frames_desde_kf >= self.KF_MAX_GAP)
        if hambre and self._frames_desde_kf >= self.KF_MIN_GAP:
            self._insertar_kf(kps, desc, len(inl), info)

    def _coast(self, info) -> None:
        """Sin pose fiable: velocidad constante. (La relocalización, que es lo
        correcto, vive en el repo padre y es el ejercicio 4 de este nivel.)"""
        self.T_w_c = self.T_w_c @ self.T_rel
        self.estado = "LOST"
        self.n_perdidos += 1
        info["estado"] = "LOST"

    # ── KEYFRAME ─────────────────────────────────────────────────────────────

    def _insertar_kf(self, kps, desc, n_inliers, info) -> None:
        # PISO DE SALUD: nunca crear mapa desde una pose incierta (nivel 10).
        if n_inliers < self.KF_HEALTH_INLIERS:
            return

        kf_ant = self._kf_db[-1]
        T_ant = self.kf_poses[kf_ant["id"]]
        kf_id = max(self.kf_poses) + 1

        matches = self._match(kf_ant["desc"], desc)
        if len(matches) < self.MIN_MAP_MATCHES:
            return
        p0 = np.float64([kf_ant["kps"][m.queryIdx].pt for m in matches])
        p1 = np.float64([kps[m.trainIdx].pt for m in matches])
        pts, val = triangular(self.K, T_ant, self.T_w_c, p0, p1)
        if val.sum() == 0:
            return

        self.kf_poses[kf_id] = self.T_w_c.copy()
        pid_por_kp = {}
        for X, i in zip(pts[val], np.where(val)[0]):
            m = matches[i]
            # ¿el keypoint del KF anterior ya tenia un punto? Entonces es una
            # RE-OBSERVACION, no un punto nuevo: registrarla (y no duplicar).
            pid_viejo = kf_ant["pts"].get(m.queryIdx)
            if pid_viejo is not None:
                self.mapa.add_obs(kf_id, pid_viejo, np.float64(kps[m.trainIdx].pt))
                pid_por_kp[m.trainIdx] = pid_viejo
                continue
            pid = self.mapa.add_punto(X, desc[m.trainIdx])
            self.mapa.add_obs(kf_ant["id"], pid, np.float64(p0[i]))
            self.mapa.add_obs(kf_id, pid, np.float64(p1[i]))
            pid_por_kp[m.trainIdx] = pid

        self._kf_db.append({"id": kf_id, "kps": kps, "desc": desc,
                            "pts": pid_por_kp})
        self.kf_frame[kf_id] = self._frame
        self._frames_desde_kf = 0
        info["kf"] = True
        info["n_mapa"] = len(self.mapa)

        if self.usar_ba:
            self._ba_local()
        if self.usar_bucle:
            self._buscar_bucle(kf_id, kps, desc, info)

    # ── BA de ventana (nivel 11) ─────────────────────────────────────────────

    def _ba_local(self) -> None:
        ventana = sorted(self.kf_poses)[-self.BA_WINDOW:]
        if len(ventana) < 3:
            return
        wset = set(ventana)
        pids = self.mapa.puntos_de_kfs(wset)
        if not pids:
            return
        obs = [(k, p, uv) for k, p, uv in self.mapa.obs
               if k in wset and p in set(pids)]
        poses = {k: self.kf_poses[k] for k in ventana}
        pts = {p: self.mapa.puntos[p] for p in pids}
        # GAUGE: anclar DOS keyframes (el de 7 gdl del nivel 11).
        fijas = set(ventana[:2])

        poses_opt, pts_opt = bundle_adjustment(
            self.K, poses, pts, obs, fixed_kfs=fijas,
            iterations=self.BA_ITERS, huber_px=2.5)

        for k, T in poses_opt.items():
            self.kf_poses[k] = T
        for p, X in pts_opt.items():
            self.mapa.puntos[p] = X
        # La pose actual hereda la correccion del ultimo keyframe.
        self.T_w_c = self.kf_poses[ventana[-1]].copy()

    # ── CIERRE DE BUCLE (nivel 12), con verificación ─────────────────────────

    def _buscar_bucle(self, kf_id, kps, desc, info) -> None:
        if kf_id - self._ultimo_bucle < self.LOOP_COOLDOWN:
            return

        # FILTRO 1 — temporal: solo keyframes LEJANOS en el tiempo.
        candidatos = [kf for kf in self._kf_db
                      if kf_id - kf["id"] >= self.LOOP_TEMPORAL_GAP]
        if not candidatos:
            return

        mejor, mejor_n = None, 0
        for kf in candidatos:
            # FILTRO 2 — matching de descriptores.
            ms = self._match(desc, kf["desc"])
            if len(ms) > mejor_n:
                mejor, mejor_n = (kf, ms), len(ms)
        if mejor is None or mejor_n < self.LOOP_MIN_MATCHES:
            return

        kf_viejo, ms = mejor
        # FILTRO 3 — VERIFICACION GEOMETRICA (la que de verdad protege: sin
        # esto, un falso positivo destroza el grafo — nivel 12, medido).
        pares = [(m, kf_viejo["pts"].get(m.trainIdx)) for m in ms]
        pares = [(m, p) for m, p in pares if p is not None]
        if len(pares) < self.LOOP_MIN_INLIERS:
            return
        pix = np.float64([kps[m.queryIdx].pt for m, _ in pares])
        pts3 = np.array([self.mapa.puntos[p] for _, p in pares])

        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            pts3, pix, self.K, None, iterationsCount=300, reprojectionError=3.0,
            confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < self.LOOP_MIN_INLIERS:
            return                          # NO hay bucle. Y no se discute.

        R, _ = cv2.Rodrigues(rvec)
        T_c_w = np.eye(4)
        T_c_w[:3, :3], T_c_w[:3, 3] = R, tvec.ravel()
        T_pnp = se3_inv(T_c_w)              # donde el mapa VIEJO dice que estoy

        info["loop"] = (kf_viejo["id"], kf_id)
        self.eventos_bucle.append((kf_viejo["id"], kf_id))
        self._ultimo_bucle = kf_id
        self._cerrar_bucle(kf_viejo["id"], kf_id, T_pnp)

    def _cerrar_bucle(self, id_viejo: int, id_nuevo: int, T_pnp) -> None:
        """Grafo Sim(3) con el segmento antiguo CONGELADO (nivel 12).

        ¿Por qué congelar todo lo anterior al bucle, y no sólo el nodo 0? Porque
        las poses ya EMITIDAS viven en un marco concreto: si el grafo mueve la
        referencia, la historia que ya reportaste queda en otro sistema de
        coordenadas (el repo padre lo midió: 6.7 -> 87 cm de ATE).
        """
        ids = sorted(self.kf_poses)
        g = GrafoDePoses("sim3")
        for k in ids:
            g.add_pose(k, self.kf_poses[k], fixed=(k <= id_viejo))

        for a, b in zip(ids[:-1], ids[1:]):
            T_rel = se3_inv(self.kf_poses[a]) @ self.kf_poses[b]
            g.add_odometry(a, b, T_rel, np.eye(7) * 1e2)

        # La medida del bucle: la pose relativa que dice el PnP contra el mapa
        # viejo. Su escala relativa la mide Umeyama sobre los puntos que ambos
        # extremos ven (aqui, s = 1: en esta escena la deriva de escala es
        # pequena — en un recorrido largo NO lo seria, y ese es el 7o gdl).
        T_rel_loop = se3_inv(self.kf_poses[id_viejo]) @ T_pnp
        g.add_loop(id_viejo, id_nuevo, T_rel_loop, np.eye(7) * 1e4)

        res = g.optimize(iterations=15)

        # Aplicar: las poses vuelven a SE(3) (la escala vive en el MAPA, no en
        # las poses) y los puntos se arrastran con la similitud de su keyframe
        # ancla. Aqui, de forma simplificada, se re-ancla el mapa con la
        # similitud del ultimo keyframe corregido.
        S_ultimo = res[ids[-1]] @ sim3_inv(_a_sim3(self.kf_poses[ids[-1]]))
        for k in ids:
            self.kf_poses[k] = _quita_escala(res[k])
        self.mapa.aplicar_similitud(S_ultimo)
        self.T_w_c = self.kf_poses[ids[-1]].copy()

    # ── bucle principal ──────────────────────────────────────────────────────

    def procesar(self, gray: np.ndarray) -> Tuple[np.ndarray, dict]:
        self._frame += 1
        info = {"estado": self.estado, "n_matches": 0, "n_inliers": 0,
                "kf": False, "loop": None, "n_mapa": len(self.mapa)}
        kps, desc = self.orb.detectAndCompute(gray, None)
        if desc is None or len(kps) < 10:
            self._coast(info)
            return self.T_w_c, info

        if self.estado == "INIT":
            self._init(kps, desc, info)
        else:
            self._track(kps, desc, info)
        return self.T_w_c, info

    def trayectoria_kfs(self) -> Tuple[np.ndarray, np.ndarray]:
        """La trayectoria FINAL de keyframes: (indices de frame, posiciones).

        ─── LA MÉTRICA HONESTA (y por qué la online miente) ───
        Las poses que el sistema EMITE frame a frame se congelan al emitirse:
        cuando el cierre de bucle corrige el mapa en el frame 190, NADIE
        reescribe las poses ya reportadas de los frames 0..189. Evaluar sobre
        ellas es no ver NADA de lo que hace el backend — de hecho el ATE
        online puede EMPEORAR al activar el bucle (la corrección introduce un
        escalón en una trayectoria que ya no se puede arreglar).

        Los keyframes SÍ se reescriben (el BA y el grafo los tocan), así que
        su trayectoria final es la que refleja el estado real del sistema. Es
        la métrica que reporta ORB-SLAM. El repo padre la descubrió tarde y le
        costó cara (su lección 25): en fr2_desk, 21.9 cm online contra 4.8 cm
        en la trayectoria final de keyframes. El "problema de deriva" era en
        gran parte un artefacto de medición.
        """
        ids = sorted(self.kf_poses)
        frames = np.array([self.kf_frame[k] for k in ids])
        pos = np.stack([self.kf_poses[k][:3, 3] for k in ids])
        return frames, pos


def _a_sim3(T: np.ndarray) -> np.ndarray:
    """Una SE(3) es una Sim(3) con s = 1 (para poder componerlas)."""
    return T.copy()


def _quita_escala(S: np.ndarray) -> np.ndarray:
    """Sim(3) -> SE(3): la escala vive en el MAPA, las poses son rígidas."""
    s = float(np.linalg.det(S[:3, :3])) ** (1.0 / 3.0)
    T = np.eye(4)
    T[:3, :3] = S[:3, :3] / s
    T[:3, 3] = S[:3, 3]
    return T
