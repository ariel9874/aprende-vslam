"""El SLAM del nivel 14, instrumentado para la INGENIERIA DE TIEMPO REAL.

El sistema no cambia: se le anaden (1) un PERFIL por etapas — de donde sale
el numero que manda en este nivel: ¿DONDE se va el tiempo? — y (2) dos
enchufes para las gemelas rapidas: `ba_fn` (el BA vectorizado de
ba_rapido.py) y `bow` (el reconocimiento de lugar sub-lineal de bow.py).

La leccion del padre (sus lecciones 30-34, la escalera 4.3 -> 46.7 fps):
PERFILAR PRIMERO — la intuicion se refuta (su docs/04 apostaba por cv2; el
perfil dijo BA 57% + matching guiado 37%, cv2 8%) — sustituir SOLO el punto
caliente, y verificar la gemela con un TEST DE EQUIVALENCIA. Aqui, el mismo
metodo con gemelas que corren en cualquier maquina (GTSAM y C++ pybind11,
las gemelas del padre, quedan como lectura y ejercicio: exigen toolchain).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from bundle_adjustment import bundle_adjustment
from lie import se3_inv, sim3_inv
from pose_graph import GrafoDePoses

# Tabla de popcount para distancia de Hamming vectorizada (nivel 06): el
# descriptor ORB son 32 bytes; la distancia es el número de BITS distintos.
_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


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


# ──────────────────────────────── el mapa ────────────────────────────────────

class Mapa:
    """Puntos 3D + descriptores + OBSERVACIONES (quién vio qué, y dónde).

    Además del listado plano de observaciones (lo que consume el BA), mantiene
    el índice `obs_de_kf`: qué puntos ve cada keyframe. Ese índice ES el grafo
    de covisibilidad — dos keyframes son covisibles si sus conjuntos se
    intersecan. En el nivel 13 la covisibilidad era el ejercicio estrella;
    aquí es parte del sistema, porque los datos reales la exigen (ver
    SLAM._kfs_locales).
    """

    def __init__(self) -> None:
        self.puntos: Dict[int, np.ndarray] = {}
        self.desc: Dict[int, np.ndarray] = {}
        self.obs: List[Tuple[int, int, np.ndarray]] = []   # (kf_id, pt_id, uv)
        self.obs_de_kf: Dict[int, set] = {}                # kf_id -> {pt_ids}
        self._next = 0

    def add_punto(self, X: np.ndarray, d: np.ndarray) -> int:
        pid = self._next
        self._next += 1
        self.puntos[pid] = np.asarray(X, float)
        self.desc[pid] = d
        return pid

    def add_obs(self, kf_id: int, pt_id: int, uv: np.ndarray) -> None:
        self.obs.append((kf_id, pt_id, np.asarray(uv, float)))
        self.obs_de_kf.setdefault(kf_id, set()).add(pt_id)

    def puntos_de_kfs(self, kf_ids: set) -> List[int]:
        """Los puntos vistos por ese conjunto de keyframes (el mapa LOCAL)."""
        out: set = set()
        for k in kf_ids:
            out |= self.obs_de_kf.get(k, set())
        return sorted(out)

    def covisibles(self, kf_id: int, min_compartidos: int = 15) -> set:
        """Keyframes que comparten >= min_compartidos puntos con `kf_id`."""
        mios = self.obs_de_kf.get(kf_id, set())
        return {k for k, suyos in self.obs_de_kf.items()
                if k != kf_id and len(mios & suyos) >= min_compartidos}

    def aplicar_similitud(self, S: np.ndarray) -> None:
        """Re-ancla TODOS los puntos con una similitud (tras un bucle Sim(3))."""
        for p, X in self.puntos.items():
            self.puntos[p] = S[:3, :3] @ X + S[:3, 3]

    def __len__(self) -> int:
        return len(self.puntos)


# ─────────────────────────────── el tracker ──────────────────────────────────

class SLAM:
    """INIT / TRACK / LOST + mapa + BA + bucle, con matching GUIADO (v. real)."""

    # ── umbrales (los del nivel 13, ajustados a 30 fps reales) ──
    RATIO = 0.75
    MIN_INIT_FLOW_PX = 20.0
    MIN_MAP_MATCHES = 30
    MIN_PNP_INLIERS = 15
    KF_MIN_GAP = 3
    KF_MAX_GAP = 15            # a 30 fps reales, medio segundo sin keyframe
    KF_INLIER_RATIO = 0.6
    KF_HEALTH_INLIERS = 45     # piso de salud (nivel 10). En datos reales fue
                               # una perilla sensible (leccion 21 del padre)...
                               # hasta el matching guiado: con inliers altos,
                               # 45 y 25 dan lo mismo (leccion 24). La cura
                               # era el matching, no el umbral.
    LOCAL_KFS = 5
    COVIS_MIN_PTS = 15         # puntos compartidos para ser "covisible"
    BA_WINDOW = 5
    BA_ITERS = 6

    # ── matching guiado (leccion 24 del padre; radios de ORB-SLAM) ──
    GUIDED_RADIUS_PX = 15.0    # ventana de busqueda alrededor de la proyeccion
    GUIDED_MAX_HAMMING = 64    # distancia ORB maxima aceptable

    # ── cierre de bucle: los tres filtros del nivel 13 ──
    LOOP_TEMPORAL_GAP = 15     # en KEYFRAMES (la trampa de unidades del n. 13)
    LOOP_MIN_MATCHES = 40      # sobre los LOOP_PR_DESC descriptores mas fuertes
    LOOP_MIN_INLIERS = 40
    LOOP_COOLDOWN = 8
    # Reconocimiento de lugar con los N descriptores MAS FUERTES por keyframe.
    # En el corredor del nivel 13 (18 KFs) la fuerza bruta con los 2000 era
    # gratis; aqui hay >100 KFs y crece O(n^2). Para VOTAR el candidato bastan
    # los mas fuertes; la verificacion geometrica (PnP) si usa todo. (La
    # solucion de verdad a escala es BoW: nivel 18.)
    LOOP_PR_DESC = 300

    GBA_ITERS = 50             # leccion 27: con 10 el BA global NO converge

    def __init__(self, K: np.ndarray, usar_ba: bool = True,
                 usar_bucle: bool = True, usar_guiado: bool = True,
                 ba_fn=None, bow=None) -> None:
        self.K = K
        self.usar_ba = usar_ba
        self.usar_bucle = usar_bucle
        self.usar_guiado = usar_guiado
        # Los ENCHUFES de las gemelas (este nivel): mismo contrato, otra
        # implementacion. El test de equivalencia autoriza el cambio.
        self._ba_fn = ba_fn or bundle_adjustment
        self._bow = bow
        self._bow_pendientes: List[Tuple[int, np.ndarray]] = []
        # El PERFIL: segundos acumulados por etapa (la tabla del nivel).
        self.perfil: Dict[str, float] = {}

        self.orb = cv2.ORB_create(nfeatures=2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.mapa = Mapa()

        self.T_w_c = np.eye(4)
        self.T_rel = np.eye(4)
        self.estado = "INIT"
        self._img_hw: Optional[Tuple[int, int]] = None

        self.kf_poses: Dict[int, np.ndarray] = {}
        self.kf_frame: Dict[int, int] = {}
        self._kf_db: List[dict] = []   # {id, kps, desc, desc_pr, pts}
        self._buffer = None
        self._frames_desde_kf = 0
        self._kf_inliers_ref = 0       # inliers al insertar el ultimo KF
        self._ultimo_bucle = -999
        self._frame = -1
        self.eventos_bucle: List[Tuple[int, int]] = []
        self.n_perdidos = 0
        self.inliers_hist: List[int] = []   # para medir la ablacion ON/OFF

    def _cronometrar(self, etapa: str, t0: float) -> None:
        self.perfil[etapa] = self.perfil.get(etapa, 0.0) \
            + (time.perf_counter() - t0)

    # ── matching ─────────────────────────────────────────────────────────────

    def _match(self, da, db) -> list:
        if da is None or db is None or len(da) < 2 or len(db) < 2:
            return []
        pares = self.bf.knnMatch(da, db, k=2)
        return [m for m, n in pares if m.distance < self.RATIO * n.distance]

    @staticmethod
    def _dist_hamming(uno: np.ndarray, varios: np.ndarray) -> np.ndarray:
        """Distancia de un descriptor ORB a un conjunto (K,): XOR + popcount."""
        xor = np.bitwise_xor(uno[None, :], varios)          # (K, 32)
        return _POPCOUNT8[xor].sum(axis=1).astype(np.int32)

    def _guided_match(self, kps, desc, T_pred, pts_mapa, desc_mapa) -> list:
        """Matching GUIADO por reproyección (la lección 24 del repo padre).

        Predicha la pose, cada punto del mapa local se PROYECTA a la imagen y
        se busca sólo entre los keypoints a < GUIDED_RADIUS_PX de su
        proyección. Asignación GREEDY por distancia ascendente: cada keypoint
        y cada punto se usan UNA vez (dos puntos peleando por el mismo
        keypoint envenenarían el PnP). Si el prior es malo (arranque, salto),
        el guiado rinde poco — y el llamador cae al matching global.

        Devuelve DMatch(queryIdx=keypoint, trainIdx=punto_local): la MISMA
        orientación que el matching global, para que el resto no cambie.
        """
        if len(kps) == 0 or len(pts_mapa) == 0:
            return []
        h, w = self._img_hw if self._img_hw else (480, 640)

        uv, Z = proyectar(self.K, T_pred, pts_mapa)
        visibles = (Z > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] < w) \
            & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        kp_xy = np.array([kp.pt for kp in kps], dtype=np.float64)
        r2 = self.GUIDED_RADIUS_PX ** 2

        candidatos = []      # (dist_hamming, idx_punto, idx_kp)
        for i in np.flatnonzero(visibles):
            cerca = np.flatnonzero(
                np.sum((kp_xy - uv[i]) ** 2, axis=1) <= r2)
            if not len(cerca):
                continue
            dists = self._dist_hamming(desc_mapa[i], desc[cerca])
            k = int(np.argmin(dists))
            if dists[k] <= self.GUIDED_MAX_HAMMING:
                candidatos.append((int(dists[k]), int(i), int(cerca[k])))

        candidatos.sort()
        kp_usado, pt_usado, out = set(), set(), []
        for d, i, j in candidatos:
            if i in pt_usado or j in kp_usado:
                continue
            pt_usado.add(i)
            kp_usado.add(j)
            out.append(cv2.DMatch(j, i, float(d)))
        return out

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

        # Todo punto nace con DOS observaciones (leccion del nivel 11).
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
            {"id": 0, "kps": kps0, "desc": desc0,
             "desc_pr": self._desc_fuertes(kps0, desc0), "pts": pid_por_kp0},
            {"id": 1, "kps": kps, "desc": desc,
             "desc_pr": self._desc_fuertes(kps, desc), "pts": pid_por_kp1},
        ]
        self.kf_frame[0] = max(self._frame - 1, 0)
        self.kf_frame[1] = self._frame
        self.T_w_c, self.T_rel = T1, T1
        self._frames_desde_kf = 0
        self._kf_inliers_ref = int(val.sum())
        self.estado = "TRACK"
        info.update(estado="INIT-OK", n_mapa=len(self.mapa))

    def _desc_fuertes(self, kps, desc) -> np.ndarray:
        """Los LOOP_PR_DESC descriptores de mayor respuesta (ver LOOP_PR_DESC)."""
        if len(kps) <= self.LOOP_PR_DESC:
            return desc
        orden = np.argsort([-kp.response for kp in kps])[:self.LOOP_PR_DESC]
        return desc[np.sort(orden)]

    # ── mapa local: recencia + COVISIBILIDAD ─────────────────────────────────

    def _kfs_locales(self) -> set:
        """Los últimos N keyframes MÁS los covisibles del último.

        En el nivel 13 bastaba la recencia (200 frames, un solo bucle). Aquí
        NO, y lo medimos construyendo este nivel: con recencia sola, al
        re-visitar una zona el mapa local EXCLUYE los puntos originales, el
        sistema triangula duplicados desplazados por la deriva, y la sesión
        entera de fr2_xyz (3669 frames) acaba en un mapa de 96 000 puntos y
        35 cm de ATE que ningún BA puede arreglar (dos copias coherentes del
        mismo mundo no se reconcilian optimizando). Es la lección 14 del repo
        padre, reproducida sin querer.

        La covisibilidad rompe el ciclo junto con el bucle: el cierre de
        bucle registra observaciones puente -> el keyframe actual se vuelve
        covisible con el segmento viejo -> sus puntos ORIGINALES entran al
        mapa local -> el tracking los re-observa en vez de duplicarlos. Cada
        re-visita se convierte en un cierre de bucle implícito.
        """
        recientes = sorted(self.kf_poses)[-self.LOCAL_KFS:]
        ids = set(recientes)
        if recientes:
            ids |= self.mapa.covisibles(recientes[-1], self.COVIS_MIN_PTS)
        return ids

    # ── TRACK ────────────────────────────────────────────────────────────────

    def _track(self, kps, desc, info) -> None:
        pids = self.mapa.puntos_de_kfs(self._kfs_locales())
        if len(pids) < self.MIN_MAP_MATCHES:
            return self._coast(info)

        desc_local = np.array([self.mapa.desc[p] for p in pids])
        pts_local = np.array([self.mapa.puntos[p] for p in pids])

        # MATCHING GUIADO con prior de velocidad constante; si rinde poco
        # (arranque, prior malo), CAER al matching global por descriptor.
        matches = []
        if self.usar_guiado:
            t0 = time.perf_counter()
            T_pred = self.T_w_c @ self.T_rel
            matches = self._guided_match(kps, desc, T_pred,
                                         pts_local, desc_local)
            self._cronometrar("matching guiado", t0)
            info["n_guiado"] = len(matches)
        if len(matches) < self.MIN_MAP_MATCHES:
            t0 = time.perf_counter()
            matches = self._match(desc, desc_local)
            self._cronometrar("matching global", t0)
        info["n_matches"] = len(matches)
        if len(matches) < self.MIN_MAP_MATCHES:
            return self._coast(info)

        pix = np.float64([kps[m.queryIdx].pt for m in matches])
        pts = pts_local[[m.trainIdx for m in matches]]

        t0 = time.perf_counter()
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts, pix, self.K, None, iterationsCount=200, reprojectionError=3.0,
            confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        self._cronometrar("PnP", t0)
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
        self.inliers_hist.append(len(inl))

        self._frames_desde_kf += 1
        # HAMBRE: los inliers caen respecto al ULTIMO keyframe (el mapa
        # visible se agota). El nivel 13 usaba inliers/matches, y aqui eso se
        # rompe: con covisibilidad el mapa local crece, el guiado propone mas
        # matches de los que el PnP puede confirmar, y el cociente se hunde
        # sin que el tracking este mal — medimos 721 keyframes en 3669 frames
        # (uno cada 3) y "bucles" contra el pasado inmediato. La referencia
        # honesta del hambre es el propio keyframe anterior (como el padre).
        hambre = (len(inl) < self.KF_INLIER_RATIO * self._kf_inliers_ref
                  or self._frames_desde_kf >= self.KF_MAX_GAP)
        if hambre and self._frames_desde_kf >= self.KF_MIN_GAP:
            # Las correspondencias inlier del PnP (keypoint <-> punto del
            # mapa) se pasan al keyframe: seran sus RE-observaciones.
            pares_mapa = [(matches[i].queryIdx, pids[matches[i].trainIdx])
                          for i in inl]
            self._insertar_kf(kps, desc, len(inl), info, pares_mapa)

    def _coast(self, info) -> None:
        """Sin pose fiable: velocidad constante (la reloc vive en el padre)."""
        self.T_w_c = self.T_w_c @ self.T_rel
        self.estado = "LOST"
        self.n_perdidos += 1
        info["estado"] = "LOST"

    # ── KEYFRAME ─────────────────────────────────────────────────────────────

    def _insertar_kf(self, kps, desc, n_inliers, info, pares_mapa) -> None:
        """Nuevo keyframe: primero OBSERVA, después triangula.

        ─── El cambio frente al nivel 13 (y por qué) ───
        El nivel 13 triangulaba TODOS los matches contra el keyframe anterior
        y sólo evitaba duplicar los puntos que ESE keyframe conocía. En una
        sesión real larga eso fabrica el mapa-espejismo: al re-visitar una
        zona, los puntos originales (de hace 200 frames) no están en el
        keyframe anterior, así que se re-triangulan como puntos NUEVOS,
        desplazados por la deriva. Lo medimos: 96 000 puntos (la escena tiene
        ~la sexta parte) y un ATE que el BA no puede bajar.

        La cura es la arquitectura del repo padre: el keyframe registra como
        observaciones las correspondencias que el PnP YA verificó (incluidos
        los puntos VIEJOS que la covisibilidad trajo al mapa local), y sólo
        triangula keypoints que quedaron SIN punto asignado. Observar antes
        que crear.
        """
        # PISO DE SALUD: nunca crear mapa desde una pose incierta (nivel 10).
        if n_inliers < self.KF_HEALTH_INLIERS:
            return

        kf_ant = self._kf_db[-1]
        T_ant = self.kf_poses[kf_ant["id"]]
        kf_id = max(self.kf_poses) + 1
        self.kf_poses[kf_id] = self.T_w_c.copy()

        # 1) RE-OBSERVACIONES: lo que el PnP trackeo, el keyframe lo observa.
        pid_por_kp, pids_vistos = {}, set()
        for kp_idx, pid in pares_mapa:
            if kp_idx in pid_por_kp or pid in pids_vistos:
                continue
            self.mapa.add_obs(kf_id, pid, np.float64(kps[kp_idx].pt))
            pid_por_kp[kp_idx] = pid
            pids_vistos.add(pid)

        # 2) PUNTOS NUEVOS: triangular contra el keyframe anterior SOLO los
        #    keypoints que siguen sin punto en el mapa.
        matches = [m for m in self._match(kf_ant["desc"], desc)
                   if m.trainIdx not in pid_por_kp]
        if matches:
            p0 = np.float64([kf_ant["kps"][m.queryIdx].pt for m in matches])
            p1 = np.float64([kps[m.trainIdx].pt for m in matches])
            pts, val = triangular(self.K, T_ant, self.T_w_c, p0, p1)
            for X, i in zip(pts[val], np.where(val)[0]):
                m = matches[i]
                # ¿El keypoint del KF anterior ya tenia punto? Re-observar.
                pid_viejo = kf_ant["pts"].get(m.queryIdx)
                if pid_viejo is not None:
                    if pid_viejo not in pids_vistos:
                        self.mapa.add_obs(kf_id, pid_viejo, np.float64(p1[i]))
                        pid_por_kp[m.trainIdx] = pid_viejo
                        pids_vistos.add(pid_viejo)
                    continue
                pid = self.mapa.add_punto(X, desc[m.trainIdx])
                self.mapa.add_obs(kf_ant["id"], pid, np.float64(p0[i]))
                self.mapa.add_obs(kf_id, pid, np.float64(p1[i]))
                pid_por_kp[m.trainIdx] = pid
                pids_vistos.add(pid)

        self._kf_db.append({"id": kf_id, "kps": kps, "desc": desc,
                            "desc_pr": self._desc_fuertes(kps, desc),
                            "pts": pid_por_kp})
        if self._bow is not None:
            self._bow_indexar(kf_id, self._kf_db[-1]["desc_pr"])
        self.kf_frame[kf_id] = self._frame
        self._frames_desde_kf = 0
        self._kf_inliers_ref = n_inliers
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
        pset = set(pids)
        obs = [(k, p, uv) for k, p, uv in self.mapa.obs
               if k in wset and p in pset]
        poses = {k: self.kf_poses[k] for k in ventana}
        pts = {p: self.mapa.puntos[p] for p in pids}
        fijas = set(ventana[:2])      # el gauge de 7 gdl (nivel 11)

        t0 = time.perf_counter()
        poses_opt, pts_opt = self._ba_fn(
            self.K, poses, pts, obs, fixed_kfs=fijas,
            iterations=self.BA_ITERS, huber_px=2.5)
        self._cronometrar("BA local", t0)

        for k, T in poses_opt.items():
            self.kf_poses[k] = T
        for p, X in pts_opt.items():
            self.mapa.puntos[p] = X
        self.T_w_c = self.kf_poses[ventana[-1]].copy()

    # ── BA GLOBAL offline (lecciones 26-27 del padre) ────────────────────────

    def global_bundle_adjustment(self, iterations: Optional[int] = None,
                                 historial: list | None = None) -> None:
        """UN bundle adjustment sobre TODO el mapa, al terminar la secuencia.

        Teoría en la cabecera del módulo. El llamador lo invoca una vez y
        luego lee `trayectoria_kfs()` — nunca se corre en caliente.
        """
        ids = sorted(self.kf_poses)
        if len(ids) < 3:
            return
        poses = {k: self.kf_poses[k] for k in ids}
        pts = dict(self.mapa.puntos)
        # GAUGE: se fijan los DOS keyframes mas viejos (baseline -> escala).
        poses_opt, pts_opt = self._ba_fn(
            self.K, poses, pts, self.mapa.obs, fixed_kfs=set(ids[:2]),
            iterations=iterations or self.GBA_ITERS, huber_px=2.5,
            historial=historial)
        for k, T in poses_opt.items():
            self.kf_poses[k] = T
        for p, X in pts_opt.items():
            self.mapa.puntos[p] = X

    # ── BoW: entrenamiento en sesion e indexado (gemela del nivel 18) ───────

    def _bow_indexar(self, kf_id: int, desc_pr: np.ndarray) -> None:
        """El vocabulario se entrena EN SESION con los primeros 5 keyframes;
        despues, cada keyframe nuevo se indexa al insertarse."""
        if not self._bow.entrenado:
            self._bow_pendientes.append((kf_id, desc_pr))
            if len(self._bow_pendientes) >= 5:
                self._bow.entrenar(
                    np.vstack([d for _, d in self._bow_pendientes]))
                for kid, d in self._bow_pendientes:
                    self._bow.indexar(kid, d)
                self._bow_pendientes = []
            return
        self._bow.indexar(kf_id, desc_pr)

    # ── CIERRE DE BUCLE (nivel 13) + observaciones puente ────────────────────

    def _buscar_bucle(self, kf_id, kps, desc, info) -> None:
        if kf_id - self._ultimo_bucle < self.LOOP_COOLDOWN:
            return

        # FILTRO 1 — temporal: solo keyframes LEJANOS (en KEYFRAMES).
        candidatos = [kf for kf in self._kf_db
                      if kf_id - kf["id"] >= self.LOOP_TEMPORAL_GAP]
        if not candidatos:
            return

        # FILTRO 2 — reconocimiento de lugar. Dos rutas con el MISMO contrato
        # (proponer un candidato; la geometria del filtro 3 decide):
        #   - fuerza bruta acotada (nivel 14): O(KFs) con constante alta;
        #   - BoW (bow.py, la gemela de este nivel): sub-lineal.
        t0 = time.perf_counter()
        desc_pr = self._desc_fuertes(kps, desc)
        ids_cand = {kf["id"] for kf in candidatos}
        mejor = None
        if self._bow is not None and self._bow.entrenado:
            for kid, _score in self._bow.consultar(desc_pr, top_k=3):
                if kid in ids_cand:
                    mejor = next(kf for kf in candidatos if kf["id"] == kid)
                    break
        else:
            mejor_n = 0
            for kf in candidatos:
                n = len(self._match(desc_pr, kf["desc_pr"]))
                if n > mejor_n:
                    mejor, mejor_n = kf, n
            if mejor_n < self.LOOP_MIN_MATCHES:
                mejor = None
        self._cronometrar("bucle: reconocimiento", t0)
        if mejor is None:
            return

        # FILTRO 3 — VERIFICACION GEOMETRICA con TODOS los descriptores:
        # matching completo contra el candidato y PnP contra sus puntos 3D.
        kf_viejo = mejor
        ms = self._match(desc, kf_viejo["desc"])
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

        # OBSERVACIONES PUENTE (leccion 26): el keyframe NUEVO observa los
        # puntos del segmento VIEJO. Son estas observaciones las que atan los
        # extremos del bucle cuando el BA global reparta la correccion — sin
        # ellas, el grafo corrige poses pero la escala intermedia queda mal.
        kf_nuevo = self._kf_db[-1]
        ya_vistos = self.mapa.obs_de_kf.get(kf_id, set())
        for idx in inl.ravel():
            m, pid = pares[idx]
            if pid in ya_vistos:
                continue                # el PnP del tracking ya lo registro
            self.mapa.add_obs(kf_id, pid, np.float64(kps[m.queryIdx].pt))
            kf_nuevo["pts"].setdefault(m.queryIdx, pid)

        info["loop"] = (kf_viejo["id"], kf_id)
        self.eventos_bucle.append((kf_viejo["id"], kf_id))
        self._ultimo_bucle = kf_id
        self._cerrar_bucle(kf_viejo["id"], kf_id, T_pnp)

    def _cerrar_bucle(self, id_viejo: int, id_nuevo: int, T_pnp) -> None:
        """Grafo Sim(3) con el segmento antiguo CONGELADO (nivel 12/13)."""
        ids = sorted(self.kf_poses)
        g = GrafoDePoses("sim3")
        for k in ids:
            g.add_pose(k, self.kf_poses[k], fixed=(k <= id_viejo))

        for a, b in zip(ids[:-1], ids[1:]):
            T_rel = se3_inv(self.kf_poses[a]) @ self.kf_poses[b]
            g.add_odometry(a, b, T_rel, np.eye(7) * 1e2)

        T_rel_loop = se3_inv(self.kf_poses[id_viejo]) @ T_pnp
        g.add_loop(id_viejo, id_nuevo, T_rel_loop, np.eye(7) * 1e4)

        res = g.optimize(iterations=15)

        S_ultimo = res[ids[-1]] @ sim3_inv(_a_sim3(self.kf_poses[ids[-1]]))
        for k in ids:
            self.kf_poses[k] = _quita_escala(res[k])
        self.mapa.aplicar_similitud(S_ultimo)
        self.T_w_c = self.kf_poses[ids[-1]].copy()

    # ── bucle principal ──────────────────────────────────────────────────────

    def procesar(self, gray: np.ndarray) -> Tuple[np.ndarray, dict]:
        self._frame += 1
        if self._img_hw is None:
            self._img_hw = gray.shape[:2]
        info = {"estado": self.estado, "n_matches": 0, "n_guiado": 0,
                "n_inliers": 0, "kf": False, "loop": None,
                "n_mapa": len(self.mapa)}
        t0 = time.perf_counter()
        kps, desc = self.orb.detectAndCompute(gray, None)
        self._cronometrar("frontend (ORB)", t0)
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

        La métrica honesta (nivel 13, lección 25 del padre): las poses online
        se congelan al emitirse; los keyframes SÍ se reescriben (BA, grafo,
        BA global) — su trayectoria final refleja el estado real del sistema.
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
