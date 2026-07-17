"""El SLAM del nivel 14, ahora en METROS: RGB-D y escala métrica.

Mismo esqueleto (INIT / TRACK / LOST, matching guiado, covisibilidad, bucle
verificado, BA global offline) con las TRES decisiones que cambia un sensor
de profundidad — cada una una lección medida del repo padre:

  1. INIT INSTANTÁNEA: el primer frame con profundidad YA es un mapa métrico
     (retro-proyección). Nada de esperar paralaje ni matriz esencial.
  2. El RESIDUO DE PROFUNDIDAD en el BA (estéreo virtual, [u, v, u_R]): el
     ancla métrica por observación (su lección 36 — es LA pieza que cruza
     fr1_desk: sin residuo 12.8 cm y 244 perdidos; con él, 2.8 cm y 0).
  3. El bucle métrico va en SE(3), NO en Sim(3) (su lección 35).

─── La matemática: la escala deja de ser gauge ───────────────────────────────
En monocular, (T, {X}) y (s·T, {s·X}) explican las mismas imágenes: la escala
es inobservable y se FIJA por convención (mediana = 1, nivel 10). La
profundidad lo cambia todo: z es una MEDICIÓN en metros, y la retro-proyección

    X_c = z · K⁻¹ · [u, v, 1]ᵀ

da puntos en la unidad del sensor. Consecuencia profunda (lección 35): en
monocular la deriva de escala es un grado de libertad que el bucle debe medir
y REDISTRIBUIR (Sim(3), Strasdat — nivel 12, sigue siendo correcto AHÍ). En
RGB-D la escala NO se negocia: un bucle Sim(3) re-escalaría el mapa viejo
mientras los puntos nuevos siguen naciendo métricos del sensor — el siguiente
bucle "corrige" la discrepancia que el anterior CREÓ, y el error se COMPONE
(el padre lo midió: s_rel de los bucles degenerando 1.0 -> 0.03, ATE 22.1 cm
con escala 2.09; en SE(3): 4.7 cm, escala 1.036). El grupo del bucle depende
de QUIÉN fija la escala.

Y el chequeo de honestidad de todo el nivel: el ATE se evalúa con alineación
RÍGIDA (sin regalar la escala al alineador), y la escala de similitud se
calcula APARTE — si el mapa está de verdad en metros, debe salir ≈ 1.000.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from bundle_adjustment import bundle_adjustment
from lie import se3_inv
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
        """Re-ancla TODOS los puntos con una transformación (aquí siempre
        RÍGIDA — el bucle métrico va en SE(3); ver SLAM._cerrar_bucle)."""
        for p, X in self.puntos.items():
            self.puntos[p] = S[:3, :3] @ X + S[:3, 3]

    def __len__(self) -> int:
        return len(self.puntos)


# ─────────────────────────────── el tracker ──────────────────────────────────

class SLAM:
    """INIT / TRACK / LOST + mapa + BA + bucle — todo en METROS (RGB-D)."""

    # ── umbrales (los del nivel 14) ──
    RATIO = 0.75
    MIN_MAP_MATCHES = 30
    MIN_PNP_INLIERS = 15
    MIN_INIT_POINTS = 50       # puntos con z valida para el mapa inicial
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

    # ── el sensor de profundidad (lo nuevo del nivel 15) ──
    DEPTH_MIN = 0.3            # rango util del Kinect (0.3-8 m); fuera de el
    DEPTH_MAX = 8.0            # la profundidad es ruido (o 0 = sin dato)
    DEPTH_MAX_NEW_POINTS = 400  # tope de puntos nuevos por KF desde profundidad
    STEREO_BF = 40.0           # fx·b de la camara derecha VIRTUAL (px·m):
                               # b = 40/517 ≈ 7.7 cm — un baseline plausible,
                               # el mismo del padre. Solo fija la CONVERSION
                               # z -> u_R; el BA usa el mismo numero.

    def __init__(self, K: np.ndarray, usar_ba: bool = True,
                 usar_bucle: bool = True, usar_guiado: bool = True,
                 usar_residuo: bool = True) -> None:
        self.K = K
        self.usar_ba = usar_ba
        self.usar_bucle = usar_bucle
        self.usar_guiado = usar_guiado
        # La ablacion del examen: bf = 0 apaga el residuo de profundidad en
        # el BA (la init sigue siendo metrica — se mide SOLO el residuo).
        self.residuo_bf = self.STEREO_BF if usar_residuo else 0.0

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
        self._depth = None
        self._frames_desde_kf = 0
        self._kf_inliers_ref = 0       # inliers al insertar el ultimo KF
        self._ultimo_bucle = -999
        self._frame = -1
        self.eventos_bucle: List[Tuple[int, int]] = []
        self.n_perdidos = 0
        self.inliers_hist: List[int] = []   # para medir la ablacion ON/OFF

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

    # ── INIT RGB-D: el primer frame con profundidad YA es un mapa ───────────

    def _init_rgbd(self, kps, desc, info) -> None:
        """Mapa MÉTRICO instantáneo por retro-proyección (teoría en cabecera).

        Comparado con la init monocular del nivel 13/14: nada de esperar
        paralaje, nada de matriz esencial, nada de gauge mediana = 1. Un solo
        frame, y en METROS. (Por eso RGB-D no sufre el fallo handheld de fr1:
        crear mapa no requiere baseline.) Los puntos nacen con UNA observación
        — no hay segunda vista — y el BA los excluye hasta que el tracking los
        re-observe (min_obs = 2, la lección del nivel 11).
        """
        prof = self._depth
        h, w = prof.shape
        pxs, zs, idxs = [], [], []
        for i, kp in enumerate(kps):
            u, v = int(round(kp.pt[0])), int(round(kp.pt[1]))
            if 0 <= u < w and 0 <= v < h:
                z = float(prof[v, u])
                if self.DEPTH_MIN < z < self.DEPTH_MAX:
                    pxs.append(kp.pt)
                    zs.append(z)
                    idxs.append(i)
        if len(idxs) < self.MIN_INIT_POINTS:
            return                       # profundidad insuficiente: reintentar

        # Retro-proyeccion: X_c = z·K⁻¹·[u,v,1]ᵀ. Con T0 = I (el mundo ES
        # esta camara), X_w = X_c.
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        px = np.float64(pxs)
        z = np.float64(zs)
        pts_w = np.stack([(px[:, 0] - cx) * z / fx,
                          (px[:, 1] - cy) * z / fy, z], axis=1)

        uv3 = self._with_virtual_right(px)     # [u, v, u_R]: la medicion
        pid_por_kp = {}
        for n, i in enumerate(idxs):
            pid = self.mapa.add_punto(pts_w[n], desc[i])
            self.mapa.add_obs(0, pid, uv3[n])
            pid_por_kp[i] = pid

        self.kf_poses[0] = np.eye(4)
        self.kf_frame[0] = self._frame
        self._kf_db = [{"id": 0, "kps": kps, "desc": desc,
                        "desc_pr": self._desc_fuertes(kps, desc),
                        "pts": pid_por_kp}]
        self.T_w_c, self.T_rel = np.eye(4), np.eye(4)
        self._frames_desde_kf = 0
        self._kf_inliers_ref = len(idxs)
        self.estado = "TRACK"
        info.update(estado="INIT-OK", n_mapa=len(self.mapa))

    def _with_virtual_right(self, px: np.ndarray) -> np.ndarray:
        """Píxeles (N, 2) -> (N, 3) añadiendo u_R = u − bf/z desde la
        profundidad del frame ACTUAL (teoría en bundle_adjustment.py).
        u_R = NaN donde no hay z válida: el BA cae al residuo 2D ahí."""
        px = np.asarray(px, np.float64).reshape(-1, 2)
        u_r = np.full(len(px), np.nan)
        if self.residuo_bf > 0.0 and self._depth is not None:
            prof = self._depth
            h, w = prof.shape
            for n, (u, v) in enumerate(px):
                ui, vi = int(round(u)), int(round(v))
                if 0 <= ui < w and 0 <= vi < h:
                    z = float(prof[vi, ui])
                    if self.DEPTH_MIN < z < self.DEPTH_MAX:
                        u_r[n] = u - self.residuo_bf / z
        return np.column_stack([px, u_r])

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
            T_pred = self.T_w_c @ self.T_rel
            matches = self._guided_match(kps, desc, T_pred,
                                         pts_local, desc_local)
            info["n_guiado"] = len(matches)
        if len(matches) < self.MIN_MAP_MATCHES:
            matches = self._match(desc, desc_local)
        info["n_matches"] = len(matches)
        if len(matches) < self.MIN_MAP_MATCHES:
            return self._coast(info)

        pix = np.float64([kps[m.queryIdx].pt for m in matches])
        pts = pts_local[[m.trainIdx for m in matches]]

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
        """Nuevo keyframe: primero OBSERVA, después RETRO-PROYECTA.

        La disciplina del nivel 14 se mantiene (observar antes que crear: las
        correspondencias que el PnP ya verificó se registran como
        observaciones, y sólo los keypoints SIN punto crean mapa). Lo que
        cambia con el sensor:

        1. Cada observación lleva su u_R = u − bf/z — TAMBIÉN las de puntos
           viejos re-observados: esa es la medición métrica fresca que ancla
           la estructura en el BA (la lección 36 del padre).
        2. Los puntos nuevos NO se triangulan: se RETRO-PROYECTAN de la
           profundidad. Métrico, sin baseline — por eso RGB-D no necesita
           paralaje para crear mapa. Nacen con UNA observación; el BA los
           excluye hasta la segunda (nivel 11).
        """
        # PISO DE SALUD: nunca crear mapa desde una pose incierta (nivel 10).
        if n_inliers < self.KF_HEALTH_INLIERS:
            return

        kf_id = max(self.kf_poses) + 1
        self.kf_poses[kf_id] = self.T_w_c.copy()

        # 1) RE-OBSERVACIONES con su medicion metrica u_R.
        pid_por_kp, pids_vistos = {}, set()
        pares = [(kp_idx, pid) for kp_idx, pid in pares_mapa
                 if kp_idx not in pid_por_kp and pid not in pids_vistos]
        if pares:
            uv3 = self._with_virtual_right(
                np.float64([kps[kp_idx].pt for kp_idx, _ in pares]))
            for n, (kp_idx, pid) in enumerate(pares):
                if kp_idx in pid_por_kp or pid in pids_vistos:
                    continue
                self.mapa.add_obs(kf_id, pid, uv3[n])
                pid_por_kp[kp_idx] = pid
                pids_vistos.add(pid)

        # 2) PUNTOS NUEVOS por retro-proyeccion: keypoints sin punto y con z.
        if self._depth is not None:
            prof = self._depth
            h, w = prof.shape
            cand = []
            for i, kp in enumerate(kps):
                if i in pid_por_kp:
                    continue
                u, v = int(round(kp.pt[0])), int(round(kp.pt[1]))
                if 0 <= u < w and 0 <= v < h:
                    z = float(prof[v, u])
                    if self.DEPTH_MIN < z < self.DEPTH_MAX:
                        cand.append((i, kp.pt[0], kp.pt[1], z))
            if len(cand) > self.DEPTH_MAX_NEW_POINTS:
                paso = int(np.ceil(len(cand) / self.DEPTH_MAX_NEW_POINTS))
                cand = cand[::paso]      # submuestreo uniforme (tope de mapa)
            if cand:
                fx, fy = self.K[0, 0], self.K[1, 1]
                cx, cy = self.K[0, 2], self.K[1, 2]
                px = np.float64([(c[1], c[2]) for c in cand])
                z = np.float64([c[3] for c in cand])
                pts_c = np.stack([(px[:, 0] - cx) * z / fx,
                                  (px[:, 1] - cy) * z / fy, z], axis=1)
                pts_w = (self.T_w_c[:3, :3] @ pts_c.T).T + self.T_w_c[:3, 3]
                uv3 = self._with_virtual_right(px)
                for n, c in enumerate(cand):
                    pid = self.mapa.add_punto(pts_w[n], desc[c[0]])
                    self.mapa.add_obs(kf_id, pid, uv3[n])
                    pid_por_kp[c[0]] = pid

        self._kf_db.append({"id": kf_id, "kps": kps, "desc": desc,
                            "desc_pr": self._desc_fuertes(kps, desc),
                            "pts": pid_por_kp})
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
        # Con el residuo de profundidad la ESCALA ya es observable (el gauge
        # baja de 7 a 6 gdl) — pero anclar 2 keyframes sigue siendo correcto
        # y mantiene el codigo identico al nivel 11/14 (y al padre).
        fijas = set(ventana[:2])

        poses_opt, pts_opt = bundle_adjustment(
            self.K, poses, pts, obs, fixed_kfs=fijas,
            iterations=self.BA_ITERS, huber_px=2.5, stereo_bf=self.residuo_bf)

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
        poses_opt, pts_opt = bundle_adjustment(
            self.K, poses, pts, self.mapa.obs, fixed_kfs=set(ids[:2]),
            iterations=iterations or self.GBA_ITERS, huber_px=2.5,
            historial=historial, stereo_bf=self.residuo_bf)
        for k, T in poses_opt.items():
            self.kf_poses[k] = T
        for p, X in pts_opt.items():
            self.mapa.puntos[p] = X

    # ── CIERRE DE BUCLE (nivel 13) + observaciones puente ────────────────────

    def _buscar_bucle(self, kf_id, kps, desc, info) -> None:
        if kf_id - self._ultimo_bucle < self.LOOP_COOLDOWN:
            return

        # FILTRO 1 — temporal: solo keyframes LEJANOS (en KEYFRAMES).
        candidatos = [kf for kf in self._kf_db
                      if kf_id - kf["id"] >= self.LOOP_TEMPORAL_GAP]
        if not candidatos:
            return

        # FILTRO 2 — reconocimiento de lugar con los descriptores fuertes.
        desc_pr = self._desc_fuertes(kps, desc)
        mejor, mejor_n = None, 0
        for kf in candidatos:
            n = len(self._match(desc_pr, kf["desc_pr"]))
            if n > mejor_n:
                mejor, mejor_n = kf, n
        if mejor is None or mejor_n < self.LOOP_MIN_MATCHES:
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
        """Grafo de poses en SE(3) — RÍGIDO, porque el mapa es MÉTRICO.

        ─── La lección 35 del padre (la moraleja más bonita del curso) ───
        En el nivel 12/13 este grafo era Sim(3): la deriva monocular incluye
        ESCALA, y el bucle debe medirla y redistribuirla (Strasdat). Aquí NO:
        la escala es una MEDICIÓN del sensor y no se negocia. Un bucle Sim(3)
        re-escalaría el mapa viejo mientras los puntos nuevos siguen naciendo
        en metros → el siguiente bucle "corrige" la discrepancia que el
        anterior CREÓ: composición de error (el padre lo midió: s_rel de los
        bucles degenerando 1.0 -> 0.03, ATE 22.1 cm, escala 2.09; en SE(3):
        4.7 cm, escala 1.036). El grupo del bucle depende de QUIÉN fija la
        escala: convención -> Sim(3); sensor -> SE(3).
        """
        ids = sorted(self.kf_poses)
        g = GrafoDePoses("se3")
        for k in ids:
            g.add_pose(k, self.kf_poses[k], fixed=(k <= id_viejo))

        for a, b in zip(ids[:-1], ids[1:]):
            T_rel = se3_inv(self.kf_poses[a]) @ self.kf_poses[b]
            g.add_odometry(a, b, T_rel, np.eye(6) * 1e2)

        T_rel_loop = se3_inv(self.kf_poses[id_viejo]) @ T_pnp
        g.add_loop(id_viejo, id_nuevo, T_rel_loop, np.eye(6) * 1e4)

        res = g.optimize(iterations=15)

        # Re-anclar el mapa con la correccion RIGIDA del ultimo keyframe.
        T_ultimo = res[ids[-1]] @ se3_inv(self.kf_poses[ids[-1]])
        for k in ids:
            self.kf_poses[k] = res[k]
        self.mapa.aplicar_similitud(T_ultimo)     # rigida: |det R| = 1
        self.T_w_c = self.kf_poses[ids[-1]].copy()

    # ── bucle principal ──────────────────────────────────────────────────────

    def procesar(self, gray: np.ndarray,
                 prof: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """Un frame RGB-D: imagen gris + mapa de profundidad en METROS
        (None si este frame no tiene pareja de profundidad asociada)."""
        self._frame += 1
        self._depth = prof
        if self._img_hw is None:
            self._img_hw = gray.shape[:2]
        info = {"estado": self.estado, "n_matches": 0, "n_guiado": 0,
                "n_inliers": 0, "kf": False, "loop": None,
                "n_mapa": len(self.mapa)}
        kps, desc = self.orb.detectAndCompute(gray, None)
        if desc is None or len(kps) < 10:
            if self.estado != "INIT":
                self._coast(info)
            return self.T_w_c, info

        if self.estado == "INIT":
            # SOLO con profundidad: inicializar sin ella crearia un mapa a
            # escala gauge que luego se mezclaria con puntos en metros — el
            # bug del mapa MIXTO que el padre cazo en fr1_desk (leccion 36:
            # escala 1.008 de pura casualidad y ni bucle SE(3) ni residuo
            # activos). El driver ademas espera al primer frame con depth.
            if prof is not None:
                self._init_rgbd(kps, desc, info)
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


