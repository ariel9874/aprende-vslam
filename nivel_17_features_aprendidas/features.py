"""Frontends intercambiables: ORB (clásico) y SuperPoint + LightGlue (deep).

El material de este nivel. Todo lo demás (tracker, BA, bucle) es el sistema
del nivel 14 sin tocar: la pregunta del nivel es SOLO qué pasa cuando cambias
QUÉ mira el sistema y CÓMO empareja — y la respuesta se mide, no se cree.

─── La matemática: SuperPoint ────────────────────────────────────────────────
CNN con encoder compartido y dos cabezas: el DETECTOR clasifica cada celda
8×8 en 65 clases (64 posiciones + "sin punto") y el DESCRIPTOR produce 256
floats por punto. El entrenamiento es AUTO-supervisado: esquinas sintéticas
(MagicPoint) + Homographic Adaptation — agregar las detecciones bajo
homografías aleatorias de la MISMA imagen fabrica el ground truth de
"esquinidad" sin etiquetar nada a mano. Lo que compra frente a ORB: el
descriptor agrega contexto de un receptive field grande, así que sobrevive
al MOTION BLUR que borra los gradientes locales de los que ORB vive.

─── La matemática: LightGlue ─────────────────────────────────────────────────
Matcher de grafos con auto/cross-atención sobre (posición + descriptor) y
asignación aprendida. Adaptativo: capas con early-exit según confianza. La
consecuencia práctica que muerde en la integración: NECESITA los keypoints
de AMBOS lados y el tamaño de imagen (su atención es espacial). Por eso solo
sirve para pares de IMÁGENES (init, puntos nuevos, verificación de bucle) —
NO para el matching 3D→2D contra el mapa, donde un lado son puntos sin
keypoint. Esa asimetría la resuelve el matcher por descriptor de siempre
(la misma solución del repo padre con su `_desc_matcher`).
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

_INSTALL = ("El frontend aprendido requiere torch y lightglue:\n"
            "  pip install torch          (con CUDA si tienes GPU NVIDIA)\n"
            "  pip install git+https://github.com/cvg/LightGlue.git")


class ExtractorORB:
    """El frontend clásico de todo el curso (niveles 05-16)."""

    nombre = "orb"

    def __init__(self, n_features: int = 2000) -> None:
        self._orb = cv2.ORB_create(nfeatures=n_features)

    def detectar(self, gray: np.ndarray):
        return self._orb.detectAndCompute(gray, None)


class ExtractorSuperPoint:
    """SuperPoint vía el paquete `lightglue` (pesos: solo investigación).

    Devuelve la MISMA firma que ORB (kps de cv2, matriz de descriptores),
    pero los descriptores son float32 de 256 dims (distancia L2, no Hamming).
    El score del detector viaja en kp.response — igual que en ORB — para que
    el voto del reconocimiento de lugar (descriptores fuertes) siga valiendo.
    """

    nombre = "superpoint"

    def __init__(self, n_features: int = 2000,
                 device: Optional[str] = None) -> None:
        try:
            import torch
            import lightglue
        except ImportError as exc:
            raise ImportError(_INSTALL) from exc
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._modelo = (lightglue.SuperPoint(max_num_keypoints=n_features)
                        .eval().to(self.device))

    def detectar(self, gray: np.ndarray):
        t = (self._torch.from_numpy(gray).float().div(255.0)[None, None]
             .to(self.device))
        with self._torch.no_grad():
            feats = self._modelo.extract(t)
        pts = feats["keypoints"][0].cpu().numpy()
        scores = feats["keypoint_scores"][0].cpu().numpy()
        desc = feats["descriptors"][0].cpu().numpy().astype(np.float32)
        kps = [cv2.KeyPoint(float(x), float(y), 8.0, response=float(s))
               for (x, y), s in zip(pts, scores)]
        return kps, desc


class MatcherRatio:
    """Fuerza bruta + ratio test (nivel 06), consciente del descriptor.

    ─── El ratio de Lowe NO es una constante universal ───
    0.75 se calibró para descriptores tipo SIFT/ORB. En un espacio float de
    256 dims (SuperPoint) las distancias al 1er y 2do vecino se COMPRIMEN
    (concentración de la medida: en alta dimensión todo está "más o menos
    igual de lejos"), y con 0.75 casi ningún match sobrevive — lo medimos
    construyendo este nivel: el voto del reconocimiento de lugar daba máximo
    20 matches con el umbral en 40, y la relocalización no disparaba NUNCA.
    Para float se usa 0.90 (la práctica estándar de los pipelines con
    SuperPoint, p. ej. hloc). El umbral correcto depende de la GEOMETRÍA del
    espacio de descriptores, no del gusto.
    """

    nombre = "ratio"

    def __init__(self, ratio_binario: float = 0.75,
                 ratio_float: float = 0.90) -> None:
        self.ratio_binario = ratio_binario
        self.ratio_float = ratio_float
        self._bf_hamming = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._bf_l2 = cv2.BFMatcher(cv2.NORM_L2)

    def match(self, da, db, kps_a=None, kps_b=None,
              image_shape=None) -> List[cv2.DMatch]:
        if da is None or db is None or len(da) < 2 or len(db) < 2:
            return []
        binario = da.dtype == np.uint8
        bf = self._bf_hamming if binario else self._bf_l2
        ratio = self.ratio_binario if binario else self.ratio_float
        pares = bf.knnMatch(da, db, k=2)
        return [m for m, n in pares if m.distance < ratio * n.distance]


class MatcherLightGlue:
    """LightGlue para pares de IMÁGENES (init, puntos nuevos, bucle, reloc).

    Exige kps de ambos lados e image_shape: razona sobre POSICIONES, no solo
    descriptores (teoría en la cabecera). Para el matching 3D->2D contra el
    mapa no sirve — ahí sigue mandando MatcherRatio.
    """

    nombre = "lightglue"

    def __init__(self, features: str = "superpoint",
                 device: Optional[str] = None) -> None:
        try:
            import torch
            import lightglue
        except ImportError as exc:
            raise ImportError(_INSTALL) from exc
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._modelo = (lightglue.LightGlue(features=features)
                        .eval().to(self.device))

    def match(self, da, db, kps_a=None, kps_b=None,
              image_shape=None) -> List[cv2.DMatch]:
        if kps_a is None or kps_b is None or image_shape is None:
            raise ValueError("LightGlue necesita kps de ambos lados e "
                             "image_shape (atencion espacial)")
        if da is None or db is None or len(da) < 2 or len(db) < 2:
            return []
        torch = self._torch
        h, w = image_shape

        def empaquetar(kps, desc):
            pts = torch.tensor([[kp.pt[0], kp.pt[1]] for kp in kps],
                               dtype=torch.float32, device=self.device)[None]
            return {"keypoints": pts,
                    "descriptors": torch.from_numpy(
                        np.ascontiguousarray(desc)).float()[None].to(self.device),
                    "image_size": torch.tensor([[w, h]], dtype=torch.float32,
                                               device=self.device)}

        with torch.no_grad():
            out = self._modelo({"image0": empaquetar(kps_a, da),
                                "image1": empaquetar(kps_b, db)})
        pares = out["matches"][0].cpu().numpy()
        scores = (out["scores"][0].cpu().numpy() if "scores" in out
                  else np.ones(len(pares)))
        return [cv2.DMatch(int(i), int(j), float(1.0 - s))
                for (i, j), s in zip(pares, scores)]
