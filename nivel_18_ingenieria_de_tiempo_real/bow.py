"""Reconocimiento de lugar por BOLSA DE PALABRAS VISUALES (BoW).

El cierre de bucle necesita responder "¿a qué keyframe antiguo se parece este
frame?". La fuerza bruta (matchear contra CADA keyframe) es O(KFs) con
constante alta — en el nivel 14 la acotamos con los 300 descriptores más
fuertes, pero sigue siendo lineal y el perfil la señala en cuanto la base
crece. BoW la vuelve sub-lineal: cada imagen se resume en un histograma
disperso de "palabras visuales" y un índice invertido devuelve candidatos en
milisegundos. Solo el mejor candidato pasa a la verificación geométrica.

─── La matemática ─────────────────────────────────────────────────────────────
1. VOCABULARIO (k-medias en el espacio del descriptor). Para descriptores
   BINARIOS (ORB) la media aritmética no existe: el centroide que minimiza la
   suma de distancias de Hamming es la MEDIANA por bit — el VOTO DE MAYORÍA
   (bit j del centroide = 1 si más de la mitad de sus miembros lo tienen).
   La asignación descriptor→palabra es un vecino más cercano, delegado en
   cv2.BFMatcher (C++).

2. TF-IDF (Sivic & Zisserman, "Video Google", 2003). El histograma crudo
   sobre-pondera palabras que salen en todas partes (textura genérica):

       tf(w, imagen) = n_w / n_total       (frecuencia en la imagen)
       idf(w)        = log(N / df_w)       (rareza en el corpus)

   La similitud entre dos imágenes es el COSENO de sus vectores tf·idf.

3. ÍNDICE INVERTIDO: palabra → keyframes que la contienen. La consulta solo
   visita keyframes que comparten ALGUNA palabra con el query.
──────────────────────────────────────────────────────────────────────────────

Nota de alcance: DBoW2/3 (lo de ORB-SLAM) añade un árbol jerárquico de
vocabulario y vocabularios pre-entrenados en millones de imágenes. Aquí el
vocabulario se entrena EN SESIÓN con los primeros keyframes — suficiente para
bases de decenas o cientos de KFs, y autocontenido.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class BolsaDePalabras:
    """Vocabulario + índice invertido + consulta tf·idf (dtype-agnóstico)."""

    def __init__(self, n_palabras: int = 512, kmeans_iters: int = 6,
                 semilla: int = 0) -> None:
        self.n_palabras = n_palabras
        self.kmeans_iters = kmeans_iters
        self._rng = np.random.default_rng(semilla)
        self._vocab: Optional[np.ndarray] = None
        self._matcher: Optional[cv2.BFMatcher] = None
        self._tf: Dict[int, Dict[int, float]] = {}    # kf_id -> {palabra: tf}
        self._df: Dict[int, int] = {}                 # palabra -> nº de KFs
        self._invertido: Dict[int, List[int]] = {}    # palabra -> [kf_id]

    @property
    def entrenado(self) -> bool:
        return self._vocab is not None

    def entrenar(self, descriptores: np.ndarray) -> None:
        """K-medias sobre una muestra (los primeros keyframes de la sesión)."""
        desc = np.asarray(descriptores)
        n = len(desc)
        k = min(self.n_palabras, max(2, n // 4))
        centroides = desc[self._rng.choice(n, size=k, replace=False)].copy()
        norm = cv2.NORM_HAMMING if desc.dtype == np.uint8 else cv2.NORM_L2
        bf = cv2.BFMatcher(norm)
        for _ in range(self.kmeans_iters):
            asigna = np.array([m.trainIdx for m in bf.match(desc, centroides)])
            for j in range(k):
                miembros = desc[asigna == j]
                if not len(miembros):
                    # cluster vacio: re-sembrar con un descriptor al azar
                    centroides[j] = desc[self._rng.integers(0, n)]
                elif desc.dtype == np.uint8:
                    # ─── centroide de Hamming: VOTO DE MAYORIA por bit ───
                    bits = np.unpackbits(miembros, axis=1)
                    mayoria = (bits.mean(axis=0) >= 0.5).astype(np.uint8)
                    centroides[j] = np.packbits(mayoria)
                else:
                    centroides[j] = miembros.mean(axis=0)
        self._vocab = centroides
        self._matcher = bf

    def _cuantizar(self, desc: np.ndarray) -> Dict[int, float]:
        """Descriptores → histograma tf disperso {palabra: frecuencia}."""
        palabras = [m.trainIdx for m in self._matcher.match(desc, self._vocab)]
        tf: Dict[int, float] = {}
        for w in palabras:
            tf[w] = tf.get(w, 0.0) + 1.0
        inv = 1.0 / max(len(palabras), 1)
        return {w: c * inv for w, c in tf.items()}

    def indexar(self, kf_id: int, desc: np.ndarray) -> None:
        """Añade un keyframe al índice (tras entrenar el vocabulario)."""
        tf = self._cuantizar(desc)
        self._tf[kf_id] = tf
        for w in tf:
            self._df[w] = self._df.get(w, 0) + 1
            self._invertido.setdefault(w, []).append(kf_id)

    def consultar(self, desc: np.ndarray, top_k: int = 5
                  ) -> List[Tuple[int, float]]:
        """Los top_k keyframes más parecidos por coseno tf·idf. Solo visita
        los que comparten alguna palabra con el query (índice invertido)."""
        if not self.entrenado or not self._tf:
            return []
        q_tf = self._cuantizar(desc)
        n_kf = len(self._tf)
        logs: Dict[int, float] = {}      # memo: los df se repiten mucho

        def idf(w: int) -> float:
            df = self._df.get(w, 0)
            if df <= 0:
                return 0.0
            if df not in logs:
                logs[df] = math.log(df)
            return math.log(n_kf) - logs[df]

        q = {w: tf * idf(w) for w, tf in q_tf.items()}
        q_norm = math.sqrt(sum(v * v for v in q.values())) or 1.0

        puntos: Dict[int, float] = {}
        for w, qv in q.items():
            if qv == 0.0:
                continue
            iw = idf(w)
            for kf in self._invertido.get(w, ()):
                puntos[kf] = puntos.get(kf, 0.0) + qv * self._tf[kf].get(w, 0.0) * iw
        scores = []
        for kf, dot in puntos.items():
            # La norma del documento usa el idf COMPLETO de sus palabras —
            # si no, el coseno queda mal normalizado.
            d_norm = math.sqrt(sum((tf * idf(w)) ** 2
                                   for w, tf in self._tf[kf].items())) or 1.0
            scores.append((kf, dot / (q_norm * d_norm)))
        scores.sort(key=lambda s: -s[1])
        return scores[:top_k]
