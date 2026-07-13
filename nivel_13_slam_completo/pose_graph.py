"""Grafo de poses: Gauss-Newton / Levenberg-Marquardt sobre la variedad.

El bundle adjustment (nivel 11) optimiza poses Y puntos. El grafo de poses
optimiza SOLO poses, usando como medidas las transformaciones RELATIVAS entre
ellas (que el tracking ya estimó). Es mucho más barato — y es lo que se corre
cuando se cierra un bucle, para repartir la deriva por toda la trayectoria.

─── La matemática: el problema ───────────────────────────────────────────────
    argmin_{T_i}  Σ_(i,j)  ‖ Log( T̂_ij⁻¹ · T_i⁻¹ · T_j ) ‖²_Λ

Cada arista (i,j) dice: "medí que del nodo i al j hay T̂_ij". El error es la
diferencia entre lo MEDIDO y lo que las poses actuales IMPLICAN, expresada en
el espacio tangente (Log): si coinciden, T̂_ij⁻¹·T_i⁻¹·T_j = I y Log(I) = 0.

─── La matemática: blanqueo (whitening) ──────────────────────────────────────
El costo Σ eᵀ·Λ·e se convierte en mínimos cuadrados ORDINARIOS factorizando
la información Λ = Lᵀ·L (Cholesky):

    ‖e‖²_Λ = eᵀLᵀL·e = ‖L·e‖²      ->  residuo blanqueado r = L·e

Así el solver no necesita saber de covarianzas: sólo minimiza ½‖r‖².

─── La matemática: jacobiano NUMÉRICO (decisión didáctica) ───────────────────
Cada pose se perturba POR LA DERECHA, T_i <- T_i·Exp(δ_i), y J = ∂r/∂δ se
evalúa por diferencias finitas. El jacobiano analítico de Log(T̂⁻¹T_i⁻¹T_j)
exige los jacobianos adjuntos de SE(3) y esconde el bosque; el numérico cuesta
12 evaluaciones por arista (gratis a esta escala) y es imposible equivocarse
de convención. En el BA (nivel 11) sí compensaba el analítico: allí hay miles
de observaciones. Aquí, decenas de aristas.

─── La matemática: el gauge, otra vez ────────────────────────────────────────
Sin anclar nada, mover TODO el grafo junto no cambia ningún residuo relativo:
H es singular. Se fija declarando una pose `fixed=True`. (En SE(3) basta UNA:
las medidas relativas ya fijan la escala. En Sim(3)... también basta una, pero
la escala GLOBAL queda libre, que es justo lo que queremos: el bucle
redistribuye la escala RELATIVA entre nodos.)

─── La matemática: Huber en los bucles ───────────────────────────────────────
Los factores de bucle llevan kernel robusto: un reconocimiento de lugar
EQUIVOCADO (dos pasillos idénticos) no debe poder doblar el grafo entero. Si
el residuo blanqueado excede δ, su peso decae w = δ/‖r‖: empuja linealmente,
no cuadráticamente.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from lie import se3_exp, se3_inv, se3_log, sim3_exp, sim3_inv, sim3_log


class _SE3Ops:
    """Operaciones de grupo para poses RÍGIDAS (6 gdl)."""
    DIM = 6
    exp = staticmethod(se3_exp)
    log = staticmethod(se3_log)
    inv = staticmethod(se3_inv)


class _Sim3Ops:
    """Operaciones de grupo para SIMILITUDES (7 gdl: + escala).

    El grupo correcto para bucles MONOCULARES: la deriva de escala es un grado
    de libertad MÁS que redistribuir. Ver el experimento de Strasdat.
    """
    DIM = 7
    exp = staticmethod(sim3_exp)
    log = staticmethod(sim3_log)
    inv = staticmethod(sim3_inv)


_GRUPOS = {"se3": _SE3Ops, "sim3": _Sim3Ops}


class GrafoDePoses:
    """Optimizador de grafo de poses, genérico en el GRUPO.

    `grupo="se3"` (rígido) o `grupo="sim3"` (similitudes). Todo el mecanismo
    (blanqueo, LM, Huber, gauge) es IDÉNTICO: lo único que cambia es en qué
    variedad viven los nodos. Esa es la gracia de escribir el optimizador
    sobre Exp/Log en vez de sobre matrices concretas.
    """

    HUBER_DELTA = 1.0     # umbral (residuo blanqueado) del kernel robusto
    JACOBIAN_EPS = 1e-6   # paso de las diferencias finitas

    def __init__(self, grupo: str = "se3") -> None:
        try:
            self._ops = _GRUPOS[grupo]
        except KeyError:
            raise ValueError(f"Grupo desconocido: {grupo!r}. "
                             f"Disponibles: {', '.join(_GRUPOS)}") from None
        self.grupo = grupo
        self._poses: Dict[int, np.ndarray] = {}
        self._fijos: set = set()
        self._aristas: List[dict] = []

    # ── construcción del grafo ───────────────────────────────────────────────

    def add_pose(self, node_id: int, T, fixed: bool = False) -> None:
        self._poses[node_id] = np.asarray(T, dtype=float).copy()
        if fixed:
            self._fijos.add(node_id)

    def add_odometry(self, i: int, j: int, T_rel, information) -> None:
        """Arista de ODOMETRÍA: lo que el tracking midió entre i y j."""
        self._add_edge(i, j, T_rel, information, robusto=False)

    def add_loop(self, i: int, j: int, T_rel, information) -> None:
        """Arista de BUCLE: "el nodo j es el mismo sitio que el i".

        Lleva kernel robusto (ver el docstring del módulo): un falso positivo
        de reconocimiento de lugar no debe poder doblar el grafo entero.
        """
        self._add_edge(i, j, T_rel, information, robusto=True)

    def _add_edge(self, i, j, T_rel, information, robusto) -> None:
        info = np.asarray(information, dtype=float)
        D = self._ops.DIM
        if info.shape != (D, D):
            raise ValueError(f"La informacion debe ser {D}x{D} para el grupo "
                             f"{self.grupo!r}; llego {info.shape}")
        L = np.linalg.cholesky(info).T                       # Λ = LᵀL
        self._aristas.append({
            "i": i, "j": j,
            "T_meas_inv": self._ops.inv(np.asarray(T_rel, dtype=float)),
            "sqrt_info": L,
            "robusto": robusto,
        })

    # ── optimización ─────────────────────────────────────────────────────────

    def _residual(self, arista, poses) -> np.ndarray:
        """r = L · Log( T̂_ij⁻¹ · T_i⁻¹ · T_j )   (blanqueado)."""
        e = self._ops.log(arista["T_meas_inv"]
                          @ self._ops.inv(poses[arista["i"]])
                          @ poses[arista["j"]])
        return arista["sqrt_info"] @ e

    def _costo(self, poses) -> float:
        """½ Σ w_k·‖r_k‖² (con el mismo peso robusto que usa el paso)."""
        total = 0.0
        for a in self._aristas:
            r = self._residual(a, poses)
            n = float(np.linalg.norm(r))
            w = 1.0
            if a["robusto"] and n > self.HUBER_DELTA:
                w = self.HUBER_DELTA / n
            total += 0.5 * w * n ** 2
        return total

    def optimize(self, iterations: int = 20,
                 historial: list | None = None) -> Dict[int, np.ndarray]:
        D = self._ops.DIM
        poses = {k: v.copy() for k, v in self._poses.items()}
        if not self._fijos and poses:                  # gauge: anclar la primera
            self._fijos.add(min(poses))
        libres = sorted(k for k in poses if k not in self._fijos)
        idx = {k: D * n for n, k in enumerate(libres)}
        n_vars = D * len(libres)
        if n_vars == 0 or not self._aristas:
            return poses

        lam = 1e-6
        costo = self._costo(poses)
        if historial is not None:
            historial.append(costo)

        for _ in range(iterations):
            J = np.zeros((D * len(self._aristas), n_vars))
            r = np.zeros(D * len(self._aristas))

            for k, a in enumerate(self._aristas):
                rk = self._residual(a, poses)
                # IRLS-Huber: reescalar residuo y jacobiano por sqrt(w).
                w = 1.0
                if a["robusto"]:
                    n = float(np.linalg.norm(rk))
                    w = 1.0 if n <= self.HUBER_DELTA else self.HUBER_DELTA / n
                sw = np.sqrt(w)
                r[D * k: D * k + D] = sw * rk

                # Jacobiano NUMERICO: perturbar cada pose implicada por la
                # derecha, en cada una de sus D direcciones tangentes.
                for nodo in (a["i"], a["j"]):
                    if nodo not in idx:
                        continue
                    col = idx[nodo]
                    T_orig = poses[nodo]
                    for d in range(D):
                        delta = np.zeros(D)
                        delta[d] = self.JACOBIAN_EPS
                        poses[nodo] = T_orig @ self._ops.exp(delta)
                        rk_p = self._residual(a, poses)
                        J[D * k: D * k + D, col + d] = sw * (rk_p - rk) / self.JACOBIAN_EPS
                    poses[nodo] = T_orig

            # Paso LM:  (H + lam·diag(H))·delta = −Jᵀ·r
            H = J.T @ J
            g = -J.T @ r
            try:
                delta = np.linalg.solve(
                    H + lam * np.diag(np.diag(H)) + 1e-12 * np.eye(n_vars), g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue

            prueba = {k: v.copy() for k, v in poses.items()}
            for nodo, col in idx.items():
                prueba[nodo] = prueba[nodo] @ self._ops.exp(delta[col: col + D])
            costo_prueba = self._costo(prueba)

            if costo_prueba < costo:      # mejora: aceptar y confiar mas
                poses, costo = prueba, costo_prueba
                lam = max(lam / 3.0, 1e-9)
            else:                         # empeora: paso mas conservador
                lam *= 5.0
            if historial is not None:
                historial.append(costo)
            if np.linalg.norm(delta) < 1e-10:
                break

        return poses
