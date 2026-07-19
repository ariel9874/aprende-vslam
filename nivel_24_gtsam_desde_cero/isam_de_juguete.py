"""iSAM DE JUGUETE (acto 4): re-resolver SOLO lo que cambió.

El dilema que cierra la trilogía de estimación del curso:

  - los backends del nivel 21 re-resuelven TODO el grafo en cada paso
    (exacto, pero el costo crece con el viaje);
  - el filtro del 23 marginaliza el pasado al instante (barato, pero cada
    linealización queda sellada: no puede arrepentirse).

iSAM (Kaess et al.) es el camino de en medio: mantener el grafo COMPLETO,
pero al llegar una medida re-factorizar únicamente la parte del árbol de
Bayes que esa medida invalida (acto 3). Ni re-resolver todo, ni sellar nada.

─── Cómo funciona este juguete (y qué es de verdad de iSAM2) ─────────────────
Orden de eliminación TEMPORAL: con él, el camino al root del acto 3 es un
SUFIJO del orden — "actualizar los cliques afectados" = re-eliminar desde la
variable más vieja que el paso toca. Tres mecanismos, los tres con su
contraparte real:

1. FACTORES CACHEADOS: al eliminar v se guarda el factor que emitió sobre su
   separador (`cache`). Para re-eliminar un sufijo no hace falta el prefijo:
   basta sumar los caches que el prefijo dejó caer sobre el sufijo. (En
   iSAM2, el factor cacheado que cada clique guarda hacia su padre.)

2. RE-LINEALIZACIÓN POR UMBRAL: una variable se re-linealiza solo si su
   corrección δ superó un umbral; entonces TODOS sus factores se recalculan
   y el sufijo se extiende hasta cubrirlos. (El relinearizeThreshold y el
   "wildfire" de iSAM2, en versión honesta de juguete.)

3. REORDEN AL VUELO: un landmark viejo re-observado se mueve al final del
   orden durante la re-eliminación que él mismo provocó — pagas su camino
   UNA vez, no en cada paso siguiente. (iSAM1 re-ordenaba todo cada 100
   pasos; iSAM2 re-ordena el sub-árbol afectado en cada update.)

El CIERRE DE BUCLE es el momento de la verdad: re-observar un landmark de la
primera vuelta dispara una re-eliminación casi total (el camino del acto 3,
ahora pagado de verdad) y las correcciones grandes disparan el wildfire. El
pico se MIDE — y después el costo vuelve a caer.

Lo que iSAM2 real hace más fino que este juguete: deltas parciales (aquí la
sustitución hacia atrás es completa, que es barata) y relinealización fluida
dentro del árbol (aquí el sufijo es la unidad). Kaess et al. (2012) es la
lectura del nivel.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List

import numpy as np

import eliminacion as el
from mundo import componer, envolver

UMBRAL_RELIN = 0.05     # la perilla de iSAM2 (ejercicio 1: barrerla)


class ISAMJuguete:
    """El estimador incremental. Estado: el orden de eliminación, el punto
    de linealización POR VARIABLE, los condicionales y los caches."""

    def __init__(self, umbral: float = UMBRAL_RELIN):
        self.umbral = umbral
        self.orden: List[tuple] = []
        self.pos: Dict[tuple, int] = {}
        self.lin: Dict[tuple, np.ndarray] = {}      # punto de linealizacion
        self.delta: Dict[tuple, np.ndarray] = {}    # la correccion resuelta
        self.fnl: List[dict] = []                   # factores no lineales
        self.alin: List[tuple] = []                 # y su forma blanqueada
        self.por_var: Dict[tuple, List[int]] = defaultdict(list)
        self.cond: Dict[tuple, dict] = {}
        self.cache: Dict[tuple, tuple] = {}
        self.relin_pendiente: set = set()
        self.stats: List[dict] = []

    def estim(self, v) -> np.ndarray:
        """La estimación actual: punto de linealización + corrección."""
        val = self.lin[v] + self.delta.get(v, 0.0)
        if v[0] == "x":
            val[2] = envolver(val[2])
        return val

    def paso(self, nuevos: List[dict], iniciales: Dict) -> None:
        """Un update de iSAM: factores nuevos + variables nuevas."""
        t0 = time.perf_counter()
        for v, val in iniciales.items():
            self.pos[v] = len(self.orden)
            self.orden.append(v)
            self.lin[v] = np.asarray(val, float).copy()
            self.delta[v] = np.zeros(el.dim(v))
        idxs_nuevos = []
        for f in nuevos:
            idx = len(self.fnl)
            self.fnl.append(f)
            self.alin.append(el.linealizar_factor(f, self.lin))
            for u in f["vars"]:
                self.por_var[u].append(idx)
            idxs_nuevos.append(idx)

        # desde dónde re-eliminar: la variable más vieja que algo toca
        k = min(min(self.pos[u] for u in self.fnl[i]["vars"])
                for i in idxs_nuevos)

        # el wildfire de juguete: re-linealizar lo que se movió > umbral
        n_relin = len(self.relin_pendiente)
        tocados = set()
        for v in self.relin_pendiente:
            self.lin[v] = self.estim(v)
            self.delta[v] = np.zeros(el.dim(v))
            tocados.update(self.por_var[v])
        for i in tocados:
            self.alin[i] = el.linealizar_factor(self.fnl[i], self.lin)
            k = min(k, min(self.pos[u] for u in self.fnl[i]["vars"]))
        self.relin_pendiente = set()

        # reorden al vuelo: landmarks viejos re-observados, al final
        movidos = sorted({u for i in idxs_nuevos for u in self.fnl[i]["vars"]
                          if u[0] == "l" and u not in iniciales},
                         key=lambda u: self.pos[u])
        if movidos:
            sufijo = [v for v in self.orden[k:] if v not in set(movidos)]
            self.orden[k:] = sufijo + movidos
            for p in range(k, len(self.orden)):
                self.pos[self.orden[p]] = p

        self._reeliminar(k)
        self._resolver()
        self.relin_pendiente = {
            v for v in self.orden
            if float(np.abs(self.delta[v]).max()) > self.umbral}
        self.stats.append({"reelim": len(self.orden) - k, "relin": n_relin,
                           "n": len(self.orden),
                           "t": time.perf_counter() - t0})

    def _reeliminar(self, k: int) -> None:
        """Re-factorizar el sufijo orden[k:]. Entrantes: los factores cuyas
        variables viven todas en el sufijo + los caches que el prefijo dejó
        caer sobre él. El prefijo NI SE TOCA (acto 3, hecho código)."""
        S = self.orden[k:]
        en_sufijo = set(S)
        entrantes, vistos = [], set()
        for v in S:
            for i in self.por_var[v]:
                if i not in vistos:
                    vistos.add(i)
                    if all(u in en_sufijo for u in self.fnl[i]["vars"]):
                        entrantes.append(self.alin[i])
        for p in self.orden[:k]:
            f = self.cache.get(p)
            if f is not None and all(u in en_sufijo for u in f[0]):
                entrantes.append(f)
        resultado, _ = el.eliminar(entrantes, S)
        for cond, nuevo in resultado:
            self.cond[cond["v"]] = cond
            self.cache[cond["v"]] = nuevo

    def _resolver(self) -> None:
        """Sustitución hacia atrás completa (lineal y barata; el wildfire
        también sobre los deltas es el refinamiento de iSAM2 real)."""
        for v in reversed(self.orden):
            cond = self.cond[v]
            rhs = cond["d"].copy()
            if cond["sep"]:
                rhs = rhs - cond["S"] @ np.concatenate(
                    [self.delta[u] for u in cond["sep"]])
            self.delta[v] = np.linalg.solve(cond["R"], rhs)


# ── el mundo, procesado EN LÍNEA ─────────────────────────────────────────────


def _agregar_obs(i, obs_por_pose, existentes, nuevos, iniciales, p):
    for j, z in obs_por_pose.get(i, []):
        v = ("l", j)
        nuevos.append({"tipo": "obs", "vars": (("x", i), v), "z": z})
        if v not in existentes and v not in iniciales:
            c, s = np.cos(p[2]), np.sin(p[2])
            iniciales[v] = np.array([p[0] + c * z[0] - s * z[1],
                                     p[1] + s * z[0] + c * z[1]])


def correr(mundo: dict, umbral: float = UMBRAL_RELIN) -> Dict:
    """El circuito pose a pose por el iSAM de juguete. Devuelve la
    trayectoria ONLINE (cada pose al emitirse) y el costo por paso."""
    obs_por_pose: Dict[int, list] = defaultdict(list)
    for i, j, z in mundo["obs"]:
        obs_por_pose[i].append((j, z))

    isam = ISAMJuguete(umbral)
    tray = []
    t0 = time.perf_counter()
    nuevos = [{"tipo": "prior", "vars": (("x", 0),), "z": np.zeros(3)}]
    iniciales = {("x", 0): np.zeros(3)}
    _agregar_obs(0, obs_por_pose, isam.lin, nuevos, iniciales, np.zeros(3))
    isam.paso(nuevos, iniciales)
    tray.append(isam.estim(("x", 0)))

    for i, _, z in mundo["odo"]:
        p = componer(isam.estim(("x", i)), z)
        nuevos = [{"tipo": "odo", "vars": (("x", i), ("x", i + 1)), "z": z}]
        iniciales = {("x", i + 1): p}
        _agregar_obs(i + 1, obs_por_pose, isam.lin, nuevos, iniciales, p)
        isam.paso(nuevos, iniciales)
        tray.append(isam.estim(("x", i + 1)))

    N = len(mundo["inicial"])
    return {"tray": np.array(tray),
            "poses": np.array([isam.estim(("x", i)) for i in range(N)]),
            "stats": isam.stats, "t": time.perf_counter() - t0}


def correr_batch(mundo: dict, max_iter: int = 3) -> Dict:
    """El estándar de oro (y el caro): TODO el grafo re-resuelto con
    Gauss-Newton caliente en cada paso — el nivel 21 puesto en línea."""
    obs_por_pose: Dict[int, list] = defaultdict(list)
    for i, j, z in mundo["obs"]:
        obs_por_pose[i].append((j, z))

    valores: Dict[tuple, np.ndarray] = {}
    claves: List[tuple] = []
    fnl: List[dict] = []
    tray, stats = [], []
    t0 = time.perf_counter()

    def gn():
        for _ in range(max_iter):
            lineales = [el.linealizar_factor(f, valores) for f in fnl]
            H, g = el.hessiana(lineales, claves)
            d = np.linalg.solve(H + 1e-9 * np.eye(len(H)), -g)
            off = 0
            delta = {}
            for u in claves:
                delta[u] = d[off:off + el.dim(u)]
                off += el.dim(u)
            if el._aplicar(valores, delta) < 1e-8:
                break

    def alta(v, val):
        valores[v] = np.asarray(val, float).copy()
        claves.append(v)

    alta(("x", 0), np.zeros(3))
    fnl.append({"tipo": "prior", "vars": (("x", 0),), "z": np.zeros(3)})
    nuevos, iniciales = [], {}
    _agregar_obs(0, obs_por_pose, valores, nuevos, iniciales, np.zeros(3))
    fnl += nuevos
    for v, val in iniciales.items():
        alta(v, val)
    t1 = time.perf_counter()
    gn()
    stats.append({"t": time.perf_counter() - t1})
    tray.append(valores[("x", 0)].copy())

    for i, _, z in mundo["odo"]:
        t1 = time.perf_counter()
        p = componer(valores[("x", i)], z)
        alta(("x", i + 1), p)
        fnl.append({"tipo": "odo", "vars": (("x", i), ("x", i + 1)), "z": z})
        nuevos, iniciales = [], {}
        _agregar_obs(i + 1, obs_por_pose, valores, nuevos, iniciales, p)
        fnl += nuevos
        for v, val in iniciales.items():
            alta(v, val)
        gn()
        stats.append({"t": time.perf_counter() - t1})
        tray.append(valores[("x", i + 1)].copy())

    N = len(mundo["inicial"])
    return {"tray": np.array(tray),
            "poses": np.array([valores[("x", i)] for i in range(N)]),
            "stats": stats, "t": time.perf_counter() - t0}
