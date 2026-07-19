"""ELIMINACIÓN DE VARIABLES: lo que GTSAM hace de verdad (actos 1 y 2).

El nivel 21 dejó la matemática lista: MAP = mínimos cuadrados, Gauss-Newton
sobre H·δ = −g con H = ΣJᵀΛJ dispersa. Este archivo enseña CÓMO se resuelve
ese sistema explotando el grafo — que es la aportación real de GTSAM, no la
matemática (esa es de Gauss).

─── La matemática: eliminar UNA variable ─────────────────────────────────────
Cada factor gaussiano, BLANQUEADO (A = √Λ·J, b = −√Λ·e), es un bloque de
filas del sistema ‖A·δ − b‖². Para eliminar la variable v:

  1. JUNTA los factores que tocan v. Las demás variables que aparecen en
     ellos son el SEPARADOR S de v.
  2. FACTORIZA (QR) el sistema apilado [A_v | A_S | b]:

         [R  T | d1]     R triangular (dim v)
         [0  E | d2]

     La fila de arriba es el CONDICIONAL  p(v | S):  δ_v = R⁻¹(d1 − T·δ_S).
     La de abajo es un FACTOR NUEVO sobre S: toda la evidencia que los
     factores de v aportan sobre el resto del grafo, comprimida.
  3. Repite con la siguiente variable hasta vaciar el grafo.

Eso ES resolver: el grafo de factores se convierte en una RED DE BAYES
(una cadena de condicionales), y sustituir hacia atrás da δ. Y la conexión
con el nivel 21: el factor nuevo sobre S es EXACTAMENTE el complemento de
Schur de v — la marginalización del 21 era eliminación con otro nombre.
(El examen verifica ambas identidades numéricamente.)

─── El orden importa: el fill-in (acto 2) ────────────────────────────────────
El factor nuevo ACOPLA todo el separador entre sí. Si eliminas una variable
muy conectada al principio, su separador es enorme y el factor nuevo llena
la matriz: eso es el FILL-IN que el nivel 21 midió (1.33×) sin explicarlo.
El orden no cambia la SOLUCIÓN (el examen lo verifica) — cambia el COSTO.

Encontrar el orden óptimo es NP-duro; la heurística estándar es glotona:
eliminar siempre la variable de MENOR GRADO (min-degree — el pariente
didáctico del COLAMD que usa GTSAM). Aquí se implementa desde cero y se
compara contra el orden temporal, el orden "landmarks primero" (¡el de BA
del nivel 11 — óptimo allí, malo aquí!) y el peor a propósito (max-degree).
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from mundo import (SIGMA_OBS, SIGMA_ODO_TH, SIGMA_ODO_XY, between, envolver,
                   observar)

# Información de cada tipo de factor (las del nivel 21) y sus raíces
# cuadradas: blanquear con W = √Λ deja todos los residuos en unidades de σ.
INFO_ODO = np.diag([1 / SIGMA_ODO_XY ** 2, 1 / SIGMA_ODO_XY ** 2,
                    1 / SIGMA_ODO_TH ** 2])
INFO_OBS = np.eye(2) / SIGMA_OBS ** 2
INFO_PRIOR = np.eye(3) * 1e8

W_ODO = np.sqrt(INFO_ODO)
W_OBS = np.sqrt(INFO_OBS)
W_PRIOR = np.sqrt(INFO_PRIOR)

# Una clave de variable es ('x', i) para poses (dim 3) o ('l', j) para
# landmarks (dim 2) — el Symbol('x', i) de GTSAM, en versión tupla.


def dim(clave) -> int:
    return 3 if clave[0] == "x" else 2


# ── los factores del mundo (los mismos del nivel 21) ─────────────────────────


def factores_del_mundo(mundo: dict) -> List[dict]:
    """La lista de factores NO lineales: prior + odometría + observaciones."""
    fs = [{"tipo": "prior", "vars": (("x", 0),), "z": np.zeros(3)}]
    for i, j, z in mundo["odo"]:
        fs.append({"tipo": "odo", "vars": (("x", i), ("x", j)), "z": z})
    for i, j, z in mundo["obs"]:
        fs.append({"tipo": "obs", "vars": (("x", i), ("l", j)), "z": z})
    return fs


def valores_iniciales(mundo: dict) -> Dict[tuple, np.ndarray]:
    """Poses desde la odometría; landmarks desde su primera observación
    (exactamente la inicialización del nivel 21)."""
    valores = {("x", i): p.copy() for i, p in enumerate(mundo["inicial"])}
    for i, j, z in mundo["obs"]:
        if ("l", j) not in valores:
            p = valores[("x", i)]
            c, s = np.cos(p[2]), np.sin(p[2])
            valores[("l", j)] = np.array([p[0] + c * z[0] - s * z[1],
                                          p[1] + s * z[0] + c * z[1]])
    return valores


def _J_between(p_i, p_j):
    """Jacobianos analíticos del factor de odometría (nivel 21)."""
    c, s = np.cos(p_i[2]), np.sin(p_i[2])
    m = between(p_i, p_j)
    J_i = np.array([[-c, -s, m[1]], [s, -c, -m[0]], [0, 0, -1.0]])
    J_j = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])
    return J_i, J_j


def _J_obs(p, l):
    """Jacobianos analíticos del factor de observación (nivel 21)."""
    c, s = np.cos(p[2]), np.sin(p[2])
    o = observar(p, l)
    J_p = np.array([[-c, -s, o[1]], [s, -c, -o[0]]])
    J_l = np.array([[c, s], [-s, c]])
    return J_p, J_l


def linealizar_factor(f: dict, valores: Dict) -> Tuple[tuple, dict, np.ndarray]:
    """El factor BLANQUEADO en el punto `valores`: (claves, {clave: A}, b)
    con el residuo ‖Σ A_v·δ_v − b‖² en unidades de σ."""
    if f["tipo"] == "prior":
        (v,) = f["vars"]
        p = valores[v]
        e = np.array([p[0], p[1], envolver(p[2])])
        return f["vars"], {v: W_PRIOR.copy()}, -W_PRIOR @ e
    if f["tipo"] == "odo":
        vi, vj = f["vars"]
        e = between(valores[vi], valores[vj]) - f["z"]
        e[2] = envolver(e[2])
        J_i, J_j = _J_between(valores[vi], valores[vj])
        return f["vars"], {vi: W_ODO @ J_i, vj: W_ODO @ J_j}, -W_ODO @ e
    vi, vl = f["vars"]
    e = observar(valores[vi], valores[vl]) - f["z"]
    J_p, J_l = _J_obs(valores[vi], valores[vl])
    return f["vars"], {vi: W_OBS @ J_p, vl: W_OBS @ J_l}, -W_OBS @ e


# ── el corazón: eliminar ─────────────────────────────────────────────────────


def factorizar(factores_v: List, v, pos: Dict) -> Tuple[dict, Optional[tuple]]:
    """El paso 2 de la cabecera: QR sobre los factores que tocan v.
    Devuelve (condicional p(v|S), factor nuevo sobre S o None)."""
    sep = sorted({u for fvars, _, _ in factores_v for u in fvars if u != v},
                 key=lambda u: (pos[u], u))
    dv = dim(v)
    cols = [v] + sep
    off, n = {}, 0
    for u in cols:
        off[u] = n
        n += dim(u)
    filas = sum(len(b) for _, _, b in factores_v)
    A = np.zeros((filas, n))
    b = np.zeros(filas)
    r = 0
    for fvars, bloques, bf in factores_v:
        for u in fvars:
            A[r:r + len(bf), off[u]:off[u] + dim(u)] = bloques[u]
        b[r:r + len(bf)] = bf
        r += len(bf)

    Q, R = np.linalg.qr(A, mode="reduced")
    d = Q.T @ b
    cond = {"v": v, "R": R[:dv, :dv], "sep": sep, "S": R[:dv, dv:],
            "d": d[:dv]}
    if sep and R.shape[0] > dv:
        resto = R[dv:, dv:]
        nuevo = (tuple(sep),
                 {u: resto[:, off[u] - dv:off[u] - dv + dim(u)] for u in sep},
                 d[dv:])
        return cond, nuevo
    return cond, None


def eliminar(lineales: List, orden: List) -> Tuple[List, List]:
    """Elimina las variables de `orden`, una a una. Devuelve
    (resultado, sobrantes): un (condicional, factor_producido) por variable,
    y los factores que quedaron vivos (vacío si el orden cubre todo — si no,
    son EL GRAFO REDUCIDO: la marginalización, ver demo_schur)."""
    pos = defaultdict(lambda: 10 ** 9)
    pos.update({v: k for k, v in enumerate(orden)})
    activos = list(lineales)
    vivo = [True] * len(activos)
    por_var: Dict[tuple, List[int]] = defaultdict(list)
    for idx, (fvars, _, _) in enumerate(activos):
        for u in fvars:
            por_var[u].append(idx)

    resultado = []
    for v in orden:
        idxs = [i for i in por_var[v] if vivo[i]]
        for i in idxs:
            vivo[i] = False
        cond, nuevo = factorizar([activos[i] for i in idxs], v, pos)
        if nuevo is not None:
            activos.append(nuevo)
            vivo.append(True)
            for u in nuevo[0]:
                por_var[u].append(len(activos) - 1)
        resultado.append((cond, nuevo))
    sobrantes = [activos[i] for i in range(len(activos)) if vivo[i]]
    return resultado, sobrantes


def resolver(resultado: List) -> Dict[tuple, np.ndarray]:
    """Sustitución hacia atrás por la red de Bayes: del último condicional
    (sin separador) hacia el primero. Esto ES resolver H·δ = −g."""
    delta: Dict[tuple, np.ndarray] = {}
    for cond, _ in reversed(resultado):
        rhs = cond["d"].copy()
        if cond["sep"]:
            rhs = rhs - cond["S"] @ np.concatenate(
                [delta[u] for u in cond["sep"]])
        delta[cond["v"]] = np.linalg.solve(cond["R"], rhs)
    return delta


def nnz(resultado: List) -> int:
    """Los no-ceros del factor R (condicionales R y S): la MEDIDA del
    fill-in. Distintos órdenes → distinto nnz, misma solución."""
    total = 0
    for cond, _ in resultado:
        total += int((np.abs(cond["R"]) > 1e-9).sum())
        total += int((np.abs(cond["S"]) > 1e-9).sum())
    return total


# ── los órdenes de eliminación (acto 2) ──────────────────────────────────────


def orden_temporal(mundo: dict) -> List[tuple]:
    """Cada variable en cuanto nace: pose i en el paso i, landmark j justo
    después de su primera observación. El orden 'natural' de un SLAM."""
    primera = {}
    for i, j, _ in mundo["obs"]:
        primera.setdefault(j, i)
    claves = [(("x", i), float(i)) for i in range(len(mundo["inicial"]))]
    claves += [(("l", j), primera[j] + 0.5) for j in sorted(primera)]
    return [v for v, _ in sorted(claves, key=lambda kv: kv[1])]


def orden_landmarks_primero(mundo: dict) -> List[tuple]:
    """El orden del Schur de BA (nivel 11): puntos primero. Óptimo allí
    (miles de puntos, pocas cámaras); aquí es al revés — cada landmark
    acopla a las ~decenas de poses que lo vieron. Malo A PROPÓSITO."""
    lms = sorted({j for _, j, _ in mundo["obs"]})
    return ([("l", j) for j in lms]
            + [("x", i) for i in range(len(mundo["inicial"]))])


def _adyacencia(factores: List[dict]) -> Dict[tuple, set]:
    ady: Dict[tuple, set] = defaultdict(set)
    for f in factores:
        for u in f["vars"]:
            ady[u].update(w for w in f["vars"] if w != u)
            ady.setdefault(u, set())
    return ady


def _orden_glotona(mundo: dict, peor: bool) -> List[tuple]:
    """Min-degree glotón (o max-degree si peor=True), simulando el fill-in:
    al eliminar v, sus vecinos vivos quedan conectados entre sí (el factor
    nuevo del acto 1, visto como grafo). Desempate: el orden temporal."""
    ady = _adyacencia(factores_del_mundo(mundo))
    rango = {v: k for k, v in enumerate(orden_temporal(mundo))}
    vivos = set(ady)
    orden = []
    while vivos:
        v = min(vivos, key=lambda u: ((-1 if peor else 1) * len(ady[u]),
                                      rango[u]))
        orden.append(v)
        vecinos = ady[v] & vivos
        for a in vecinos:
            ady[a].update(vecinos - {a})       # el fill-in, simulado
            ady[a].discard(v)
        vivos.remove(v)
    return orden


def orden_min_degree(mundo: dict) -> List[tuple]:
    return _orden_glotona(mundo, peor=False)


def orden_max_degree(mundo: dict) -> List[tuple]:
    return _orden_glotona(mundo, peor=True)


# ── ensamblar H densa (para las identidades del examen) ──────────────────────


def hessiana(lineales: List, claves: List) -> Tuple[np.ndarray, np.ndarray]:
    """(H, g) densas sobre `claves`: H = ΣAᵀA, g = −ΣAᵀb. Es la misma H del
    nivel 21 (AᵀA = JᵀΛJ porque A está blanqueada)."""
    off, n = {}, 0
    for u in claves:
        off[u] = n
        n += dim(u)
    H = np.zeros((n, n))
    g = np.zeros(n)
    for fvars, bloques, b in lineales:
        for u in fvars:
            g[off[u]:off[u] + dim(u)] -= bloques[u].T @ b
            for w in fvars:
                H[off[u]:off[u] + dim(u), off[w]:off[w] + dim(w)] += (
                    bloques[u].T @ bloques[w])
    return H, g


def demo_schur(mundo: dict, corte: int) -> Dict[str, float]:
    """La identidad del acto 1 en grande: eliminar TODAS las variables del
    pasado (poses < corte y landmarks que solo vio el pasado) deja un grafo
    reducido sobre el presente. Su H ensamblada == el complemento de Schur
    del nivel 21, entrada por entrada."""
    valores = valores_iniciales(mundo)
    fnl = factores_del_mundo(mundo)
    lineales = [linealizar_factor(f, valores) for f in fnl]

    N = len(mundo["inicial"])
    lm_futuro = {j for i, j, _ in mundo["obs"] if i >= corte}
    claves_A = [("x", i) for i in range(corte)]
    claves_A += [("l", j) for j in sorted({j for _, j, _ in mundo["obs"]}
                                          - lm_futuro)]
    claves_B = [("x", i) for i in range(corte, N)]
    claves_B += [("l", j) for j in sorted(lm_futuro)]

    # lado 1: eliminación parcial -> el grafo reducido, ensamblado
    _, sobrantes = eliminar(lineales, claves_A)
    H_e, g_e = hessiana(sobrantes, claves_B)

    # lado 2: el Schur denso del nivel 21
    H, g = hessiana(lineales, claves_A + claves_B)
    nA = sum(dim(u) for u in claves_A)
    H_AA, H_AB, H_BB = H[:nA, :nA], H[:nA, nA:], H[nA:, nA:]
    K = np.linalg.solve(H_AA, H_AB)
    H_s = H_BB - H_AB.T @ K
    g_s = g[nA:] - H_AB.T @ np.linalg.solve(H_AA, g[:nA])

    escala = float(np.abs(H_s).max())
    return {"dif_H": float(np.abs(H_e - H_s).max()) / escala,
            "dif_g": float(np.abs(g_e - g_s).max())
            / max(float(np.abs(g_s).max()), 1.0)}


# ── Gauss-Newton con la eliminación como solver lineal ───────────────────────


def _aplicar(valores: Dict, delta: Dict) -> float:
    norma2 = 0.0
    for v, dv in delta.items():
        norma2 += float(dv @ dv)
        if v[0] == "x":
            valores[v][:2] += dv[:2]
            valores[v][2] = envolver(valores[v][2] + dv[2])
        else:
            valores[v] += dv
    return float(np.sqrt(norma2))


def gauss_newton(mundo: dict, orden: Optional[List] = None,
                 iteraciones: int = 15) -> Dict:
    """El MISMO Gauss-Newton del nivel 21, con UN cambio: el solve denso se
    reemplaza por eliminar + sustituir. Misma solución, otro camino."""
    valores = valores_iniciales(mundo)
    fnl = factores_del_mundo(mundo)
    if orden is None:
        orden = orden_temporal(mundo)
    resultado = []
    for _ in range(iteraciones):
        lineales = [linealizar_factor(f, valores) for f in fnl]
        resultado, _ = eliminar(lineales, orden)
        if _aplicar(valores, resolver(resultado)) < 1e-10:
            break
    N = len(mundo["inicial"])
    M = len(mundo["landmarks"])
    return {"poses": np.array([valores[("x", i)] for i in range(N)]),
            "landmarks": np.array([valores.get(("l", j), np.full(2, np.nan))
                                   for j in range(M)]),
            "resultado": resultado, "orden": orden}


def gauss_newton_batch(mundo: dict, iteraciones: int = 15) -> Dict:
    """La referencia: el batch del nivel 21 tal cual (H densa + solve con su
    misma regularización 1e-9). Contra esto se compara la eliminación."""
    valores = valores_iniciales(mundo)
    fnl = factores_del_mundo(mundo)
    claves = orden_temporal(mundo)
    off, n = {}, 0
    for u in claves:
        off[u] = n
        n += dim(u)
    for _ in range(iteraciones):
        lineales = [linealizar_factor(f, valores) for f in fnl]
        H, g = hessiana(lineales, claves)
        d = np.linalg.solve(H + 1e-9 * np.eye(n), -g)
        delta = {u: d[off[u]:off[u] + dim(u)] for u in claves}
        if _aplicar(valores, delta) < 1e-10:
            break
    N = len(mundo["inicial"])
    M = len(mundo["landmarks"])
    return {"poses": np.array([valores[("x", i)] for i in range(N)]),
            "landmarks": np.array([valores.get(("l", j), np.full(2, np.nan))
                                   for j in range(M)])}
