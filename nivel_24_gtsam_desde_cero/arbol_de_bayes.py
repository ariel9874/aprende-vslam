"""EL ÁRBOL DE BAYES: la estructura de datos que hace posible iSAM2 (acto 3).

La eliminación (eliminacion.py) convierte el grafo de factores en una RED DE
BAYES: un condicional p(v | separador) por variable. Este archivo la agrupa
en CLIQUES y descubre que forman un ÁRBOL — la representación con la que
GTSAM piensa.

─── La construcción (Kaess et al., 2012) ─────────────────────────────────────
Recorre los condicionales en orden INVERSO de eliminación:

  - sin separador → la RAÍZ del árbol.
  - con separador S → su PADRE es el clique donde vive (como frontal) la
    variable de S eliminada primero. Si S es exactamente TODO el padre
    (frontales + separador), el condicional se FUSIONA en él (cliques
    maximales); si no, nace un clique hijo [v | S].

Propiedad que lo hace árbol (y que el código asserta): el separador de cada
clique está CONTENIDO en su padre — la información fluye solo por el camino
hacia la raíz.

─── Por qué importa (la revelación estructural) ──────────────────────────────
1. RESOLVER = bajar desde la raíz: cada clique despeja sus frontales usando
   su separador, que ya resolvió un ancestro. Misma sustitución hacia atrás
   del acto 1, ahora organizada por el árbol (el examen verifica identidad).

2. Un factor NUEVO solo invalida los cliques desde donde toca hasta la RAÍZ.
   Todo lo que cuelga fuera de ese camino NI SE ENTERA: sus condicionales
   siguen siendo válidos. Ese es el corazón de iSAM2 (acto 4): la odometría
   toca un camino corto; un cierre de bucle, un camino largo — el costo del
   loop closure ES la forma del árbol, y aquí se cuenta en cliques.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np


def construir(resultado: List, orden: List) -> Tuple[List[dict], Dict]:
    """El árbol desde los condicionales de `eliminar`. Devuelve
    (cliques, clique_de): cada clique con frontales/separador/padre/hijos,
    y el mapa variable → clique donde es frontal."""
    pos = {v: k for k, v in enumerate(orden)}
    cliques: List[dict] = []
    clique_de: Dict[tuple, int] = {}
    for cond, _ in reversed(resultado):
        v, sep = cond["v"], cond["sep"]
        if not sep:
            cliques.append({"frontales": [v], "separador": [],
                            "padre": None, "hijos": []})
            clique_de[v] = len(cliques) - 1
            continue
        w = min(sep, key=lambda u: pos[u])       # la eliminada primero
        p = clique_de[w]
        vars_padre = set(cliques[p]["frontales"]) | set(cliques[p]["separador"])
        assert set(sep) <= vars_padre, "separador fuera del padre (no-arbol)"
        if set(sep) == vars_padre:
            cliques[p]["frontales"].append(v)    # fusion: clique maximal
            clique_de[v] = p
        else:
            cliques.append({"frontales": [v], "separador": list(sep),
                            "padre": p, "hijos": []})
            clique_de[v] = len(cliques) - 1
            cliques[p]["hijos"].append(len(cliques) - 1)
    return cliques, clique_de


def profundidad(cliques: List[dict]) -> List[int]:
    """La profundidad de cada clique (raíz = 0), en orden de construcción
    (el padre siempre se construye antes que el hijo)."""
    prof = [0] * len(cliques)
    for k, c in enumerate(cliques):
        prof[k] = 0 if c["padre"] is None else prof[c["padre"]] + 1
    return prof


def resolver_por_arbol(cliques: List[dict], resultado: List,
                       orden: List) -> Dict[tuple, np.ndarray]:
    """Bajar desde la raíz: al llegar a un clique, su separador ya está
    resuelto por sus ancestros. Es la sustitución hacia atrás del acto 1,
    reorganizada — el examen comprueba que da EXACTAMENTE lo mismo."""
    pos = {v: k for k, v in enumerate(orden)}
    cond_de = {cond["v"]: cond for cond, _ in resultado}
    delta: Dict[tuple, np.ndarray] = {}
    pila = [k for k, c in enumerate(cliques) if c["padre"] is None]
    while pila:
        c = cliques[pila.pop()]
        # dentro del clique, el frontal eliminado MAS TARDE primero (sus
        # separadores internos son frontales posteriores, ya resueltos)
        for v in sorted(c["frontales"], key=lambda u: -pos[u]):
            cond = cond_de[v]
            rhs = cond["d"].copy()
            if cond["sep"]:
                rhs = rhs - cond["S"] @ np.concatenate(
                    [delta[u] for u in cond["sep"]])
            delta[v] = np.linalg.solve(cond["R"], rhs)
        pila += c["hijos"]
    return delta


def afectados(claves: List[tuple], cliques: List[dict],
              clique_de: Dict) -> Set[int]:
    """Los cliques que un factor nuevo sobre `claves` invalida: el camino de
    cada clique tocado hasta la raíz (variables aún fuera del árbol no
    invalidan nada: son nuevas). El resto del árbol NI SE ENTERA."""
    marcados: Set[int] = set()
    for v in claves:
        k = clique_de.get(v)
        while k is not None and k not in marcados:
            marcados.add(k)
            k = cliques[k]["padre"]
    return marcados


def arbol_del_mundo(mundo: dict, orden: List = None,
                    solo_odometria: bool = False) -> dict:
    """Composición de conveniencia: linealizar el mundo (o solo su cadena de
    odometría), eliminar, construir el árbol y medir su forma."""
    import eliminacion as el

    fnl = el.factores_del_mundo(mundo)
    if solo_odometria:
        fnl = [f for f in fnl if f["tipo"] != "obs"]
    valores = el.valores_iniciales(mundo)
    lineales = [el.linealizar_factor(f, valores) for f in fnl]
    if orden is None:
        orden = [("x", i) for i in range(len(mundo["inicial"]))] \
            if solo_odometria else el.orden_temporal(mundo)
    resultado, _ = el.eliminar(lineales, orden)
    cliques, clique_de = construir(resultado, orden)
    tam = [len(c["frontales"]) + len(c["separador"]) for c in cliques]
    return {"resultado": resultado, "orden": orden, "cliques": cliques,
            "clique_de": clique_de, "prof": profundidad(cliques),
            "tam_max": max(tam)}
