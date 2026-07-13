#!/usr/bin/env python3
"""Examen del nivel 09: la triangulacion y sus filtros.

Genera los datos si faltan. Numeros medidos (par 0->6): 547/554 puntos
sobreviven, reproyeccion media 0.32 px, y el filtro de paralaje tira el 43%
en el par 0->1 (baseline 5 cm) y 0% en el 0->6 (31 cm).
Si todo pasa: NIVEL 09: VERIFICADO.

Uso:
    python verificacion.py
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
DATOS = AQUI / "data" / "secuencia"

spec = importlib.util.spec_from_file_location("n9", AQUI / "09_triangulacion.py")
n9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n9)

fallos = []


def check(nombre: str, ok: bool, detalle: str = "") -> None:
    estado = "OK " if ok else "FALLO"
    print(f"  [{estado}] {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def main() -> int:
    if not (DATOS / "images").is_dir():
        print("Generando la secuencia sintetica...")
        if subprocess.run([sys.executable, str(AQUI / "genera_datos.py")]).returncode:
            raise SystemExit("genera_datos.py fallo")

    K = n9.leer_calibracion()
    gt = n9.leer_gt()
    print("Verificando la triangulacion (poses del ground truth)\n")

    # 1. El par con buen baseline produce un mapa sano.
    pts0, pts1 = n9.emparejar(n9.cargar_frame(0), n9.cargar_frame(6))
    pts_w, m = n9.triangular(K, gt[0], gt[6], pts0, pts1)
    val = m["valido"]
    check("sobrevive >90% de los matches (par sano)", val.mean() > 0.90,
          f"{int(val.sum())}/{len(pts_w)} = {100*val.mean():.1f}%")

    err = m["error_reproj"][val]
    check("reproyeccion media < 1 px", err.mean() < 1.0,
          f"{err.mean():.3f} px (medido: 0.323)")
    check("ningun superviviente supera el umbral", err.max() < n9.REPROJ_THRESH_PX,
          f"max {err.max():.3f} px")

    # 2. Los puntos triangulados EXPLICAN sus observaciones: re-proyectarlos
    #    en la vista 1 debe caer sobre los pixeles que los generaron.
    uv, Z = n9.proyectar(K, gt[6], pts_w[val])
    d = np.linalg.norm(uv - pts1[val], axis=1)
    check("re-proyectar el mapa cae sobre los matches", np.median(d) < 1.0,
          f"mediana {np.median(d):.3f} px")
    check("todos los puntos del mapa estan DELANTE de la camara", bool((Z > 0).all()),
          f"Z min {Z.min():.2f} m")

    # 3. La escena tiene 3 planos a z = 4, 7 y 14 m: el mapa debe reflejarlo
    #    (comprobacion de que la geometria recuperada es la REAL, no una
    #    nube plausible cualquiera).
    z = pts_w[val][:, 2]
    check("las profundidades caen en el rango de la escena (3..16 m)",
          bool((z > 3).mean() > 0.95 and (z < 16).mean() > 0.95),
          f"z: p5 {np.percentile(z,5):.1f} / p95 {np.percentile(z,95):.1f} m")

    # 4. EL filtro de paralaje se gana el sueldo: con baseline diminuto (par
    #    0->1, 5 cm) tira una fraccion enorme de puntos mal condicionados;
    #    con baseline sano no tira nada.
    qa, qb = n9.emparejar(n9.cargar_frame(0), n9.cargar_frame(1))
    _, m1 = n9.triangular(K, gt[0], gt[1], qa, qb)
    tira_corto = float((~m1["paralaje"]).mean())
    tira_largo = float((~m["paralaje"]).mean())
    check("el filtro de paralaje muerde en baseline corto y no en el largo",
          tira_corto > 0.2 and tira_largo < 0.05,
          f"par 0->1 tira {100*tira_corto:.0f}%, par 0->6 tira {100*tira_largo:.0f}%")

    # 5. La sensibilidad al ruido CAE con el baseline (dZ ~ Z^2/(f*B)).
    rng = np.random.default_rng(0)

    def sensibilidad(jb: int) -> float:
        a, b = n9.emparejar(n9.cargar_frame(0), n9.cargar_frame(jb))
        p1, mm = n9.triangular(K, gt[0], gt[jb], a, b)
        p2, _ = n9.triangular(K, gt[0], gt[jb], a, b + rng.normal(0, 0.5, b.shape))
        v = mm["valido"]
        return float(np.median(np.linalg.norm(p2[v] - p1[v], axis=1)))

    s_corto, s_largo = sensibilidad(1), sensibilidad(30)
    check("mas baseline = punto 3D mas estable (>=5x)", s_corto / s_largo >= 5.0,
          f"5 cm: {100*s_corto:.0f} cm  vs  154 cm: {100*s_largo:.0f} cm")

    # 6. El PLY se escribe y se puede releer.
    salida = AQUI / "salida"
    salida.mkdir(exist_ok=True)
    n9.guardar_ply(pts_w[val], salida / "_test.ply")
    txt = (salida / "_test.ply").read_text(encoding="utf-8").splitlines()
    ok_ply = txt[0] == "ply" and f"element vertex {int(val.sum())}" in txt
    (salida / "_test.ply").unlink()
    check("el PLY se escribe con la cabecera correcta", ok_ply)

    print()
    if fallos:
        print(f"NIVEL 09: {len(fallos)} fallo(s): {', '.join(fallos)}")
        return 1
    print("NIVEL 09: VERIFICADO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
