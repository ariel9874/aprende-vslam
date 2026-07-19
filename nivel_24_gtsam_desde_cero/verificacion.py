#!/usr/bin/env python3
"""Examen del nivel 24: la máquina de GTSAM, verificada pieza por pieza.

Sin gtsam y sin dataset (el mundo es el del 21, sintético y sembrado):
corre en segundos. El acto 5 (GTSAM real) es opcional con --docker.

  1. ELIMINAR ES RESOLVER: la eliminación reproduce el batch del nivel 21
     (misma solución a precisión de máquina) y el factor que deja sobre el
     separador ES el complemento de Schur del 21, entrada por entrada.
  2. EL ORDEN: los cuatro órdenes dan la MISMA solución; el fill-in cambia
     ~11x entre el mejor (min-degree) y el peor — contado en no-ceros.
  3. EL ÁRBOL: resolver bajando desde la raíz == la sustitución del acto 1;
     una odometría toca 1 clique, un bucle casi todo el árbol (medidos) —
     y min-degree acorta todos los caminos.
  4. iSAM DE JUGUETE: misma trayectoria que el batch por paso (tolerancia
     medida), speedup >= 4x, y el pico del cierre de bucle >= 5x la mediana.
  5. (--docker) GTSAM real en el contenedor: su ATE == el nuestro (en mm).

Uso:
    python verificacion.py [--docker]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import arbol_de_bayes as ab
import eliminacion as el
import isam_de_juguete as isam
from mundo import generar, rmse_xy

fallos = []


def check(nombre: str, ok: bool, detalle: str = "") -> None:
    estado = "OK " if ok else "FALLO"
    print(f"  [{estado}] {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker", action="store_true",
                        help="incluir el acto 5 (GTSAM real en contenedor)")
    args = parser.parse_args()
    print("Verificando la maquina de GTSAM (sin gtsam, sin dataset)\n")

    m = generar()
    gt = m["gt"]
    N = len(m["inicial"])

    # ── [1/5] eliminar es resolver ───────────────────────────────────────────
    print("[1/5] Eliminacion: el grafo -> red de Bayes...")
    valores = el.valores_iniciales(m)
    lineales = [el.linealizar_factor(f, valores)
                for f in el.factores_del_mundo(m)]
    orden = el.orden_temporal(m)
    resultado, sobr = el.eliminar(lineales, orden)
    delta_e = el.resolver(resultado)
    H, g = el.hessiana(lineales, orden)
    d = np.linalg.solve(H, -g)
    off, difs = 0, []
    for u in orden:
        difs.append(float(np.abs(delta_e[u] - d[off:off + el.dim(u)]).max()))
        off += el.dim(u)
    check("un solve lineal: eliminacion == denso (y el grafo queda vacio)",
          max(difs) < 1e-9 and not sobr,
          f"dif maxima {max(difs):.1e} (medido: 1.5e-14)")

    re_ = el.gauss_newton(m)
    rb = el.gauss_newton_batch(m)
    rmse_e = rmse_xy(re_["poses"], gt)
    dif = float(np.abs(re_["poses"] - rb["poses"]).max())
    print(f"  GN por eliminacion: {100*rmse_e:.2f} cm | batch del 21: "
          f"{100*rmse_xy(rb['poses'], gt):.2f} cm")
    check("Gauss-Newton entero: eliminacion == batch del nivel 21",
          dif < 1e-9 and rmse_e < 0.15,
          f"dif maxima {dif:.1e}; RMSE {100*rmse_e:.2f} cm (el 8.2 del 21)")

    r = el.demo_schur(m, N // 2)
    check("el factor al separador == complemento de Schur del 21",
          r["dif_H"] < 1e-9 and r["dif_g"] < 1e-9,
          f"dif rel {r['dif_H']:.1e} — la marginalizacion del 21 era "
          "eliminacion con otro nombre")

    # ── [2/5] el orden importa ───────────────────────────────────────────────
    print("\n[2/5] El orden de eliminacion (fill-in medido)...")
    nnzs, difs_o = {}, []
    for nombre, ofn in [("temporal", el.orden_temporal),
                        ("landmarks_primero", el.orden_landmarks_primero),
                        ("min_degree", el.orden_min_degree),
                        ("max_degree", el.orden_max_degree)]:
        res, _ = el.eliminar(lineales, ofn(m))
        nnzs[nombre] = el.nnz(res)
        de = el.resolver(res)
        difs_o.append(max(float(np.abs(de[u] - delta_e[u]).max())
                          for u in orden))
    print("  nnz(R): " + " | ".join(f"{k} {v}" for k, v in nnzs.items()))
    check("los cuatro ordenes dan la MISMA solucion",
          max(difs_o) < 1e-8, f"dif maxima {max(difs_o):.1e} — el orden "
          "cambia el costo, jamas la respuesta")
    check("min-degree esquiva el fill-in (< 0.25x el temporal)",
          nnzs["min_degree"] < 0.25 * nnzs["temporal"],
          f"{nnzs['min_degree']} vs {nnzs['temporal']} "
          f"({nnzs['min_degree']/nnzs['temporal']:.2f}x; medido: 0.11x — el "
          "temporal paga los bucles de la 2a vuelta)")
    check("el peor orden >= 10x el mejor (contado)",
          nnzs["max_degree"] >= 10 * nnzs["min_degree"],
          f"{nnzs['max_degree']/nnzs['min_degree']:.1f}x (medido: 11.1x)")

    # ── [3/5] el arbol de Bayes ──────────────────────────────────────────────
    print("\n[3/5] El arbol de Bayes...")
    delta_a = ab.resolver_por_arbol(*(lambda a: (a["cliques"], a["resultado"],
                                                 a["orden"]))(
        ab.arbol_del_mundo(m)))
    dif = max(float(np.abs(delta_a[u] - delta_e[u]).max()) for u in orden)
    check("resolver bajando desde la raiz == la sustitucion del acto 1",
          dif < 1e-12, f"dif maxima {dif:.1e}")

    cad = ab.arbol_del_mundo(m, solo_odometria=True)
    odo = ab.afectados([("x", N - 1)], cad["cliques"], cad["clique_de"])
    bucle = ab.afectados([("x", 5), ("x", N - 1)], cad["cliques"],
                         cad["clique_de"])
    check("cadena: una odometria nueva toca O(1) cliques del arbol",
          len(odo) <= 3 and len(cad["cliques"]) >= 80,
          f"{len(odo)} de {len(cad['cliques'])} cliques")
    check("cadena: el bucle x5-x104 paga su camino entero (>= 30x la odo)",
          len(bucle) >= 30 * len(odo),
          f"{len(bucle)} cliques ({100*len(bucle)/len(cad['cliques']):.0f}% "
          "del arbol) — el costo del loop closure ES la forma del arbol")
    md = ab.arbol_del_mundo(m, orden=el.orden_min_degree(m))
    bucle_md = ab.afectados([("x", 5), ("x", N - 1)], md["cliques"],
                            md["clique_de"])
    check("min-degree acorta el camino del bucle (<= 15% del arbol)",
          len(bucle_md) <= 0.15 * len(md["cliques"]),
          f"{len(bucle_md)} de {len(md['cliques'])} cliques "
          f"({100*len(bucle_md)/len(md['cliques']):.0f}%; en la cadena era "
          f"{100*len(bucle)/len(cad['cliques']):.0f}%)")

    # ── [4/5] iSAM de juguete ────────────────────────────────────────────────
    print("\n[4/5] iSAM de juguete: en linea, pose a pose...")
    ri = isam.correr(m)
    rbv = isam.correr_batch(m)
    dif_fin = float(np.abs(ri["poses"][:, :2] - rbv["poses"][:, :2]).max())
    dif_onl = float(np.abs(ri["tray"][:, :2] - rbv["tray"][:, :2]).max())
    rmse_i = rmse_xy(ri["poses"], gt)
    print(f"  isam: online {100*rmse_xy(ri['tray'], gt):.2f} cm, final "
          f"{100*rmse_i:.2f} cm, {ri['t']:.2f} s | batch/paso: "
          f"{rbv['t']:.2f} s")
    check("la trayectoria final == la del batch por paso (< 1 cm)",
          dif_fin < 0.01 and rmse_i < 0.15,
          f"dif maxima {100*dif_fin:.2f} cm (medido: 0.12) — el precio del "
          "umbral de re-linealizacion, medido")
    check("la trayectoria ONLINE tambien empata (< 2 cm)",
          dif_onl < 0.02, f"dif maxima {100*dif_onl:.2f} cm (medido: 0.35)")
    check("speedup >= 4x sobre re-resolver todo en cada paso",
          rbv["t"] >= 4 * ri["t"],
          f"{rbv['t']/ri['t']:.1f}x (medido: ~13x, y CRECE con el viaje)")
    reelim = np.array([s["reelim"] for s in ri["stats"]])
    mediana = float(np.median(reelim))
    check("el paso tipico re-elimina una fraccion minima del grafo",
          mediana <= 15 and len(ri["stats"]) > 100,
          f"mediana {mediana:.0f} de {N+14} variables")
    check("el pico del cierre de bucle es real (>= 5x la mediana)",
          reelim.max() >= 5 * mediana,
          f"pico {reelim.max()} en el paso {int(reelim.argmax())} "
          f"({reelim.max()/mediana:.0f}x la mediana) — cada vuelta paga su "
          "cierre")

    # ── [5/5] GTSAM real (opcional) ──────────────────────────────────────────
    if args.docker:
        print("\n[5/5] Docker: GTSAM de verdad sobre el mismo mundo...")
        r = subprocess.run(["docker", "build", "-t", "aprende-vslam-gtsam",
                            "."], cwd=AQUI, capture_output=True, text=True)
        check("la imagen construye (python:3.11-slim + gtsam pip Linux)",
              r.returncode == 0,
              (r.stderr or r.stdout).strip().splitlines()[-1][:70]
              if (r.stderr or r.stdout) else "")
        if r.returncode == 0:
            r = subprocess.run(["docker", "run", "--rm",
                                "aprende-vslam-gtsam"],
                               cwd=AQUI, capture_output=True, text=True)
            salida = r.stdout.strip()
            print("  " + "\n  ".join(salida.splitlines()[-6:]))
            check("GTSAM da NUESTRO numero (ACTO5_OK: dif de ATE en mm)",
                  r.returncode == 0 and "ACTO5_OK" in salida,
                  "el examen de fuera valida al de dentro")
    else:
        print("\n[5/5] SALTADO: GTSAM real (pasa --docker con el daemon "
              "corriendo).\n      La demo: docker compose up --build "
              "(ver README).")

    print()
    if fallos:
        print(f"NIVEL 24: {len(fallos)} fallo(s): {', '.join(fallos)}")
        return 1
    print("NIVEL 24: VERIFICADO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
