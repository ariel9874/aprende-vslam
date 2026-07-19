#!/usr/bin/env python3
"""
Nivel 24 (bonus) — GTSAM desde cero: eliminación, árbol de Bayes, iSAM
======================================================================

El nivel 21 dejó la matemática del grafo de factores; este nivel construye
la MÁQUINA que la resuelve — lo que GTSAM aporta de verdad:

    acto 1  eliminar variables (el grafo -> red de Bayes; Schur con otro nombre)
    acto 2  el orden importa (el fill-in, medido y controlado: min-degree)
    acto 3  el árbol de Bayes (un factor nuevo solo toca su camino al root)
    acto 4  iSAM de juguete (re-resolver solo lo que cambió, EN LÍNEA)
    acto 5  GTSAM real (docker compose up — ver README)

    python 24_gtsam.py          # actos 1-4: tablas + graficas en salida/
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import arbol_de_bayes as ab
import eliminacion as el
import filtro_ekf
import isam_de_juguete as isam
from mundo import generar, rmse_xy

AQUI = Path(__file__).resolve().parent


def main() -> int:
    m = generar()
    gt = m["gt"]
    N = len(m["inicial"])
    print(f"El mundo del nivel 21: {N} poses, {len(m['landmarks'])} "
          f"landmarks, {len(m['odo'])} odo, {len(m['obs'])} observaciones\n")

    # ── acto 1: eliminar ES resolver ─────────────────────────────────────────
    print("[acto 1] Eliminacion de variables")
    t0 = time.perf_counter()
    re_ = el.gauss_newton(m)
    t_e = time.perf_counter() - t0
    t0 = time.perf_counter()
    rb = el.gauss_newton_batch(m)
    t_b = time.perf_counter() - t0
    dif = float(np.abs(re_["poses"] - rb["poses"]).max())
    print(f"  GN por eliminacion: {100*rmse_xy(re_['poses'], gt):.2f} cm "
          f"({t_e:.2f} s) | batch del 21: "
          f"{100*rmse_xy(rb['poses'], gt):.2f} cm ({t_b:.2f} s)")
    print(f"  misma solucion: dif maxima {dif:.1e} — eliminar ES resolver")
    r = el.demo_schur(m, N // 2)
    print(f"  el factor al separador == Schur del 21: dif rel "
          f"{r['dif_H']:.1e} (H), {r['dif_g']:.1e} (g)\n")

    # ── acto 2: el orden importa ─────────────────────────────────────────────
    print("[acto 2] El fill-in por orden de eliminacion (mismo grafo)")
    valores = el.valores_iniciales(m)
    lineales = [el.linealizar_factor(f, valores)
                for f in el.factores_del_mundo(m)]
    ordenes = [("temporal (el de un SLAM)", el.orden_temporal(m)),
               ("landmarks primero (BA n11)", el.orden_landmarks_primero(m)),
               ("min-degree (COLAMD de juguete)", el.orden_min_degree(m)),
               ("max-degree (el peor, a proposito)", el.orden_max_degree(m))]
    nnzs = {}
    for nombre, orden in ordenes:
        t0 = time.perf_counter()
        res, _ = el.eliminar(lineales, orden)
        dt = time.perf_counter() - t0
        nnzs[nombre] = el.nnz(res)
        print(f"  {nombre:34s} nnz(R) = {nnzs[nombre]:6d}   ({dt*1000:5.0f} ms)")
    print("  mismo grafo, misma solucion — el orden solo cambia el COSTO.")
    print("  (el temporal paga los bucles: las re-observaciones de la 2a")
    print("  vuelta acoplan las dos vueltas; min-degree lo esquiva solo)\n")

    # ── acto 3: el arbol de Bayes ────────────────────────────────────────────
    print("[acto 3] El arbol de Bayes (cliques afectados por un factor nuevo)")
    arboles = [("cadena de odometria", ab.arbol_del_mundo(m, solo_odometria=True)),
               ("completo, orden temporal", ab.arbol_del_mundo(m)),
               ("completo, min-degree",
                ab.arbol_del_mundo(m, orden=el.orden_min_degree(m)))]
    for nombre, a in arboles:
        cl, cd = a["cliques"], a["clique_de"]
        odo = ab.afectados([("x", N - 1)], cl, cd)
        bucle = ab.afectados([("x", 5), ("x", N - 1)], cl, cd)
        print(f"  {nombre:26s} {len(cl):3d} cliques (prof {max(a['prof']):3d}) | "
              f"odo nueva: {len(odo):3d} | bucle x5-x{N-1}: {len(bucle):3d} "
              f"({100*len(bucle)/len(cl):3.0f}%)")
    print("  la odometria toca su camino al root; el bucle paga el suyo —")
    print("  y un buen orden (min-degree) acorta TODOS los caminos.\n")

    # ── acto 4: iSAM de juguete, y la trilogia ───────────────────────────────
    print("[acto 4] iSAM de juguete: el circuito EN LINEA, pose a pose")
    filas = []
    for vueltas in (2, 3):
        mv = generar(vueltas=vueltas) if vueltas != 2 else m
        gtv = mv["gt"]
        ri = isam.correr(mv)
        rbv = isam.correr_batch(mv)
        reelim = np.array([s["reelim"] for s in ri["stats"]])
        filas.append((vueltas, len(mv["inicial"]), ri, rbv, gtv, reelim))
        print(f"  vueltas={vueltas}: isam {100*rmse_xy(ri['tray'], gtv):.2f} cm "
              f"({ri['t']:.2f} s) | batch/paso {100*rmse_xy(rbv['tray'], gtv):.2f} cm "
              f"({rbv['t']:.2f} s) -> speedup {rbv['t']/ri['t']:.1f}x")
        print(f"           reelim por paso: mediana {np.median(reelim):.0f} de "
              f"{len(mv['inicial'])+14} vars | pico (cierre de bucle): "
              f"{reelim.max()}")
    print("  el speedup CRECE con el viaje: el batch paga todo el grafo en")
    print("  cada paso; iSAM, solo el camino afectado (y los picos del bucle).\n")

    # la tabla que cierra la trilogia 21/23/24 (mismas medidas)
    vueltas2 = filas[0]
    ri, rbv = vueltas2[2], vueltas2[3]
    t0 = time.perf_counter()
    rf = filtro_ekf.correr(m)
    t_f = time.perf_counter() - t0
    print("  la trilogia de estimacion (online = cada pose al emitirse;")
    print("  final = la trayectoria refinada al terminar):")
    print(f"  {'estimador':34s} {'online':>8s} {'final':>8s} {'tiempo':>8s}")
    print(f"  {'FILTRO EKF (nivel 23)':34s} "
          f"{100*rmse_xy(rf['tray'], gt):6.1f}cm {'no hay':>8s} {t_f:7.2f}s")
    print(f"  {'iSAM de juguete (este nivel)':34s} "
          f"{100*rmse_xy(ri['tray'], gt):6.1f}cm "
          f"{100*rmse_xy(ri['poses'], gt):6.1f}cm {ri['t']:7.2f}s")
    print(f"  {'batch por paso (nivel 21 en linea)':34s} "
          f"{100*rmse_xy(rbv['tray'], gt):6.1f}cm "
          f"{100*rmse_xy(rbv['poses'], gt):6.1f}cm {rbv['t']:7.2f}s")
    print("  online, los tres casi empatan (nadie reescribe lo emitido);")
    print("  la diferencia es lo que ADEMAS te llevas: el filtro nada, el")
    print("  batch todo (pero re-pagandolo), iSAM todo a precio incremental.")

    # ── graficas ─────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))

        # (a) el arbol min-degree, con el camino del bucle en rojo
        a = arboles[2][1]
        cl, cd = a["cliques"], a["clique_de"]
        rojo = ab.afectados([("x", 5), ("x", N - 1)], cl, cd)
        xs = [0.0] * len(cl)
        sig = [0]

        def acomodar(k):     # x = orden DFS de hojas; y = -profundidad
            hijos = cl[k]["hijos"]
            if not hijos:
                xs[k] = sig[0]
                sig[0] += 1
                return xs[k]
            xs[k] = float(np.mean([acomodar(h) for h in hijos]))
            return xs[k]

        for k, c in enumerate(cl):
            if c["padre"] is None:
                acomodar(k)
        prof = a["prof"]
        ax = axs[0]
        for k, c in enumerate(cl):
            if c["padre"] is not None:
                p = c["padre"]
                ax.plot([xs[k], xs[p]], [-prof[k], -prof[p]],
                        color="0.75", lw=0.7, zorder=1)
        col = ["tab:red" if k in rojo else "tab:blue" for k in range(len(cl))]
        tam = [22 * (len(c["frontales"]) + len(c["separador"])) ** 0.5
               for c in cl]
        ax.scatter(xs, [-p for p in prof], c=col, s=tam, zorder=2)
        ax.set_title(f"el arbol de Bayes (min-degree): {len(cl)} cliques\n"
                     f"en rojo: lo que un bucle x5-x{N-1} recomputa "
                     f"({len(rojo)})")
        ax.axis("off")

        # (b) costo por paso: el pico del cierre de bucle
        ax = axs[1]
        reelim = np.array([s["reelim"] for s in ri["stats"]])
        ax.plot(np.array([s["n"] for s in ri["stats"]]), color="0.6", lw=1.2,
                label="batch: TODO el grafo")
        ax.plot(reelim, color="tab:blue", lw=1.2,
                label="iSAM: variables re-eliminadas")
        pico = int(reelim.argmax())
        ax.annotate("cierre de bucle", (pico, reelim[pico]),
                    textcoords="offset points", xytext=(-70, 8), fontsize=8,
                    arrowprops=dict(arrowstyle="->", lw=0.8))
        ax.set_xlabel("paso"), ax.set_ylabel("variables tocadas")
        ax.legend(fontsize=8), ax.grid(alpha=0.3)
        ax.set_title("el costo por paso: mediana ~8, pico en el bucle")

        # (c) fill-in por orden
        ax = axs[2]
        nombres = ["temporal", "landmarks\nprimero", "min-degree",
                   "max-degree"]
        vals = list(nnzs.values())
        barras = ax.bar(nombres, vals,
                        color=["0.6", "0.6", "tab:green", "tab:red"])
        for b, v in zip(barras, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("nnz del factor R")
        ax.set_title("el fill-in: mismo grafo, orden distinto\n"
                     "(la solucion es identica; el costo no)")
        ax.grid(alpha=0.3, axis="y")

        salida = AQUI / "salida"
        salida.mkdir(exist_ok=True)
        fig.savefig(salida / "gtsam.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\nGraficas: {salida / 'gtsam.png'}")
    except ImportError:
        pass

    print("Ahora corre `python verificacion.py` (y el acto 5: README).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
