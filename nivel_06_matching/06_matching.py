#!/usr/bin/env python3
"""
Nivel 06 — Matching
===================

Empareja descriptores ORB entre dos frames reales separados ~1 s y mide
cuánto limpia el ratio test de Lowe:

    1. matching por fuerza bruta (Hamming) con y sin ratio test
    2. curva medida: matches supervivientes vs umbral de ratio
    3. residuo: cuantos matches SOSPECHOSOS sobreviven (lo que el ratio
       no puede ver, y el nivel 07 si)

Uso:
    python 06_matching.py [--root <secuencia_TUM>] [--gap 30] [--ratio 0.75]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent
DATASET_DEFAULT = AQUI / "data" / "rgbd_dataset_freiburg1_xyz"


def cargar_par(root: Path, gap: int) -> tuple[np.ndarray, np.ndarray, str, str]:
    rgb = sorted((root / "rgb").glob("*.png"))
    if len(rgb) <= gap:
        raise SystemExit(f"No hay suficientes imagenes en {root / 'rgb'}. "
                         "Corre `python descarga_datos.py` o pasa --root.")
    a, b = rgb[0], rgb[gap]
    return (cv2.imread(str(a), cv2.IMREAD_GRAYSCALE),
            cv2.imread(str(b), cv2.IMREAD_GRAYSCALE), a.name, b.name)


def emparejar(desc_a: np.ndarray, desc_b: np.ndarray, ratio: float
              ) -> tuple[list, list]:
    """Devuelve (todos_los_mejores, supervivientes_del_ratio).

    ─── La matemática: distancia de Hamming y el ratio de Lowe ───
    Cada descriptor ORB son 256 bits. La distancia entre dos es el número de
    bits distintos: popcount(a XOR b) — una instrucción de CPU por palabra,
    por eso el matching binario es tan barato. Dos parches del MISMO punto
    físico difieren en pocos bits (ruido, iluminación); dos parches
    cualesquiera difieren en ~128 (la mitad, puro azar).

    El problema del vecino más cercano: SIEMPRE hay un "más parecido",
    aunque el punto ya no sea visible. El test de Lowe compara el mejor
    candidato d1 contra el segundo d2:

        aceptar  ⇔  d1 < ratio · d2      (ratio ≈ 0.75)

    Si el match es real hay UN doble claro (d1 ≪ d2). Si no lo hay — punto
    fuera de campo, textura repetitiva — el mejor y el segundo son impostores
    parecidos (d1 ≈ d2) y se descarta. Mide AMBIGÜEDAD, no verdad: por eso
    el nivel 07 añade la verificación geométrica.
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    pares = bf.knnMatch(desc_a, desc_b, k=2)
    todos = [m for m, _ in pares]
    buenos = [m for m, n in pares if m.distance < ratio * n.distance]
    return todos, buenos


def desplazamientos(matches: list, kps_a: list, kps_b: list) -> np.ndarray:
    """Vector de desplazamiento (dx, dy) de cada match, en píxeles."""
    pa = np.float64([kps_a[m.queryIdx].pt for m in matches])
    pb = np.float64([kps_b[m.trainIdx].pt for m in matches])
    return pb - pa


def fraccion_sospechosos(desp: np.ndarray, k_mad: float = 3.0) -> float:
    """Fracción de matches cuyo desplazamiento se sale de la manada.

    Entre frames cercanos, los matches VERDADEROS se mueven de forma
    coherente (la cámara indujo un flujo suave); los falsos apuntan a
    cualquier parte. Robusto a los propios outliers: mediana +- k·MAD
    (desviación absoluta mediana) por componente. Es un PROXY didáctico de
    lo que RANSAC hará con rigor geométrico en el nivel 07.
    """
    if len(desp) == 0:
        return 0.0
    med = np.median(desp, axis=0)
    mad = np.median(np.abs(desp - med), axis=0) + 1e-9
    fuera = np.any(np.abs(desp - med) > k_mad * 1.4826 * mad, axis=1)
    return float(fuera.mean())


def main() -> int:
    parser = argparse.ArgumentParser(description="Nivel 06: matching")
    parser.add_argument("--root", default=str(DATASET_DEFAULT))
    parser.add_argument("--gap", type=int, default=30, help="frames de separacion (~1 s)")
    parser.add_argument("--ratio", type=float, default=0.75)
    args = parser.parse_args()

    gray_a, gray_b, na, nb = cargar_par(Path(args.root), args.gap)
    salida = AQUI / "salida"
    salida.mkdir(exist_ok=True)
    print(f"Par: {na}  ->  {nb}  (gap {args.gap} frames)\n")

    orb = cv2.ORB_create(nfeatures=2000)
    kps_a, desc_a = orb.detectAndCompute(gray_a, None)
    kps_b, desc_b = orb.detectAndCompute(gray_b, None)
    print(f"Keypoints: {len(kps_a)} / {len(kps_b)}")

    # ── 1 · Antes y después del ratio ─────────────────────────────────────────
    todos, buenos = emparejar(desc_a, desc_b, args.ratio)
    print(f"Matches (mejor vecino, sin filtro): {len(todos)}")
    print(f"Matches tras ratio {args.ratio}:            {len(buenos)} "
          f"({100*len(buenos)/max(1,len(todos)):.0f}% sobreviven)")

    # El numero de la leccion: el ratio elimina sobre todo BASURA.
    f_antes = fraccion_sospechosos(desplazamientos(todos, kps_a, kps_b))
    f_despues = fraccion_sospechosos(desplazamientos(buenos, kps_a, kps_b))
    print(f"Matches sospechosos (desplazamiento fuera de la manada, 3xMAD):")
    print(f"   antes del ratio:  {100*f_antes:.0f}%")
    print(f"   despues:          {100*f_despues:.0f}%   <- esto compra el ratio test")

    # Visual: los 60 mejores de cada lado, apilados.
    top = sorted(buenos, key=lambda m: m.distance)[:60]
    top_todos = sorted(todos, key=lambda m: m.distance)[:60]
    vis_antes = cv2.drawMatches(gray_a, kps_a, gray_b, kps_b, top_todos, None,
                                matchColor=(0, 165, 255), flags=2)
    vis_desp = cv2.drawMatches(gray_a, kps_a, gray_b, kps_b, top, None,
                               matchColor=(0, 255, 0), flags=2)
    cv2.putText(vis_antes, "sin ratio (60 mejores)", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(vis_desp, f"con ratio {args.ratio} (60 mejores)", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(str(salida / "matches_antes_despues.png"),
                np.vstack([vis_antes, vis_desp]))

    # ── 2 · La curva del ratio ────────────────────────────────────────────────
    umbrales = np.arange(0.50, 0.96, 0.05)
    supervivientes = []
    sospechosos = []
    for r in umbrales:
        _, b = emparejar(desc_a, desc_b, float(r))
        supervivientes.append(len(b))
        sospechosos.append(fraccion_sospechosos(desplazamientos(b, kps_a, kps_b)))
    print("\nCurva del ratio (umbral: matches / % sospechosos):")
    for r, s, f in zip(umbrales, supervivientes, sospechosos):
        print(f"   {r:.2f}: {s:5d} / {100*f:4.0f}%")
    print("Lee la curva con cuidado (es una U, no una recta): pasado ~0.85 la")
    print("basura EXPLOTA (mas laxo = impostores dentro); y en el extremo")
    print("estricto el % tambien sube — pero por el METRO, no por los matches:")
    print("con pocos matches ultra-coherentes el MAD se encoge y el detector de")
    print("'fuera de la manada' se vuelve paranoico (ver EJERCICIOS 2).")
    print("0.7-0.8 es el compromiso clasico (Lowe 2004 midio 0.8 en SIFT).")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(umbrales, supervivientes, "o-", color="tab:blue")
        ax1.set_xlabel("umbral de ratio"), ax1.set_ylabel("matches", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(umbrales, [100 * f for f in sospechosos], "s--", color="tab:red")
        ax2.set_ylabel("% sospechosos", color="tab:red")
        ax1.set_title("El precio del umbral de ratio")
        ax1.grid(True, alpha=0.3)
        fig.savefig(salida / "curva_ratio.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        print("[aviso] matplotlib no instalado: se omite curva_ratio.png")

    print(f"\nGuardado en {salida}: matches_antes_despues.png, curva_ratio.png")
    print("Ahora corre `python verificacion.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
