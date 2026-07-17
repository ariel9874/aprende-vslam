#!/usr/bin/env python3
"""
Nivel 19 — Mapa denso 3DGS: renderiza y compara
===============================================

El experimento central del nivel, para VER (no solo pasar el examen): un
conjunto de gaussianas aleatorias aprende a re-sintetizar una imagen objetivo
por puro descenso de gradiente a través del rasterizador diferenciable.

    python 19_gaussianas.py            # sobreajuste 64x64 + graficas
    python 19_gaussianas.py --iters 3000 --n 500

Salida en salida/: objetivo vs render (inicial, intermedio, final) y la
curva PSNR(iteración). Con GPU tarda segundos; en CPU, unos minutos.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Nivel 19: 3DGS")
    parser.add_argument("--iters", type=int, default=1500)
    parser.add_argument("--n", type=int, default=300,
                        help="gaussianas del modelo")
    parser.add_argument("--res", type=int, default=64)
    args = parser.parse_args()

    import torch
    from render_gaussianas import psnr, render

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"dispositivo: {dev}"
          + ("" if dev == "cuda" else "  (CPU: unos minutos)"))
    torch.manual_seed(0)
    H = W = args.res
    f, c = 60.0 * args.res / 48, args.res / 2.0
    K = torch.tensor([[f, 0, c], [0, f, c], [0, 0, 1.0]], device=dev)
    T = torch.eye(4, device=dev)

    def gaussianas_en_frustum(n):
        z = torch.rand(n, 1, device=dev) * 2.0 + 2.0
        xy = (torch.rand(n, 2, device=dev) - 0.5) * 1.6
        return torch.cat([xy * z, z], dim=1)

    # El OBJETIVO: el render de una "escena verdad" de 60 gaussianas.
    with torch.no_grad():
        tg = 60
        target, _ = render(gaussianas_en_frustum(tg),
                           torch.randn(tg, 4, device=dev),
                           torch.rand(tg, 3, device=dev) * 0.08 + 0.06,
                           torch.full((tg,), 0.9, device=dev),
                           torch.rand(tg, 3, device=dev), T, K, H, W)
        target = target.clamp(0, 1)

    # El MODELO: n gaussianas aleatorias; TODO es optimizable. Ojo a las
    # parametrizaciones: log-escala (positividad), logit-opacidad ([0,1]),
    # cuaternion libre (se normaliza dentro del render).
    n = args.n
    means = gaussianas_en_frustum(n).requires_grad_(True)
    quats = torch.randn(n, 4, device=dev).requires_grad_(True)
    log_s = torch.log(torch.full((n, 3), 0.07, device=dev)).requires_grad_(True)
    op_l = torch.full((n,), 1.0, device=dev).requires_grad_(True)
    col_l = torch.zeros(n, 3, device=dev).requires_grad_(True)
    opt = torch.optim.Adam([
        {"params": [means], "lr": 0.008}, {"params": [quats], "lr": 0.01},
        {"params": [log_s], "lr": 0.01}, {"params": [op_l], "lr": 0.05},
        {"params": [col_l], "lr": 0.03}])

    def renderizar():
        return render(means, quats, torch.exp(log_s), torch.sigmoid(op_l),
                      torch.sigmoid(col_l), T, K, H, W)[0]

    curva, fotos = [], {}
    t0 = time.perf_counter()
    for it in range(args.iters + 1):
        opt.zero_grad()
        img = renderizar()
        loss = torch.abs(img - target).mean()      # L1: la de 3DGS (con D-SSIM)
        if it in (0, 100, args.iters) or it % 50 == 0:
            db = psnr(img.detach().clamp(0, 1), target)
            curva.append((it, db))
            if it in (0, 100, args.iters):
                fotos[it] = img.detach().clamp(0, 1).cpu().numpy()
        if it == args.iters:
            break
        loss.backward()
        opt.step()
    print(f"{args.iters} iteraciones en {time.perf_counter()-t0:.0f} s | "
          f"PSNR final: {curva[-1][1]:.1f} dB (el criterio del padre: >30; "
          "su techo con datos REALES: ~21 — leccion 41)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 5, figsize=(16, 3.4))
        axs[0].imshow(target.cpu().numpy())
        axs[0].set_title("objetivo")
        for ax, it in zip(axs[1:4], sorted(fotos)):
            ax.imshow(fotos[it])
            ax.set_title(f"iter {it}")
        for ax in axs[:4]:
            ax.axis("off")
        its, dbs = zip(*curva)
        axs[4].plot(its, dbs)
        axs[4].set_xlabel("iteracion"), axs[4].set_ylabel("PSNR [dB]")
        axs[4].axhline(30, ls=":", color="tab:red")
        axs[4].set_title("renderiza y compara"), axs[4].grid(alpha=0.3)
        salida = AQUI / "salida"
        salida.mkdir(exist_ok=True)
        fig.savefig(salida / "sobreajuste.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Grafica: {salida / 'sobreajuste.png'}")
    except ImportError:
        pass

    print("Ahora corre `python verificacion.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
