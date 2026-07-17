#!/usr/bin/env python3
"""Examen del nivel 19: el rasterizador diferenciable, verificado.

Cuatro actos, SIN dataset (todo sintético — el examen del padre, ampliado):

  1. PROYECCIÓN: una gaussiana en el eje óptico cae en el punto principal.
  2. GRADIENTE: autograd == diferencias finitas (la prueba de que la cadena
     proyección → covarianza EWA → blending está bien derivada; si esto
     falla, el "renderiza y compara" no aprende nada).
  3. SOBREAJUSTE: desde gaussianas aleatorias, el descenso de gradiente
     re-sintetiza una imagen a PSNR > 30 dB (el criterio v0.7 del padre).
  4. EL MEDIO PÍXEL (lección 40): dos renders de los MISMOS parámetros con
     convención de centro distinta discrepan a ~25-30 dB — el bug de
     equivalencia que al padre le costó 25 → 60 dB encontrar.

Con GPU dura ~1 min; en CPU, varios minutos (el acto 3 es el caro).

Uso:
    python verificacion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

fallos = []


def check(nombre: str, ok: bool, detalle: str = "") -> None:
    estado = "OK " if ok else "FALLO"
    print(f"  [{estado}] {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def main() -> int:
    try:
        import torch
    except ImportError:
        raise SystemExit("Este nivel necesita torch: pip install torch "
                         "(con CUDA si tienes GPU NVIDIA).")
    from render_gaussianas import psnr, render

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Verificando el rasterizador (sin dataset) | acto 3 en: {dev}\n")

    # ── Acto 1: proyeccion ───────────────────────────────────────────────────
    print("[1/4] La proyeccion...")
    dt = torch.float64
    K = torch.tensor([[60.0, 0, 24], [0, 60, 24], [0, 0, 1]], dtype=dt)
    T = torch.eye(4, dtype=dt)
    img, alpha = render(torch.tensor([[0.0, 0.0, 3.0]], dtype=dt),
                        torch.tensor([[1.0, 0, 0, 0]], dtype=dt),
                        torch.full((1, 3), 0.08, dtype=dt),
                        torch.ones(1, dtype=dt), torch.ones(1, 3, dtype=dt),
                        T, K, 48, 48)
    pico = int(torch.argmax(alpha))
    py, px = pico // 48, pico % 48
    check("una gaussiana en el eje optico cae en el punto principal",
          abs(px - 24) <= 1 and abs(py - 24) <= 1 and float(alpha.max()) > 0.5,
          f"pico en ({px},{py}), esperado (24,24); alpha max "
          f"{float(alpha.max()):.2f}")

    # ── Acto 2: gradiente vs diferencias finitas ─────────────────────────────
    print("\n[2/4] El gradiente (autograd == diferencias finitas)...")
    means = torch.tensor([[0.2, -0.1, 3.0], [-0.3, 0.15, 2.5]], dtype=dt,
                         requires_grad=True)
    quats = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]], dtype=dt)
    scales = torch.full((2, 3), 0.1, dtype=dt)
    opac = torch.full((2,), 0.8, dtype=dt)
    col = torch.tensor([[0.9, 0.2, 0.1], [0.1, 0.5, 0.9]], dtype=dt)

    def perdida(m):
        return (render(m, quats, scales, opac, col, T, K, 32, 32)[0] ** 2).sum()

    perdida(means).backward()
    grad = means.grad.clone()
    eps, peor = 1e-6, 0.0
    for i in range(2):
        for j in range(3):
            d = torch.zeros_like(means)
            d[i, j] = eps
            num = (perdida((means + d).detach())
                   - perdida((means - d).detach())) / (2 * eps)
            peor = max(peor, abs(float(num) - float(grad[i, j])))
    check("el jacobiano de autograd coincide con diferencias finitas",
          peor < 1e-3, f"dif maxima {peor:.1e} (6 derivadas parciales)")

    # ── Acto 3: sobreajuste > 30 dB ──────────────────────────────────────────
    print("\n[3/4] El sobreajuste (renderiza y compara, 1500 iteraciones)...")
    dtf = torch.float32
    torch.manual_seed(0)
    Kf = K.to(dev, dtf)
    Tf = T.to(dev, dtf)

    def frustum(n):
        z = torch.rand(n, 1, device=dev) * 2.0 + 2.0
        xy = (torch.rand(n, 2, device=dev) - 0.5) * 1.6
        return torch.cat([xy * z, z], dim=1)

    with torch.no_grad():
        objetivo, _ = render(frustum(60), torch.randn(60, 4, device=dev),
                             torch.rand(60, 3, device=dev) * 0.08 + 0.06,
                             torch.full((60,), 0.9, device=dev),
                             torch.rand(60, 3, device=dev), Tf, Kf, 48, 48)
        objetivo = objetivo.clamp(0, 1)
    n = 300
    means = frustum(n).requires_grad_(True)
    quats = torch.randn(n, 4, device=dev).requires_grad_(True)
    log_s = torch.log(torch.full((n, 3), 0.07, device=dev)).requires_grad_(True)
    op_l = torch.full((n,), 1.0, device=dev).requires_grad_(True)
    col_l = torch.zeros(n, 3, device=dev).requires_grad_(True)
    opt = torch.optim.Adam([
        {"params": [means], "lr": 0.008}, {"params": [quats], "lr": 0.01},
        {"params": [log_s], "lr": 0.01}, {"params": [op_l], "lr": 0.05},
        {"params": [col_l], "lr": 0.03}])
    for _ in range(1500):
        opt.zero_grad()
        img, _ = render(means, quats, torch.exp(log_s), torch.sigmoid(op_l),
                        torch.sigmoid(col_l), Tf, Kf, 48, 48)
        torch.abs(img - objetivo).mean().backward()
        opt.step()
    with torch.no_grad():
        img, _ = render(means, quats, torch.exp(log_s), torch.sigmoid(op_l),
                        torch.sigmoid(col_l), Tf, Kf, 48, 48)
        db = psnr(img.clamp(0, 1), objetivo)
    check("el descenso de gradiente re-sintetiza la vista (PSNR > 30 dB)",
          db > 30.0, f"{db:.1f} dB (el techo del padre con datos REALES: "
          "~21 dB — leccion 41: la capacidad no era el cuello)")

    # ── Acto 4: el medio pixel (leccion 40) ──────────────────────────────────
    print("\n[4/4] El medio pixel: dos convenciones, mismos parametros...")
    with torch.no_grad():
        con, _ = render(means, quats, torch.exp(log_s), torch.sigmoid(op_l),
                        torch.sigmoid(col_l), Tf, Kf, 48, 48,
                        centro_pixel=True)
        sin, _ = render(means, quats, torch.exp(log_s), torch.sigmoid(op_l),
                        torch.sigmoid(col_l), Tf, Kf, 48, 48,
                        centro_pixel=False)
        db_conv = psnr(sin.clamp(0, 1), con.clamp(0, 1))
    check("medio pixel de desalineacion arruina la 'equivalencia' (< 40 dB)",
          db_conv < 40.0,
          f"{db_conv:.1f} dB entre convenciones (el padre: sus gemelas "
          "discrepaban a ~25 dB hasta alinear ESTO; 25 -> 60 dB con una linea)")

    print()
    if fallos:
        print(f"NIVEL 19: {len(fallos)} fallo(s): {', '.join(fallos)}")
        return 1
    print("NIVEL 19: VERIFICADO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
