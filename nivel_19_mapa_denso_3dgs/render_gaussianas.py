"""Rasterizador de 3D Gaussian Splatting DIFERENCIABLE, desde cero.

El mapa disperso del curso (puntos 3D) sirve para LOCALIZAR; para re-SINTETIZAR
la escena hace falta un mapa DENSO. 3DGS: el mapa es un conjunto de gaussianas
3D con color y opacidad, y el render las proyecta y mezcla en una imagen. Todo
DIFERENCIABLE: el mapa se ajusta por descenso de gradiente para re-sintetizar
las vistas ("renderiza y compara" — la misma filosofia del BA, con fotometria
en vez de reproyeccion de puntos).

Esta es la REFERENCIA legible (como el BA NumPy del nivel 11): densa y
vectorizada en PyTorch puro, sin kernels CUDA. Coste O(N·H·W) — correcta,
diferenciable de punta a punta, y suficiente para aprender. Las gemelas de
rendimiento (tiles, gsplat/CUDA) son la historia del nivel 18 otra vez: mismo
contrato, test de equivalencia (el repo padre las tiene; ver README).

─── La matemática: de gaussiana 3D a píxel ───────────────────────────────────
Cada gaussiana i tiene media μ_i (mundo), covarianza Σ_i = R·S·Sᵀ·Rᵀ (rotación
por escalas: definida positiva POR CONSTRUCCIÓN — por eso se optimizan R y S,
no Σ), opacidad α_i y color c_i.

1) A la cámara: μ_c = R_cw·μ + t_cw; pinhole: (u, v) = (fx·x/z+cx, fy·y/z+cy).

2) Covarianza proyectada (EWA splatting, Zwicker 2001): se linealiza la
   proyección con su jacobiano J (2×3) — ¡el MISMO d_pi del BA del nivel 11! —
   y se propaga la covarianza por la cadena mundo→cámara→imagen:

       Σ' = J·W·Σ·Wᵀ·Jᵀ  (2×2),   W = R_cw

   Más un desenfoque isótropo pequeño (DILATION) para que ninguna gaussiana
   colapse por debajo de un píxel (anti-aliasing + estabilidad).

3) Peso por píxel: g_i(p) = exp(−½·(p−μ')ᵀ·Σ'⁻¹·(p−μ')); aporte a_i = α_i·g_i.

4) α-blending por profundidad (front-to-back), ordenadas por z creciente:

       C(p) = Σ_i c_i·a_i(p)·T_i(p),   T_i(p) = Π_{j<i} (1 − a_j(p))

   T_i es la TRANSMITANCIA (cuánta luz llega sin ser ocluida): un producto
   acumulado EXCLUSIVO — tensorial y diferenciable.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Tuple

import torch

# Desenfoque isótropo que se suma a la diagonal de Σ' (en px²): impide que una
# gaussiana sub-píxel tenga determinante ~0 (inversa inestable) y aliasing.
DILATION = 0.3


def quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
    """Cuaterniones (N, 4) [w, x, y, z] → matrices de rotación (N, 3, 3).
    Se normalizan aquí: el optimizador puede moverlos libremente sin restringir
    la norma (la parametrización estándar de 3DGS)."""
    q = quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)


def _covariance_3d(quats: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Σ = R·S·Sᵀ·Rᵀ (N, 3, 3) con S = diag(scales). Definida positiva por
    construcción — por eso se optimizan rotación y escalas, no Σ directa."""
    R = quat_to_rotmat(quats)
    M = R * scales[:, None, :]                       # R·S  (columnas escaladas)
    return M @ M.transpose(1, 2)


def render(
    means: torch.Tensor,       # (N, 3) posiciones en el MUNDO
    quats: torch.Tensor,       # (N, 4) [w, x, y, z] (se normalizan)
    scales: torch.Tensor,      # (N, 3) > 0 (desviaciones por eje)
    opacities: torch.Tensor,   # (N,) en [0, 1]
    colors: torch.Tensor,      # (N, 3) en [0, 1]
    T_w_c: torch.Tensor,       # (4, 4) cámara→mundo (ejes OpenCV)
    K: torch.Tensor,           # (3, 3) intrínsecos
    height: int,
    width: int,
    background: float = 0.0,
    centro_pixel: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Renderiza las gaussianas a la vista (T_w_c, K). Devuelve (imagen (H, W, 3),
    alpha acumulada (H, W)). Diferenciable respecto a TODOS los parámetros de las
    gaussianas y a la pose. Denso: coste O(N·H·W) — para la referencia y tests.
    """
    device, dtype = means.device, means.dtype
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 1) Mundo → cámara. T_c_w = T_w_c⁻¹ = [Rᵀ, −Rᵀ·t].
    R_wc = T_w_c[:3, :3]
    t_wc = T_w_c[:3, 3]
    R_cw = R_wc.transpose(0, 1)
    mu_c = (R_cw @ (means - t_wc).T).T               # (N, 3)
    x, y, z = mu_c[:, 0], mu_c[:, 1], mu_c[:, 2]
    z_safe = torch.clamp(z, min=1e-4)

    # 2) Media 2D (proyección pinhole).
    u = fx * x / z_safe + cx
    v = fy * y / z_safe + cy
    mean2d = torch.stack([u, v], dim=-1)             # (N, 2)

    # 2b) Covarianza proyectada Σ' = J·W·Σ·Wᵀ·Jᵀ (EWA).
    Sigma = _covariance_3d(quats, scales)            # (N, 3, 3)
    zero = torch.zeros_like(z_safe)
    J = torch.stack([
        torch.stack([fx / z_safe, zero, -fx * x / z_safe ** 2], dim=-1),
        torch.stack([zero, fy / z_safe, -fy * y / z_safe ** 2], dim=-1),
    ], dim=1)                                        # (N, 2, 3)
    W = R_cw.unsqueeze(0)                            # (1, 3, 3)
    JW = J @ W                                       # (N, 2, 3)
    cov2d = JW @ Sigma @ JW.transpose(1, 2)          # (N, 2, 2)
    cov2d = cov2d + DILATION * torch.eye(2, device=device, dtype=dtype)

    # Inversa 2×2 explícita (barata y diferenciable).
    a, b = cov2d[:, 0, 0], cov2d[:, 0, 1]
    c, d = cov2d[:, 1, 0], cov2d[:, 1, 1]
    det = (a * d - b * c).clamp(min=1e-8)
    inv = torch.stack([
        torch.stack([d, -b], dim=-1),
        torch.stack([-c, a], dim=-1),
    ], dim=1) / det[:, None, None]                   # (N, 2, 2)

    # 3) Peso gaussiano por píxel. Rejilla de píxeles (H, W, 2).
    #    +0.5: se evalúa en el CENTRO del píxel. El píxel i cubre [i, i+1) y
    #    su centro cae en i+0.5 — la convención estándar (3DGS, gsplat,
    #    OpenGL). Este medio píxel le costó caro al repo padre (su lección
    #    40): su referencia y gsplat discrepaban a ~25 dB hasta alinear la
    #    convención (25 → 60 dB con UNA línea). El examen de este nivel
    #    reproduce esa discrepancia a propósito: dos gemelas que no comparten
    #    convención no pasan el test de equivalencia del nivel 18.
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype), indexing="ij")
    #    (centro_pixel=False existe SOLO para el experimento del examen:
    #    reproducir el bug de convención de la lección 40.)
    grid = torch.stack([xs, ys], dim=-1) + (0.5 if centro_pixel else 0.0)
    delta = grid[None] - mean2d[:, None, None, :]    # (N, H, W, 2)
    power = -0.5 * torch.einsum("nhwi,nij,nhwj->nhw", delta, inv, delta)
    g = torch.exp(power.clamp(max=0.0))              # (N, H, W)

    visible = (z > 1e-4).to(dtype)[:, None, None]    # detrás de cámara: no aporta
    a_i = (opacities[:, None, None] * g * visible).clamp(0.0, 0.999)  # (N, H, W)

    # 4) α-blending front-to-back: ordenar por profundidad y acumular.
    order = torch.argsort(z)
    a_i = a_i[order]
    col = colors[order]                              # (N, 3)
    one_minus = 1.0 - a_i
    # Transmitancia EXCLUSIVA: T_i = Π_{j<i}(1−a_j). cumprod inclusivo, desplazado.
    T = torch.cumprod(one_minus, dim=0)
    T_excl = torch.ones_like(a_i)
    T_excl[1:] = T[:-1]
    weight = a_i * T_excl                            # (N, H, W)

    image = torch.einsum("nhw,nc->hwc", weight, col)
    alpha = weight.sum(dim=0)                         # (H, W)
    image = image + (1.0 - alpha)[..., None] * background
    return image, alpha


def psnr(rendered: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR (dB) entre dos imágenes en [0, 1] — la métrica del criterio v0.7."""
    mse = torch.mean((rendered - target) ** 2).clamp(min=1e-12)
    return float(10.0 * torch.log10(1.0 / mse))
