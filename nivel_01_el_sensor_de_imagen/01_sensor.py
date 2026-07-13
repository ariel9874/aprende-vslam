#!/usr/bin/env python3
"""
Nivel 01 — El sensor de imagen
==============================

Tres experimentos con un sensor SIMULADO (sin dataset, sin webcam):

    1. ruido vs señal: el shot noise crece como la raiz (pendiente 1/2)
    2. mosaico de Bayer: demosaicar A MANO y medir el error
    3. rolling shutter: una barra vertical en movimiento se inclina,
       y la inclinacion se PREDICE y se MIDE

Uso:
    python 01_sensor.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent

# Parametros del sensor simulado (tipicos de un CMOS modesto).
FULL_WELL = 1000.0    # capacidad del pozo, en electrones
READ_NOISE = 5.0      # ruido de lectura, en electrones (gaussiano)
RNG = np.random.default_rng(7)


def exponer(radiancia: np.ndarray, exposicion: float) -> np.ndarray:
    """Simula la captura: radiancia [0,1] -> electrones contados.

    ─── La matemática: el shot noise es Poisson ───
    El fotosito CUENTA sucesos discretos (fotoelectrones). Contar sucesos
    independientes a tasa constante es un proceso de Poisson: si el valor
    esperado es N electrones, la varianza TAMBIÉN es N, así que

        σ = √N        (la señal crece como N, el ruido como √N)

    La relación señal/ruido mejora como N/√N = √N: por eso más luz (o más
    exposición) = foto más limpia, y por eso las sombras levantadas en
    postproceso "granulan". A eso se suma el ruido de LECTURA (electrónica
    del ADC, gaussiano, independiente de la señal): domina en lo MUY oscuro.
    Y el pozo se llena: por encima de FULL_WELL el conteo se TRUNCA
    (saturación: varianza cero, información cero).
    """
    esperado = radiancia * exposicion * FULL_WELL
    electrones = RNG.poisson(esperado).astype(np.float64)
    electrones += RNG.normal(0.0, READ_NOISE, electrones.shape)
    return np.clip(electrones, 0.0, FULL_WELL)


def a_uint8(electrones: np.ndarray) -> np.ndarray:
    """El ADC: electrones -> niveles digitales 0..255 (cuantizacion)."""
    return np.clip(np.round(electrones / FULL_WELL * 255.0), 0, 255).astype(np.uint8)


# ───────────────────── 1 · ruido vs señal (la ley de la raíz) ────────────────

def experimento_ruido(salida: Path) -> float:
    """Mide sigma vs media en un parche uniforme y ajusta la pendiente log-log.

    Devuelve la pendiente (esperada: 0.5 en el regimen dominado por shot).
    """
    parche = np.full((200, 200), 0.5)      # radiancia uniforme al 50%
    exposiciones = np.geomspace(0.02, 1.6, 12)
    medias, sigmas = [], []
    for t in exposiciones:
        e = exponer(parche, t)
        medias.append(e.mean())
        sigmas.append(e.std())
    medias, sigmas = np.array(medias), np.array(sigmas)

    # ─── La matemática: la curva de transferencia de fotones ───
    # Las dos fuentes de ruido son independientes, así que sus VARIANZAS
    # suman:  σ_total² = N + σ_read²  (shot Poisson + lectura gaussiana).
    # Ajustar σ_total directamente contra N da pendiente < 0.5 porque el
    # piso de lectura "levanta" el extremo oscuro. El método estándar para
    # caracterizar sensores reales (photon transfer curve) hace lo que
    # hacemos aquí: restar la varianza de lectura y ajustar la componente
    # shot, que debe salir con pendiente 1/2 exacta.
    ok = (medias > 100) & (medias < 0.7 * FULL_WELL)   # lejos de saturar
    sigma_shot = np.sqrt(np.maximum(sigmas ** 2 - READ_NOISE ** 2, 1e-9))
    cruda = float(np.polyfit(np.log(medias[ok]), np.log(sigmas[ok]), 1)[0])
    pendiente = float(np.polyfit(np.log(medias[ok]), np.log(sigma_shot[ok]), 1)[0])

    print("1. Ruido vs senal (parche uniforme, 12 exposiciones):")
    print(f"   {'media e-':>10s} {'sigma e-':>9s} {'sqrt(media)':>12s}")
    for m, s in zip(medias[::3], sigmas[::3]):
        print(f"   {m:10.1f} {s:9.2f} {np.sqrt(m):12.2f}")
    print(f"   pendiente log-log cruda: {cruda:.3f}  (el piso de lectura la achata)")
    print(f"   pendiente con la varianza de lectura restada: {pendiente:.3f}"
          "  (teoria Poisson: 0.500)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.loglog(medias, sigmas, "o-", label="medido")
        ax.loglog(medias, np.sqrt(medias), "--", label="sqrt(N) (Poisson puro)")
        ax.axhline(READ_NOISE, color="gray", ls=":", label="piso de lectura")
        ax.set_xlabel("senal media [e-]"), ax.set_ylabel("ruido sigma [e-]")
        ax.set_title("El shot noise crece como la raiz de la senal")
        ax.legend(), ax.grid(True, which="both", alpha=0.3)
        fig.savefig(salida / "ruido_vs_senal.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        print("[aviso] matplotlib no instalado: se omite ruido_vs_senal.png")
    return pendiente


# ───────────────────────── 2 · Bayer y demosaicado ───────────────────────────

def escena_color(h: int = 240, w: int = 320) -> np.ndarray:
    """Escena RGB flotante [0,1]: gradientes suaves + discos de color."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.stack([xx / w, yy / h, 0.5 + 0.5 * np.sin(xx / 23)], axis=2)
    for (cx, cy, r, color) in [(80, 60, 30, (0.9, 0.2, 0.1)),
                               (220, 90, 40, (0.1, 0.8, 0.2)),
                               (150, 180, 35, (0.2, 0.3, 0.9))]:
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        for c in range(3):
            img[..., c][mask] = color[c]
    return img


def mosaico_bayer(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Aplica el patron RGGB: en cada pixel sobrevive UN canal.

    Devuelve (crudo, mascaras) — crudo es la imagen de UN canal (lo que el
    sensor entrega de verdad) y mascaras (H,W,3) dice que canal vive donde.
    """
    h, w, _ = rgb.shape
    masks = np.zeros((h, w, 3), bool)
    masks[0::2, 0::2, 0] = True                       # R en filas/cols pares
    masks[0::2, 1::2, 1] = True                       # G
    masks[1::2, 0::2, 1] = True                       # G (el doble de verdes)
    masks[1::2, 1::2, 2] = True                       # B
    crudo = np.where(masks, rgb, 0.0).sum(axis=2)
    return crudo, masks


def demosaicar(crudo: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Demosaicado bilineal A MANO: rellenar cada canal desde sus vecinos.

    ─── La matemática: interpolar un canal disperso ───
    Cada canal es una rejilla con huecos (el canal R solo existe en 1 de
    cada 4 pixeles). La interpolacion bilineal de los huecos equivale a una
    CONVOLUCION del canal disperso con un kernel piramidal, normalizada por
    la misma convolucion de la mascara (cuantos vecinos reales aportaron):

        canal_lleno = (disperso * K) / (mascara * K),   K = [1 2 1]
                                                            [2 4 2] / 4
                                                            [1 2 1]

    Es el demosaico mas simple que existe; los reales (y el de tu telefono)
    ademas miran los BORDES para no interpolar a traves de ellos — de ahi
    los artefactos de color en rejillas finas (moire, "cremalleras").
    """
    K = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float64) / 4.0
    out = np.zeros(masks.shape, np.float64)
    for c in range(3):
        disperso = np.where(masks[..., c], crudo, 0.0)
        num = cv2.filter2D(disperso, -1, K, borderType=cv2.BORDER_REFLECT)
        den = cv2.filter2D(masks[..., c].astype(np.float64), -1, K,
                           borderType=cv2.BORDER_REFLECT)
        out[..., c] = num / np.maximum(den, 1e-9)
    return np.clip(out, 0.0, 1.0)


def experimento_bayer(salida: Path) -> float:
    """Mosaico + demosaico propio; devuelve el RMS (0..1) contra el original."""
    rgb = escena_color()
    crudo, masks = mosaico_bayer(rgb)
    rec = demosaicar(crudo, masks)
    borde = 4                               # el borde interpola con reflejo
    rms = float(np.sqrt(((rec - rgb)[borde:-borde, borde:-borde] ** 2).mean()))
    print(f"\n2. Bayer: 2/3 del color es interpolado. RMS del demosaico "
          f"propio: {rms:.4f} (escala 0-1)")

    panel = np.hstack([rgb, np.repeat(crudo[..., None], 3, axis=2), rec])
    cv2.imwrite(str(salida / "bayer_demosaico.png"),
                cv2.cvtColor((panel * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    return rms


# ─────────────────────────── 3 · rolling shutter ─────────────────────────────

def experimento_rolling(salida: Path) -> tuple[float, float]:
    """Barra vertical que se mueve; el sensor lee fila a fila -> se inclina.

    ─── La matemática: la inclinación del rolling shutter ───
    Si la fila y se lee en t(y) = y/H · t_frame y la barra se mueve a
    v píxeles por frame, la fila y la ve desplazada v·y/H píxeles:

        x(y) = x0 + (v / H) · y      →   pendiente dx/dy = v/H

    Una vertical se vuelve una recta inclinada; con v = 80 px/frame y
    H = 240, la pendiente esperada es 1/3 de píxel por fila.
    Devuelve (pendiente_medida, pendiente_esperada).
    """
    h, w = 240, 320
    v = 80.0                                 # px por tiempo-de-frame
    x0, grosor = 120.0, 10
    img = np.zeros((h, w))
    for y in range(h):                       # cada fila, en su instante
        x_en_t = x0 + v * (y / h)
        img[y, int(x_en_t): int(x_en_t) + grosor] = 1.0

    # medir: centroide de la barra por fila -> ajuste lineal x(y)
    xs = np.array([np.average(np.arange(w), weights=img[y] + 1e-12)
                   for y in range(h)])
    pendiente = float(np.polyfit(np.arange(h), xs, 1)[0])
    esperada = v / h
    print(f"\n3. Rolling shutter: barra a {v:.0f} px/frame, lectura fila a fila.")
    print(f"   pendiente medida {pendiente:.4f} px/fila | esperada {esperada:.4f}")

    vis = (np.repeat(img[..., None], 3, axis=2) * 255).astype(np.uint8)
    cv2.line(vis, (int(x0 + grosor / 2), 0), (int(x0 + grosor / 2), h - 1),
             (0, 200, 255), 1)               # donde ESTARIA con global shutter
    cv2.imwrite(str(salida / "rolling_shutter.png"), vis)
    return pendiente, esperada


def main() -> int:
    salida = AQUI / "salida"
    salida.mkdir(exist_ok=True)

    experimento_ruido(salida)
    experimento_bayer(salida)
    experimento_rolling(salida)

    print(f"\nGuardado en {salida}: ruido_vs_senal.png, bayer_demosaico.png, "
          "rolling_shutter.png")
    print("Ahora corre `python verificacion.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
