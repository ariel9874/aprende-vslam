"""Álgebra de Lie de SO(3), SE(3) y Sim(3): los mapas Exp y Log.

Son el puente entre el mundo de las MATRICES (donde se componen poses) y el
mundo de los VECTORES (donde se optimiza).

─── La matemática: por qué hacen falta ───────────────────────────────────────
SO(3), SE(3) y Sim(3) son grupos de Lie: variedades CURVAS con estructura de
grupo. No se puede "sumar ruido" ni "dar un paso de gradiente" sumando
matrices — I + δ ya no es una rotación. La solución: trabajar en el ESPACIO
TANGENTE en la identidad (el álgebra de Lie), que sí es un espacio vectorial:

    so(3)  = { [ω]_× : ω ∈ R³ }          (matrices antisimétricas)
    se(3)  = { (ρ, ω) ∈ R⁶ }             (traslación, rotación)
    sim(3) = { (ρ, ω, λ) ∈ R⁷ }          (+ log-escala)

y moverse entre ambos mundos con la exponencial de matrices y su inversa:

    Exp: R⁶/R⁷ -> grupo      (un "paso" vectorial -> un movimiento)
    Log: grupo -> R⁶/R⁷      (un movimiento -> su paso vectorial)

Convención del curso: el vector tangente es ξ = [ρ, ω] (traslación primero);
en Sim(3), ξ = [ρ, ω, λ]. (GTSAM usa (ω, ρ): ojo al comparar con su doc.)

─── La matemática: Sim(3), el grupo del SLAM monocular ───────────────────────
S = [[s·R, t], [0, 1]], que actúa como x' = s·R·x + t. Es el grupo natural
del SLAM MONOCULAR: como la escala es inobservable (nivel 07/10), la deriva
vive en SIETE grados de libertad, no seis. Un cierre de bucle sólo puede
redistribuir la deriva de escala si el grafo optimiza en Sim(3) — el
resultado de Strasdat et al. (RSS 2010), que este nivel REPRODUCE con números.

Álgebra: ξ^ = [[λ·I + [ω]_×, ρ], [0, 0]]. Como λ·I conmuta con [ω]_×, la
exponencial del bloque superior factoriza:

    exp(λ·I + [ω]_×) = e^λ · Exp_SO3(ω) = s·R

y la traslación es t = W·ρ, con W = ∫₀¹ exp(u·(λI + [ω]_×)) du, que generaliza
la V de SE(3) acoplando giro Y escala al avance. Sus cuatro ramas de Taylor
(λ→0, θ→0, ambos, ninguno) están en _sim3_W.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-8


def hat(v: np.ndarray) -> np.ndarray:
    """Matriz antisimétrica [v]_× tal que [v]_×·u = v × u."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


# ══════════════════════════════ SO(3) ════════════════════════════════════════

def so3_exp(omega: np.ndarray) -> np.ndarray:
    """Exp de SO(3): eje-ángulo (3,) -> rotación (fórmula de Rodrigues).

        Exp(ω) = I + sin θ·[k]_× + (1 − cos θ)·[k]_×²,  θ = ‖ω‖, k = ω/θ

    Cerca de θ = 0 la fórmula dividiría por cero: se usa su serie de Taylor.
    """
    theta = np.linalg.norm(omega)
    W = hat(omega)
    if theta < _EPS:
        return np.eye(3) + W + 0.5 * (W @ W)
    return (np.eye(3) + (np.sin(theta) / theta) * W
            + ((1.0 - np.cos(theta)) / theta ** 2) * (W @ W))


def so3_log(R: np.ndarray) -> np.ndarray:
    """Log de SO(3): rotación -> eje-ángulo (3,).

    Tres regímenes numéricos (la fórmula ingenua divide por sin θ, que se
    anula en θ = 0 y θ = π):
      · θ ≈ 0: Taylor, ω ≈ ½·vee(R − Rᵀ).
      · θ ≈ π: R ≈ 2kkᵀ − I  =>  kkᵀ = (R + I)/2; se extrae k de la columna
        con mayor diagonal (el signo da igual: Exp(πk) = Exp(−πk)).
      · resto: la fórmula estándar θ/(2 sin θ)·vee(R − Rᵀ).
    """
    trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(trace))
    vee = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < _EPS:
        return vee
    if np.pi - theta < 1e-6:
        S = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(S)))
        k = S[:, i] / np.sqrt(S[i, i])
        return theta * k
    return (theta / np.sin(theta)) * vee


# ══════════════════════════════ SE(3) ════════════════════════════════════════

def _left_jacobian(omega: np.ndarray) -> np.ndarray:
    """V(ω): acopla rotación y traslación en Exp_SE3.

    V = I + (1−cos θ)/θ²·[ω]_× + (θ − sin θ)/θ³·[ω]_×²

    Aparece porque al girar MIENTRAS avanzas, el camino se curva: V·ρ es el
    ARCO recorrido, no la cuerda. Con θ -> 0, V -> I (caso euclidiano).
    """
    theta = np.linalg.norm(omega)
    W = hat(omega)
    if theta < _EPS:
        return np.eye(3) + 0.5 * W + (W @ W) / 6.0
    return (np.eye(3) + ((1.0 - np.cos(theta)) / theta ** 2) * W
            + ((theta - np.sin(theta)) / theta ** 3) * (W @ W))


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Exp de SE(3): ξ = [ρ, ω] (6,) -> matriz 4x4."""
    rho, omega = np.asarray(xi[:3], float), np.asarray(xi[3:], float)
    T = np.eye(4)
    T[:3, :3] = so3_exp(omega)
    T[:3, 3] = _left_jacobian(omega) @ rho
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """Log de SE(3): matriz 4x4 -> ξ = [ρ, ω] (6,).

    ρ se recupera resolviendo V·ρ = t (más estable que invertir V a mano).
    """
    omega = so3_log(T[:3, :3])
    rho = np.linalg.solve(_left_jacobian(omega), T[:3, 3])
    return np.concatenate([rho, omega])


def se3_inv(T: np.ndarray) -> np.ndarray:
    """T⁻¹ = [[Rᵀ, −Rᵀ·t], [0, 1]] (nivel 03)."""
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


# ══════════════════════════════ Sim(3) ═══════════════════════════════════════

def _sim3_W(omega: np.ndarray, lam: float) -> np.ndarray:
    """W(ω, λ) = ∫₀¹ exp(u·(λI + [ω]ₓ)) du — el acoplador de Sim(3).

    Generaliza la V de SE(3): acopla giro Y escala al avance. Cuatro ramas
    numéricas para que degenere suavemente (λ->0 da C->1, y si además θ->0,
    A->½ y B->⅙: exactamente la V de SE(3)).
    """
    theta = np.linalg.norm(omega)
    Wx = hat(omega)
    s = np.exp(lam)
    if abs(lam) < _EPS:
        C = 1.0
        if theta < _EPS:
            A, B = 0.5, 1.0 / 6.0
        else:
            A = (1.0 - np.cos(theta)) / theta ** 2
            B = (theta - np.sin(theta)) / theta ** 3
    else:
        C = (s - 1.0) / lam
        if theta < _EPS:
            A = ((lam - 1.0) * s + 1.0) / lam ** 2
            B = (s * (0.5 * lam ** 2 - lam + 1.0) - 1.0) / lam ** 3
        else:
            a = s * np.sin(theta)
            b = s * np.cos(theta)
            c = theta ** 2 + lam ** 2
            A = (a * lam + (1.0 - b) * theta) / (theta * c)
            B = (C - ((b - 1.0) * lam + a * theta) / c) / theta ** 2
    return A * Wx + B * (Wx @ Wx) + C * np.eye(3)


def sim3_exp(xi: np.ndarray) -> np.ndarray:
    """Exp de Sim(3): ξ = [ρ, ω, λ] (7,) -> matriz 4x4 [[e^λ·R, W·ρ], [0, 1]]."""
    rho = np.asarray(xi[:3], float)
    omega = np.asarray(xi[3:6], float)
    lam = float(xi[6])
    S = np.eye(4)
    S[:3, :3] = np.exp(lam) * so3_exp(omega)
    S[:3, 3] = _sim3_W(omega, lam) @ rho
    return S


def sim3_log(S: np.ndarray) -> np.ndarray:
    """Log de Sim(3): matriz 4x4 -> ξ = [ρ, ω, λ] (7,).

    La escala sale del determinante (det(s·R) = s³ porque det R = 1), y ρ de
    resolver W·ρ = t.
    """
    sR = S[:3, :3]
    s = float(np.linalg.det(sR)) ** (1.0 / 3.0)
    lam = float(np.log(s))
    omega = so3_log(sR / s)
    rho = np.linalg.solve(_sim3_W(omega, lam), S[:3, 3])
    return np.concatenate([rho, omega, [lam]])


def sim3_inv(S: np.ndarray) -> np.ndarray:
    """Inversa cerrada: si x' = s·R·x + t, entonces x = (1/s)·Rᵀ·(x' − t)."""
    sR, t = S[:3, :3], S[:3, 3]
    s2 = float(np.linalg.det(sR)) ** (2.0 / 3.0)      # s²
    Si = np.eye(4)
    Si[:3, :3] = sR.T / s2                             # (s·R)ᵀ/s² = (1/s)·Rᵀ
    Si[:3, 3] = -(Si[:3, :3] @ t)
    return Si
