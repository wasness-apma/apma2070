"""Fractional Cole–Hopf transform, in TensorFlow.

The substitution ``u = -2ν D^α ln θ`` reduces the fractional Burgers
equation (with the operator structure used in this project) to the
classical heat equation ``θ_t = ν θ_{xx}``. Initial data are related
by

    θ_0(x) = exp(-I^α u_0(x) / (2ν))
    u_0(x) = -2ν D^α ln θ_0(x)

All inputs and outputs are ``tf.Tensor``. Operations are pure (no
state) and differentiable end-to-end via TF autograd.
"""

from __future__ import annotations

import tensorflow as tf

from fracburgers.grid import FourierGrid
import fracburgers.operators as operators


def u_to_log_theta_0(u_0: tf.Tensor, alpha: float, nu: float,
                     grid: FourierGrid) -> tf.Tensor:
    """Forward transform in log form: ``u_0 ↦ log θ_0 = -I^α u_0 / (2ν)``.

    Algebraic — no ``exp``.  Prefer this over ``u_to_theta_0`` whenever
    the consumer ultimately wants ``log θ`` (PINN IC loss, Cole–Hopf
    inverse), since it avoids both overflow risk and the wasted
    ``exp`` → ``log`` roundtrip.
    """

    # The fractional integral is only defined up to the k=0 mode,
    # so we remove the spatial mean as a fixed gauge.
    u_0_centered = u_0 - tf.reduce_mean(u_0, axis=-1, keepdims=True)
    ialpha_u_0 = operators.fractional_integral(u_0_centered, alpha, grid)
    return -ialpha_u_0 / (2 * nu)


def u_to_theta_0(u_0: tf.Tensor, alpha: float, nu: float,
                 grid: FourierGrid) -> tf.Tensor:
    """Forward transform: ``u_0 ↦ θ_0 = exp(-I^α u_0 / (2ν))``.

    Thin wrapper over ``u_to_log_theta_0`` followed by ``exp``.  Output
    is positive by construction.
    """
    return tf.exp(u_to_log_theta_0(u_0, alpha, nu, grid))


def log_theta_to_u(log_theta: tf.Tensor, alpha: float, nu: float,
                   grid: FourierGrid) -> tf.Tensor:
    """Inverse transform from log θ: ``u = -2ν D^α log θ``.

    Use this when log θ is already available (e.g. from the PINN's
    linear pre-activation) — avoids the ``log()`` call in ``theta_to_u``
    and the float32 underflow → ``-∞`` → NaN trap.
    """
    dlog_theta = operators.fractional_derivative(log_theta, alpha, grid)
    return -2 * nu * dlog_theta


def theta_to_u(theta: tf.Tensor, alpha: float, nu: float,
               grid: FourierGrid) -> tf.Tensor:
    """Inverse transform: ``θ ↦ u = -2ν D^α log θ``.

    Works on a single snapshot ``(N,)`` or a stack ``(T, N)``. Assumes
    ``θ > 0`` everywhere (true under the heat flow when ``θ_0 > 0``).
    """
    return log_theta_to_u(tf.math.log(theta), alpha, nu, grid)
