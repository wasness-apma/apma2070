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


def u_to_theta_0(u_0: tf.Tensor, alpha: float, nu: float,
                 grid: FourierGrid) -> tf.Tensor:
    """Forward transform on initial data: ``u_0 ↦ θ_0``.

    ``θ_0 = exp(-I^α u_0 / (2ν))``. The fractional integral is
    evaluated spectrally; see ``operators.fractional_integral``.
    Output is positive by construction.
    """

    # The fractional integral is only defined up to the k=0 mode,
    # so we remove the spatial mean as a fixed gauge.
    u_0_centered = u_0 - tf.reduce_mean(u_0, axis=-1, keepdims=True)
    ialpha_u_0 = operators.fractional_integral(u_0_centered, alpha, grid)
    theta_0 = tf.exp(-ialpha_u_0 / (2 * nu))
    return theta_0


def theta_to_u(theta: tf.Tensor, alpha: float, nu: float,
               grid: FourierGrid) -> tf.Tensor:
    """Inverse transform: ``θ ↦ u = -2ν D^α ln θ``.

    Works on a single snapshot ``(N,)`` or a stack ``(T, N)``. Assumes
    ``θ > 0`` everywhere (true under the heat flow when ``θ_0 > 0``).
    """

    dlog_theta = operators.fractional_derivative(tf.math.log(theta), alpha, grid)
    return -2 * nu * dlog_theta
