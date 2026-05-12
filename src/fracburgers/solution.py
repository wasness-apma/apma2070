"""``Solution``: a TF-callable representation of ``u(x, t)``.

A ``Solution`` wraps the spectral grid plus a routine that evaluates
``u`` on the grid at given time(s). Calling the solution at arbitrary
``(x, t)`` returns trigonometrically-interpolated values, so the
object behaves like a smooth function on ``[-L, L] × [0, ∞)``.

Both ``SpectralSolver`` and the trained ``HeatPINN`` produce
``Solution`` objects; downstream code (plots, error metrics,
comparisons) treats them uniformly.

Differentiability: trig interpolation is implemented in TF, so
gradients of ``u(x, t)`` with respect to ``x`` and ``t`` flow through
``__call__`` and can be used in PINN-style residual checks or
sensitivity analyses.
"""

from __future__ import annotations

from typing import Callable

import tensorflow as tf

from fracburgers.grid import FourierGrid
from fracburgers.interpolation import trig_interp


GridEvaluator = Callable[[tf.Tensor], tf.Tensor]
"""Signature: ``t (scalar or (T, 1)) -> u_grid ((N,) or (T, N))``."""

ArbitraryEvaluator = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
"""Signature: ``(x_flat (B,), t_flat (B,)) -> u_flat (B,)``.

Used by solutions that can *calculate* u at arbitrary (x, t) pairs
directly (e.g. closed-form references), bypassing trig interpolation.
"""


class Solution:
    """Smooth, differentiable representation of ``u(x, t)`` on the grid.

    Parameters
    ----------
    grid : FourierGrid
    on_grid : GridEvaluator
        Routine returning ``u`` sampled on ``grid.x`` for one or many
        times. Called by ``__call__`` and ``sample`` below.
    on_arbitrary : ArbitraryEvaluator, optional
        If provided, ``__call__`` routes through this instead of
        ``sample`` + trig interpolation. Use for closed-form references
        that can evaluate directly at any ``(x, t)`` pair.
    """

    def __init__(self, grid: FourierGrid, on_grid: GridEvaluator, *,
                 on_arbitrary: ArbitraryEvaluator | None = None):
        self.grid = grid
        self._on_grid = on_grid
        self._on_arbitrary = on_arbitrary

    def sample(self, t: tf.Tensor) -> tf.Tensor:
        """Evaluate ``u(x_grid, t)`` at the requested times.

        Returns shape ``(N,)`` for scalar ``t`` or ``(T, N)`` for
        ``t`` of shape ``(T, 1)``.
        """
        return self._on_grid(t)

    def __call__(self, x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        """Evaluate ``u`` at arbitrary ``(x, t)`` via trig interpolation.

        ``x`` and ``t`` should broadcast against each other; the result
        has the broadcast shape.
        """
        x = tf.cast(tf.convert_to_tensor(x), dtype=tf.float64)
        t = tf.cast(tf.convert_to_tensor(t), dtype=tf.float64)

        out_shape = tf.broadcast_dynamic_shape(tf.shape(x), tf.shape(t))
        x_b = tf.broadcast_to(x, out_shape)
        t_b = tf.broadcast_to(t, out_shape)

        x_flat = tf.reshape(x_b, [-1])       # (B,)
        t_flat = tf.reshape(t_b, [-1, 1])     # (B, 1)

        if self._on_arbitrary is not None:
            # Calculate directly at each (x_i, t_i) pair — no interpolation.
            u_flat = self._on_arbitrary(x_flat, tf.reshape(t_flat, [-1]))
        else:
            # Sample on the solver grid, then trig-interpolate to x_flat.
            u_grid = self.sample(t_flat)          # (B, N)
            # trig_interp does an outer product: (B, N) × (B,) → (B, B).
            # Each row i holds interpolants of time i at every x; we only want the
            # diagonal entry where x = x_flat[i] matches t = t_flat[i].
            u_all = trig_interp(u_grid, x_flat, self.grid)  # (B, B)
            u_flat = tf.linalg.diag_part(u_all)             # (B,)
        return tf.reshape(u_flat, out_shape)
