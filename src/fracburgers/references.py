"""Closed-form reference solutions for validating the spectral pipeline.

The pure-cosine-mode example: pick ``θ_0(x) = a + b cos(kx)`` with
``a > |b| > 0``. The heat-evolved θ admits a clean Fourier expansion
of ``ln θ``, and the resulting ``u = -2ν D^α ln θ`` has the explicit
series

    u(x, t) = 4ν k^α  Σ_{n≥1}  r(t)^n / n^{1-α}  cos(nkx + απ/2),

where

    β(t) = b · exp(-ν k² t),
    r(t) = -β(t) / (a + √(a² - β(t)²)).

The series converges geometrically with ratio ``|r(t)| < 1``, so a
few dozen terms suffice for spectral precision.

Use this to validate the spectral solver: build a ``CosineModeReference``,
hand its ``initial_condition()`` to ``SpectralSolver.solve``, and
compare against ``reference_solution(grid)``. The two should agree to
``~1e-12`` if everything (sign conventions, FFT normalization,
``(ik)^α`` branch) is correct.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from fracburgers.grid import FourierGrid
from fracburgers.initial_conditions import InitialCondition
from fracburgers.solution import Solution


@dataclass
class CosineModeReference:
    """Closed-form reference for the IC ``θ_0 = a + b cos(kx)``.

    Parameters
    ----------
    a, b : float
        Cosine offset and amplitude. Must satisfy ``a > |b| > 0`` for
        positivity of θ.
    k : float
        Wavenumber. Must be positive; pick the grid's ``L`` so that
        ``2π/k`` divides ``2L`` evenly (otherwise periodicity breaks).
    nu : float
        Diffusivity. Must match the solver's ``nu``.
    alpha : float
        Fractional order in ``(0, 1)``. Must match the solver's ``alpha``.
    n_terms : int
        Number of terms to retain in the series. Default 80 gives
        ``< 1e-15`` truncation error for typical ``|r| ≤ 0.5``.
    """

    a: float
    b: float
    k: float
    nu: float
    alpha: float
    n_terms: int = 80

    def __post_init__(self) -> None:
        if self.a <= abs(self.b):
            raise ValueError(f"need a > |b|; got a={self.a}, b={self.b}")
        if self.k <= 0:
            raise ValueError(f"need k > 0; got k={self.k}")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"need 0 < α < 1; got α={self.alpha}")
        if self.nu <= 0:
            raise ValueError(f"need ν > 0; got ν={self.nu}")

    # ------------------------------------------------------------------
    # Internals: the closed-form series u(x, t).
    # ------------------------------------------------------------------

    def _r(self, t: tf.Tensor) -> tf.Tensor:
        """``r(t) = -β/(a + √(a² - β²))``  for  ``β = b·e^{-νk²t}``."""
        beta = self.b * tf.exp(-self.nu * self.k**2 * t)
        return -beta / (self.a + tf.sqrt(self.a**2 - beta**2))

    def _u_on_grid(self, x_grid: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        """Evaluate the truncated series at the grid points.

        ``x_grid``: shape ``(N,)``.
        ``t``: scalar or shape ``(T, 1)`` or ``(T,)``.
        Returns ``(N,)`` for scalar ``t``, otherwise ``(T, N)``.
        """
        t = tf.cast(tf.convert_to_tensor(t), tf.float64)
        original_rank = t.shape.rank
        t_flat = tf.reshape(t, [-1])  # (B,)

        n = tf.range(1, self.n_terms + 1, dtype=tf.float64)  # (n_terms,)

        beta = self.b * tf.exp(-self.nu * self.k**2 * t_flat)  # (B,)
        r = -beta / (self.a + tf.sqrt(self.a**2 - beta**2))    # (B,)

        # r^n / n^{1-α}, shape (B, n_terms)
        coeffs = (r[:, None] ** n[None, :]) / n**(1.0 - self.alpha)

        # cos(n k x + α π / 2), shape (n_terms, N)
        phase = (n * self.k)[:, None] * x_grid[None, :] + self.alpha * np.pi / 2.0
        cos_terms = tf.cos(phase)

        # Contract: (B, n_terms) @ (n_terms, N) -> (B, N)
        u = 4.0 * self.nu * self.k**self.alpha * (coeffs @ cos_terms)

        if original_rank == 0:
            u = tf.squeeze(u, axis=0)
        return u

    def _u_at_points(self, x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        """Evaluate the series at paired ``(x_i, t_i)`` points.

        Both ``x`` and ``t`` have shape ``(B,)``; output is ``(B,)``.
        Unlike ``_u_on_grid``, this does *not* form the full outer
        product — it computes only the B diagonal entries needed.
        """
        n = tf.range(1, self.n_terms + 1, dtype=tf.float64)  # (n_terms,)

        beta = self.b * tf.exp(-self.nu * self.k**2 * t)         # (B,)
        r = -beta / (self.a + tf.sqrt(self.a**2 - beta**2))      # (B,)

        coeffs = (r[:, None] ** n[None, :]) / n[None, :] ** (1.0 - self.alpha)  # (B, n_terms)
        phases = n[None, :] * self.k * x[:, None] + self.alpha * np.pi / 2.0   # (B, n_terms)
        cos_terms = tf.cos(phases)                                               # (B, n_terms)

        return 4.0 * self.nu * self.k**self.alpha * tf.reduce_sum(coeffs * cos_terms, axis=-1)

    # ------------------------------------------------------------------
    # Public API: hand to the solver, or use as a Solution.
    # ------------------------------------------------------------------

    def initial_condition(self) -> InitialCondition:
        """Return an ``InitialCondition`` to feed to ``SpectralSolver.solve``.

        ``u_0`` is the truncated analytic series at ``t = 0``; ``θ_0``
        is the exact closed form ``a + b cos(kx)``. The solver uses
        ``θ_0`` and skips its own spectral ``I^α`` step, so any
        discrepancy from the reference is purely from heat-evolution
        and the post-processor ``D^α ln θ``.
        """
        zero = tf.constant(0.0, dtype=tf.float64)

        def u_0(x: tf.Tensor) -> tf.Tensor:
            return self._u_on_grid(x, zero)

        def theta_0(x: tf.Tensor, nu: float, alpha: float) -> tf.Tensor:
            # Closed form: doesn't actually depend on nu, alpha.
            return self.a + self.b * tf.cos(self.k * x)

        return InitialCondition(
            name=f"cosine_mode(a={self.a}, b={self.b}, k={self.k})",
            u_0=u_0,
            theta_0=theta_0,
        )

    def reference_solution(self, grid: FourierGrid) -> Solution:
        """Return the analytic ``u(x, t)`` wrapped as a ``Solution``.

        ``sample()`` evaluates the series on the solver grid (for
        compatibility with spectral comparisons). ``__call__(x, t)``
        *calculates* directly at arbitrary ``(x, t)`` pairs via the
        closed-form series — it does **not** interpolate, so it is
        exact regardless of how coarse ``grid`` is.
        """
        x_grid = grid.x_tf

        def on_grid(t: tf.Tensor) -> tf.Tensor:
            return self._u_on_grid(x_grid, t)

        return Solution(grid, on_grid, on_arbitrary=self._u_at_points)
