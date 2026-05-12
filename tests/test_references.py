"""Tests for ``CosineModeReference`` and the spectral solver against it."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from fracburgers.grid import FourierGrid
from fracburgers.references import CosineModeReference


# ----------------------------------------------------------------------
# Reference itself: sanity checks that don't require the solver.
# ----------------------------------------------------------------------

def test_cosine_mode_reference_at_t_zero_is_finite():
    """Series at t=0 evaluates without overflow/NaN."""
    grid = FourierGrid.make(N=128, L=np.pi)
    ref = CosineModeReference(a=2.0, b=1.0, k=1.0, nu=0.1, alpha=0.5)
    u_0 = ref.reference_solution(grid).sample(tf.constant(0.0, dtype=tf.float64))
    assert tf.reduce_all(tf.math.is_finite(u_0))


def test_cosine_mode_reference_decays_to_zero_at_late_time():
    """As t → ∞, r(t) → 0 and u → 0."""
    grid = FourierGrid.make(N=128, L=np.pi)
    ref = CosineModeReference(a=2.0, b=1.0, k=1.0, nu=1.0, alpha=0.5)
    u_late = ref.reference_solution(grid).sample(tf.constant(20.0, dtype=tf.float64))
    assert tf.reduce_max(tf.abs(u_late)) < 1e-8


def test_cosine_mode_initial_condition_consistent_with_reference():
    """``ic.u_0(x_grid)`` matches ``reference_solution(grid).sample(0)``."""
    grid = FourierGrid.make(N=128, L=np.pi)
    ref = CosineModeReference(a=2.0, b=1.0, k=1.0, nu=0.1, alpha=0.5)
    ic = ref.initial_condition()
    sol = ref.reference_solution(grid)

    u_from_ic = ic.u_0(grid.x_tf)
    u_from_ref = sol.sample(tf.constant(0.0, dtype=tf.float64))
    np.testing.assert_allclose(u_from_ic.numpy(), u_from_ref.numpy(), atol=1e-14)


def test_cosine_mode_theta_0_matches_a_plus_b_cos():
    """``ic.theta_0`` is the closed form ``a + b cos(kx)``."""
    grid = FourierGrid.make(N=128, L=np.pi)
    ref = CosineModeReference(a=2.0, b=1.0, k=1.0, nu=0.1, alpha=0.5)
    theta_0 = ref.initial_condition().theta_0(grid.x_tf, 0.1, 0.5)
    expected = 2.0 + 1.0 * tf.cos(1.0 * grid.x_tf)
    np.testing.assert_allclose(theta_0.numpy(), expected.numpy(), atol=1e-14)


# ----------------------------------------------------------------------
# Spectral solver against the reference (requires SpectralSolver impl).
# ----------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75])
def test_spectral_solver_matches_cosine_mode_reference(alpha: float):
    """Spectral solver agrees with the analytic series across α and t.

    The grid is sized so several harmonics of ``k`` are well-resolved
    (N/2 ≫ n_terms · k · L / π), and ``b/a = 1/2`` keeps ``|r| ≈ 0.27``,
    so 80 terms gives effectively-zero truncation error.
    """
    from fracburgers.spectral import SpectralSolver

    grid = FourierGrid.make(N=512, L=np.pi)
    nu = 0.1
    ref = CosineModeReference(a=2.0, b=1.0, k=1.0, nu=nu, alpha=alpha,
                               n_terms=80)

    solver = SpectralSolver(grid=grid, nu=nu, alpha=alpha)
    sol_spec = solver.solve(ref.initial_condition())
    sol_ref = ref.reference_solution(grid)

    for t_test in [0.0, 0.1, 0.5, 1.0]:
        t_tf = tf.constant(t_test, dtype=tf.float64)
        u_spec = sol_spec.sample(t_tf).numpy()
        u_ref = sol_ref.sample(t_tf).numpy()
        max_err = np.max(np.abs(u_spec - u_ref))
        assert max_err < 1e-10, (
            f"α={alpha}, t={t_test}: max |u_spec - u_ref| = {max_err:.3e}"
        )
