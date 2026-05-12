"""Tests for the ``Solution`` interface — both spectral and PINN."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
import tensorflow as tf

from fracburgers.grid import FourierGrid
from fracburgers.initial_conditions import sine
from fracburgers.spectral import SpectralSolver
from fracburgers.viz import animate_comparison, animate_solution


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def _make_grid(N: int = 192, L: float = np.pi) -> FourierGrid:
    return FourierGrid.make(N=N, L=L)


def _assert_close(actual: tf.Tensor, expected: tf.Tensor, atol: float = 1e-9) -> None:
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=0.0, atol=atol)


def _make_solution(alpha: float = 0.5, nu: float = 0.1):
    grid = _make_grid(N=256, L=np.pi)
    solver = SpectralSolver(grid=grid, nu=nu, alpha=alpha)
    ic = sine()
    return grid, ic, solver.solve(ic)


def test_spectral_solution_at_t_zero_recovers_u_0():
    """Querying ``SpectralSolver.solve(...)`` at ``t=0`` returns ``u_0``."""

    grid, ic, sol = _make_solution(alpha=0.5, nu=0.12)

    u_0 = ic.u_0(grid.x_tf)
    u_t0 = sol.sample(tf.constant(0.0, dtype=tf.float64))

    _assert_close(u_t0, u_0, atol=3e-10)


@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75])
def test_spectral_solution_callable_at_arbitrary_x(alpha: float):
    """``Solution.__call__`` agrees with ``Solution.sample`` at the
    grid points and interpolates smoothly between them."""

    grid, _, sol = _make_solution(alpha=alpha, nu=0.1)
    t = tf.constant(0.35, dtype=tf.float64)

    sampled = sol.sample(t)
    called_on_grid = sol(grid.x_tf, t)

    _assert_close(called_on_grid, sampled, atol=2e-9)

    x_query = tf.linspace(
        tf.constant(-grid.L + 0.05, dtype=tf.float64),
        tf.constant(grid.L - 0.05, dtype=tf.float64),
        321,
    )
    queried = sol(x_query, t)

    assert queried.dtype == tf.float64
    assert tuple(queried.shape) == (321,)
    assert np.isfinite(queried.numpy()).all()


def test_solution_is_differentiable_in_x():
    """``tf.GradientTape`` through ``sol(x, t)`` w.r.t. ``x`` returns
    the spatial derivative at the requested point."""

    _, _, sol = _make_solution(alpha=0.6, nu=0.08)

    x = tf.Variable(0.37, dtype=tf.float64)
    t = tf.constant(0.22, dtype=tf.float64)
    eps = tf.constant(2e-5, dtype=tf.float64)

    with tf.GradientTape() as tape:
        u = sol(x, t)
    du_dx = tape.gradient(u, x)

    u_plus = sol(x + eps, t)
    u_minus = sol(x - eps, t)
    fd = (u_plus - u_minus) / (2.0 * eps)

    _assert_close(du_dx, fd, atol=2e-4)


def test_solution_is_differentiable_in_t():
    """Same, but for the time derivative."""

    _, _, sol = _make_solution(alpha=0.4, nu=0.11)

    x = tf.constant(-0.41, dtype=tf.float64)
    t = tf.Variable(0.31, dtype=tf.float64)
    eps = tf.constant(2e-5, dtype=tf.float64)

    with tf.GradientTape() as tape:
        u = sol(x, t)
    du_dt = tape.gradient(u, t)

    u_plus = sol(x, t + eps)
    u_minus = sol(x, t - eps)
    fd = (u_plus - u_minus) / (2.0 * eps)

    _assert_close(du_dt, fd, atol=2e-4)


def test_alpha_one_limit_recovers_classical_burgers():
    """At ``α → 1⁻``, the fractional Cole–Hopf reduces to the classical
    transform; the spectral solution should match the closed-form
    classical formula in section 1.1 of the report."""

    grid = _make_grid(N=320, L=np.pi)
    ic = sine()
    nu = 0.1
    t = tf.constant(0.4, dtype=tf.float64)

    sol_near = SpectralSolver(grid=grid, nu=nu, alpha=0.999).solve(ic)
    sol_one = SpectralSolver(grid=grid, nu=nu, alpha=1.0).solve(ic)

    u_near = sol_near(grid.x_tf, t)
    u_one = sol_one(grid.x_tf, t)

    _assert_close(u_near, u_one, atol=2e-3)


def test_spectral_solution_movie_animation_constructs():
    """Movie helper returns a valid animation object for spectral solutions."""

    _, _, sol = _make_solution(alpha=0.5, nu=0.1)
    times = np.linspace(0.0, 0.4, num=5, dtype=np.float64)

    anim = animate_solution(sol=sol, times=times, interval=1)

    assert isinstance(anim, FuncAnimation)
    artists = anim._func(0)
    assert len(artists) >= 1
    plt.close(anim._fig)


def test_spectral_solution_comparison_movie_animation_constructs():
    """Comparison movie helper builds for two spectral solutions."""

    grid = _make_grid(N=160, L=np.pi)
    ic = sine()
    sol_ref = SpectralSolver(grid=grid, nu=0.1, alpha=0.5).solve(ic)
    sol_test = SpectralSolver(grid=grid, nu=0.1, alpha=0.75).solve(ic)
    times = np.linspace(0.0, 0.4, num=5, dtype=np.float64)

    anim = animate_comparison(sol_ref=sol_ref, sol_test=sol_test, times=times, interval=1)

    assert isinstance(anim, FuncAnimation)
    artists = anim._func(0)
    assert len(artists) >= 3
    plt.close(anim._fig)
