"""Tests for spectral operators against analytic identities (TF tensors)."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from fracburgers.grid import FourierGrid
from fracburgers.operators import (
    fractional_derivative,
    fractional_integral,
    heat_evolve,
)


def _make_grid(N: int = 128, L: float = np.pi) -> FourierGrid:
    return FourierGrid.make(N=N, L=L)


def _assert_close(actual: tf.Tensor, expected: tf.Tensor, atol: float = 1e-10) -> None:
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=0.0, atol=atol)


def test_fractional_derivative_of_constant_is_zero():
    """``D^α [c] = 0`` (the k=0 mode is killed by construction)."""

    grid = _make_grid(N=96, L=np.pi)
    u = tf.fill([grid.N], tf.constant(2.75, dtype=tf.float64))

    du = fractional_derivative(u=u, alpha=0.5, grid=grid)

    _assert_close(du, tf.zeros_like(u), atol=1e-12)


@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75])
def test_fractional_derivative_of_sine(alpha: float):
    """``D^α [sin x] = sin(x + α π / 2)`` to spectral precision."""

    grid = _make_grid(N=256, L=np.pi)
    u = tf.sin(grid.x_tf)
    expected = tf.sin(grid.x_tf + tf.constant(alpha * np.pi / 2.0, dtype=tf.float64))

    du = fractional_derivative(u=u, alpha=alpha, grid=grid)

    _assert_close(du, expected, atol=1e-10)


def test_fractional_derivative_then_integral_recovers_function():
    """``I^α [D^α [f]] = f`` for mean-zero ``f`` (k=0 mode is dropped)."""

    grid = _make_grid(N=192, L=np.pi)
    alpha = 0.6
    u = tf.sin(2.0 * grid.x_tf) + 0.35 * tf.cos(7.0 * grid.x_tf)

    recovered = fractional_integral(
        u=fractional_derivative(u=u, alpha=alpha, grid=grid),
        alpha=alpha,
        grid=grid,
    )

    _assert_close(recovered, u, atol=2e-10)


def test_heat_propagator_preserves_total_mass():
    """``∫ θ(x, t) dx = ∫ θ_0(x) dx`` (heat kernel is mass-preserving)."""

    grid = _make_grid(N=160, L=np.pi)
    nu = 0.2
    theta_0 = 1.3 + 0.2 * tf.sin(3.0 * grid.x_tf) - 0.1 * tf.cos(4.0 * grid.x_tf)
    t = tf.constant([[0.0], [0.1], [0.35], [0.8]], dtype=tf.float64)

    theta_t = heat_evolve(theta_0=theta_0, t=t, nu=nu, grid=grid)

    mass_0 = tf.reduce_sum(theta_0) * grid.dx
    masses = tf.reduce_sum(theta_t, axis=-1) * grid.dx
    expected = tf.fill(tf.shape(masses), mass_0)

    _assert_close(masses, expected, atol=1e-11)


def test_heat_propagator_at_t_zero_is_identity():
    """``θ(x, 0) = θ_0(x)`` exactly."""

    grid = _make_grid(N=96, L=np.pi)
    nu = 0.15
    theta_0 = 0.9 + 0.3 * tf.cos(2.0 * grid.x_tf) + 0.15 * tf.sin(5.0 * grid.x_tf)
    t = tf.zeros((3, 1), dtype=tf.float64)

    theta_t = heat_evolve(theta_0=theta_0, t=t, nu=nu, grid=grid)
    expected = tf.repeat(theta_0[None, :], repeats=3, axis=0)

    _assert_close(theta_t, expected, atol=1e-12)


@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75])
def test_fractional_derivative_real_input_real_output(alpha: float):
    """Real input → real output (imag part is float64 round-off)."""

    grid = _make_grid(N=128, L=np.pi)
    u = tf.stack(
        [
            tf.sin(grid.x_tf),
            0.5 * tf.cos(3.0 * grid.x_tf),
            tf.sin(2.0 * grid.x_tf) - 0.25 * tf.cos(5.0 * grid.x_tf),
        ],
        axis=0,
    )

    du = fractional_derivative(u=u, alpha=alpha, grid=grid)

    assert du.dtype == tf.float64
    assert tuple(du.shape) == tuple(u.shape)
    assert np.isrealobj(du.numpy())
    assert np.isfinite(du.numpy()).all()


def test_operators_are_differentiable_via_autograd():
    """``tf.GradientTape`` through ``fractional_derivative`` yields the
    expected linear operator (the operator equals its own Jacobian)."""

    grid = _make_grid(N=32, L=np.pi)
    alpha = 0.4

    u = tf.Variable(tf.sin(2.0 * grid.x_tf) + 0.2 * tf.cos(3.0 * grid.x_tf))
    direction = tf.sin(5.0 * grid.x_tf) - 0.4 * tf.cos(2.0 * grid.x_tf)

    with tf.GradientTape() as tape:
        du = fractional_derivative(u=u, alpha=alpha, grid=grid)
    jac = tape.jacobian(du, u)

    jvp_from_jacobian = tf.linalg.matvec(jac, direction)
    expected_jvp = fractional_derivative(u=direction, alpha=alpha, grid=grid)

    _assert_close(jvp_from_jacobian, expected_jvp, atol=1e-10)
