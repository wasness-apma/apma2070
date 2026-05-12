"""Tests for the Cole–Hopf transform pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from fracburgers.cole_hopf import theta_to_u, u_to_theta_0
from fracburgers.grid import FourierGrid
from fracburgers.initial_conditions import gaussian, sine


def _make_grid(N: int = 256, L: float = np.pi) -> FourierGrid:
    return FourierGrid.make(N=N, L=L)


def _assert_close(actual: tf.Tensor, expected: tf.Tensor, atol: float = 1e-10) -> None:
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=0.0, atol=atol)


@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75])
def test_u_to_theta_to_u_round_trip_sine(alpha: float):
    """``u_0 -> theta_0 -> u_0`` round-trip on ``sin x`` to spectral precision."""

    grid = _make_grid(N=256, L=np.pi)
    nu = 0.12
    u_0 = tf.sin(grid.x_tf)

    theta_0 = u_to_theta_0(u_0=u_0, alpha=alpha, nu=nu, grid=grid)
    recovered = theta_to_u(theta=theta_0, alpha=alpha, nu=nu, grid=grid)

    _assert_close(recovered, u_0, atol=4e-9)


@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75])
def test_u_to_theta_to_u_round_trip_gaussian(alpha: float):
    """Same, on a Gaussian-shaped initial condition (up to constant mode)."""

    grid = _make_grid(N=512, L=12.0)
    nu = 0.1

    ic = gaussian()
    u_raw = ic.u_0(grid.x_tf)
    u_0 = u_raw - tf.reduce_mean(u_raw)

    theta_0 = u_to_theta_0(u_0=u_0, alpha=alpha, nu=nu, grid=grid)
    recovered = theta_to_u(theta=theta_0, alpha=alpha, nu=nu, grid=grid)

    _assert_close(recovered, u_0, atol=2e-8)


def test_theta_0_positive():
    """``theta_0 = exp(-I^alpha u_0 / (2nu))`` is positive for any real ``u_0``."""

    grid = _make_grid(N=192, L=np.pi)
    alpha = 0.6
    nu = 0.2

    u_0 = 0.7 * tf.sin(3.0 * grid.x_tf) - 0.25 * tf.cos(5.0 * grid.x_tf)

    theta_0 = u_to_theta_0(u_0=u_0, alpha=alpha, nu=nu, grid=grid)

    assert theta_0.dtype == tf.float64
    assert tuple(theta_0.shape) == (grid.N,)
    assert np.isfinite(theta_0.numpy()).all()
    assert np.all(theta_0.numpy() > 0.0)


def test_sine_closed_form_theta_0_matches_spectral():
    """Closed-form ``sine().theta_0`` matches spectral transform."""

    grid = _make_grid(N=256, L=np.pi)
    alpha = 0.7
    nu = 0.08

    ic = sine()
    theta_closed = ic.theta_0(grid.x_tf, nu=nu, alpha=alpha)
    theta_spec = u_to_theta_0(u_0=ic.u_0(grid.x_tf), alpha=alpha, nu=nu, grid=grid)

    _assert_close(theta_spec, theta_closed, atol=2e-10)
