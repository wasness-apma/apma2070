"""Tests for trigonometric interpolation."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from fracburgers.grid import FourierGrid
from fracburgers.interpolation import trig_interp


def _make_grid(N: int = 64, L: float = np.pi) -> FourierGrid:
    return FourierGrid.make(N=N, L=L)


def _assert_close(actual: tf.Tensor, expected: tf.Tensor, atol: float = 1e-11) -> None:
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=0.0, atol=atol)


def test_interp_at_grid_points_recovers_samples():
    """Querying ``trig_interp`` at the grid points returns the input
    samples to machine precision."""

    grid = _make_grid(N=96, L=np.pi)
    u_grid = 0.3 + tf.sin(3.0 * grid.x_tf) - 0.5 * tf.cos(11.0 * grid.x_tf)

    u_interp = trig_interp(u_grid=u_grid, x=grid.x_tf, grid=grid)

    _assert_close(u_interp, u_grid, atol=1e-12)


def test_interp_of_low_frequency_sinusoid_is_exact():
    """For a single sinusoid below the Nyquist frequency, interpolation
    is exact at any query point."""

    grid = _make_grid(N=128, L=np.pi)
    mode = 7.0

    u_grid = tf.sin(mode * grid.x_tf)
    x_query = tf.linspace(
        tf.constant(-grid.L + 0.07, dtype=tf.float64),
        tf.constant(grid.L - 0.07, dtype=tf.float64),
        257,
    )
    u_expected = tf.sin(mode * x_query)

    u_interp = trig_interp(u_grid=u_grid, x=x_query, grid=grid)

    _assert_close(u_interp, u_expected, atol=1e-11)


def test_interp_is_differentiable():
    """``tf.GradientTape`` through ``trig_interp`` w.r.t. query points
    yields the spectral derivative (matches multiplication by ``ik``)."""

    grid = _make_grid(N=128, L=np.pi)
    u_grid = tf.sin(2.0 * grid.x_tf) + 0.75 * tf.cos(5.0 * grid.x_tf)
    x_query = tf.Variable(
        tf.linspace(
            tf.constant(-grid.L + 0.1, dtype=tf.float64),
            tf.constant(grid.L - 0.1, dtype=tf.float64),
            211,
        )
    )

    with tf.GradientTape() as tape:
        u_interp = trig_interp(u_grid=u_grid, x=x_query, grid=grid)
    du_dx_tape = tape.gradient(u_interp, x_query)

    u_hat = tf.signal.fft(tf.cast(u_grid, dtype=tf.complex128))
    ik_u_hat = (1j * grid.kc_tf) * u_hat
    phases = grid.kc_tf[:, None] * (
        tf.cast(x_query, dtype=tf.complex128)[None, :]
        + tf.cast(grid.L, dtype=tf.complex128)
    )
    du_dx_expected = tf.math.real(
        tf.reduce_sum(ik_u_hat[:, None] * tf.exp(1j * phases), axis=0)
    ) / tf.cast(grid.N, dtype=tf.float64)

    _assert_close(du_dx_tape, du_dx_expected, atol=1e-5)


@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 5)])
def test_interp_broadcasts(batch_shape: tuple):
    """``u_grid`` of shape ``(*batch, N)`` and ``x`` of arbitrary shape
    yield broadcast-compatible output."""

    grid = _make_grid(N=64, L=np.pi)
    mode = 4.0
    x_query = tf.constant(
        [[-2.1, -0.4, 0.6, 1.3], [2.0, -1.7, 0.2, -0.9]], dtype=tf.float64
    )

    base = tf.sin(mode * grid.x_tf)
    base_query = tf.sin(mode * x_query)

    if batch_shape:
        num_batches = int(np.prod(batch_shape))
        amplitudes = tf.constant(
            np.arange(1, num_batches + 1, dtype=np.float64).reshape(batch_shape)
        )
        u_grid = amplitudes[..., None] * base
        u_expected = amplitudes[..., None, None] * base_query
    else:
        u_grid = base
        u_expected = base_query

    u_interp = trig_interp(u_grid=u_grid, x=x_query, grid=grid)

    assert tuple(u_interp.shape) == tuple(u_expected.shape)
    _assert_close(u_interp, u_expected, atol=1e-11)
