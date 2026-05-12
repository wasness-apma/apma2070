"""Trigonometric interpolation on a periodic Fourier grid.

Given a real field sampled at the ``N`` grid points of a
``FourierGrid``, the trigonometric interpolant

    u(x) = (1/N) Σ_k û_k exp(i k (x + L))

is the unique band-limited periodic extension that agrees with the
samples at the grid points. It coincides with the spectral
representation used by all FFT-based operators in the package, so
``Solution.__call__`` is a self-consistent extension of
``Solution.sample``.

Implemented in TF for autograd compatibility.
"""

from __future__ import annotations

import tensorflow as tf

from fracburgers.grid import FourierGrid


def trig_interp(u_grid: tf.Tensor, x: tf.Tensor, grid: FourierGrid) -> tf.Tensor:
    """Trig-interpolate ``u_grid`` (sampled on ``grid.x``) at points ``x``.

    Parameters
    ----------
    u_grid : tf.Tensor
        Real samples of shape ``(N,)`` or ``(T, N)``.
    x : tf.Tensor
        Query points, arbitrary shape.
    grid : FourierGrid

    Returns
    -------
    tf.Tensor
        Interpolated values. Shape is the broadcast of the leading
        axes of ``u_grid`` against the shape of ``x``.

    Notes
    -----
    Cost is ``O(N · M)`` for ``M`` query points. For repeated queries
    on a large set of points consider precomputing the FFT and
    reusing the spectral coefficients.
    """

    u_hat: tf.Tensor = tf.signal.fft(tf.cast(u_grid, dtype=tf.complex128))

    N: tf.Tensor = tf.constant(grid.N, dtype=tf.float64)
    L: tf.Tensor = tf.constant(grid.L, dtype=tf.float64)
    k_t: tf.Tensor = grid.k_tf

    # Build Fourier basis on a flattened query axis, then restore x-shape.
    x_shape = tf.shape(x)
    x_flat = tf.reshape(x, [-1])
    phases = k_t[:, None] * (x_flat[None, :] + L)  # (N, M), real-valued
    basis = tf.exp(tf.complex(tf.zeros_like(phases), phases))  # (N, M)

    # u_hat has shape (..., N); output has shape (..., M).
    u_complex = tf.einsum("...n,nm->...m", u_hat, basis)
    u_real = tf.math.real(u_complex) / N

    out_shape = tf.concat([tf.shape(u_grid)[:-1], x_shape], axis=0)
    return tf.reshape(u_real, out_shape)
