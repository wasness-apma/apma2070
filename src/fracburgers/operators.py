"""FFT-based spectral operators on a ``FourierGrid``, in TensorFlow.

All operators here are diagonal in Fourier space and applied as
``F^{-1}[m(k) * F[u]]`` for some symbol ``m(k)``:

    fractional derivative      D^α     m(k) = (ik)^α
    fractional integral        I^α     m(k) = (ik)^{-α}
    heat propagator (time t)            m(k) = exp(-ν k² t)

The ``k = 0`` mode of fractional symbols is set to 0.

All inputs and outputs are ``tf.Tensor`` with dtype ``float64`` for
real-valued fields and ``complex128`` for spectral coefficients.
Inputs may carry leading batch dimensions; the FFT acts on the last
axis (the spatial axis on the grid).
"""

from __future__ import annotations

import tensorflow as tf

from fracburgers.grid import FourierGrid


def fractional_symbol(grid: FourierGrid, alpha: float) -> tf.Tensor:
    """Return ``(ik)^α`` on the grid as a ``complex128`` tensor.

    Uses the principal branch. The ``k=0`` entry is set to 0 so the
    symbol is well-defined for any real ``alpha``. The result is
    conjugate-symmetric in ``k`` so that real input gives real output
    after IFFT.
    """

    kc = grid.kc_tf
    one_c = tf.constant(1.0 + 0.0j, dtype=tf.complex128)
    zero_c = tf.constant(0.0 + 0.0j, dtype=tf.complex128)

    # Avoid evaluating 0**negative (inf) at k=0; restore that mode to 0 after.
    nonzero = tf.not_equal(kc, zero_c)
    safe_kc = tf.where(nonzero, kc, one_c)
    symbol = tf.pow(tf.constant(1j, dtype=tf.complex128) * safe_kc, alpha)
    return tf.where(nonzero, symbol, zero_c)


def heat_symbol(grid: FourierGrid, t: tf.Tensor, nu: float) -> tf.Tensor:
    """Return ``exp(-ν k² t)`` as a ``complex128`` tensor.

    ``t`` may be a scalar or shape ``(T, 1)`` for batched evolution;
    the result then has shape ``(T, N)``.
    """
    
    kc = grid.kc_tf
    t_c = tf.cast(t, tf.complex128)
    nu_c = tf.cast(nu, tf.complex128)
    return tf.exp(-nu_c * kc**2 * t_c)


def apply_symbol(u: tf.Tensor, symbol: tf.Tensor) -> tf.Tensor:
    """Apply a Fourier multiplier: ``F^{-1}[symbol * F[u]]``.

    Returns ``tf.math.real`` of the IFFT (any imaginary component is
    rounding noise from numerical conjugate-asymmetry). FFT acts on
    the last axis; ``symbol`` must broadcast against ``F[u]`` along
    that axis.
    """

    u_hat = tf.signal.fft(tf.cast(u, dtype=tf.complex128))
    u_hat_symbol = u_hat * symbol
    u_ifft = tf.signal.ifft(u_hat_symbol)
    return tf.math.real(u_ifft)


def fractional_derivative(u: tf.Tensor, alpha: float,
                          grid: FourierGrid) -> tf.Tensor:
    """Compute ``D^α u`` for ``α > 0`` via the symbol ``(ik)^α``."""
    symbol = fractional_symbol(grid, alpha)
    return apply_symbol(u, symbol)


def fractional_integral(u: tf.Tensor, alpha: float,
                        grid: FourierGrid) -> tf.Tensor:
    """Compute ``I^α u`` (the order-α antiderivative) via ``(ik)^{-α}``."""
    u_c = tf.cast(u, dtype=tf.complex128)
    u_hat = tf.signal.fft(u_c)

    # I^alpha is undefined for nonzero mean (k=0 mode). Mark those entries NaN.
    k0_mag = tf.abs(u_hat[..., 0])
    tol = tf.constant(1e-12, dtype=tf.float64)
    bad_zero_mode = k0_mag > tol

    symbol = fractional_symbol(grid, -alpha)
    u_ifft = tf.signal.ifft(u_hat * symbol)
    u_real = tf.math.real(u_ifft)

    nan_out = tf.fill(tf.shape(u_real), tf.constant(float("nan"), dtype=tf.float64))
    return tf.where(bad_zero_mode[..., None], nan_out, u_real)


def heat_evolve(theta_0: tf.Tensor, t: tf.Tensor, nu: float,
                grid: FourierGrid) -> tf.Tensor:
    """Evolve initial data ``theta_0`` under the heat equation to time(s) `t`.

    Returns ``θ(x, t)`` on the grid; shape is ``(N,)`` for scalar `t`
    or ``(T, N)`` if ``t`` has shape ``(T, 1)``.
    """
    symbol = heat_symbol(grid, t, nu)
    return apply_symbol(theta_0, symbol)
