"""Uniform Fourier grid on a periodic interval.

The grid is `N` equally-spaced points on `[-L, L)` with corresponding
wavenumbers `k = 2π * fftfreq(N, dx)`. All FFT-based operators in this
package assume samples live on a ``FourierGrid`` and inherit its
periodicity.

Both numpy and TF views are exposed so that the same grid object can
be used for analytic preprocessing (numpy) and for the spectral /
PINN pipelines (TF).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class FourierGrid:
    """Uniform periodic grid on ``[-L, L)`` with `N` points.

    Attributes
    ----------
    N : int
    L : float
    x : np.ndarray
        Spatial grid, shape ``(N,)``, dtype ``float64``.
    dx : float
    k : np.ndarray
        Angular wavenumbers in standard FFT order, shape ``(N,)``.
    x_tf : tf.Tensor
        TF view of ``x``, dtype ``float64``.
    k_tf : tf.Tensor
        TF view of ``k``, dtype ``float64``.
    kc_tf : tf.Tensor
        TF view of ``k`` cast to ``complex128`` for use in symbol
        construction.
    """

    N: int
    L: float
    x: np.ndarray
    dx: float
    k: np.ndarray
    x_tf: tf.Tensor
    k_tf: tf.Tensor
    kc_tf: tf.Tensor

    @classmethod
    def make(cls, N: int, L: float) -> "FourierGrid":
        """Construct a grid with `N` points on `[-L, L)`."""

        x, dx = np.linspace(-L, L, num=N, endpoint=False, retstep=True)

        k = np.fft.fftfreq(N, d=dx) * 2 * np.pi

        x_tf = tf.convert_to_tensor(x, dtype=tf.float64)
        k_tf = tf.convert_to_tensor(k, dtype=tf.float64)
        
        # 4. Complex TF View (for spectral operations)
        kc_tf = tf.cast(k_tf, dtype=tf.complex128)


        return cls(
            N = N,
            L = L,
            x = x,
            dx = dx,
            k = k,
            x_tf = x_tf,
            k_tf = k_tf,
            kc_tf = kc_tf
        )

    def __repr__(self) -> str:
        return f"FourierGrid(N={self.N}, L={self.L})"
