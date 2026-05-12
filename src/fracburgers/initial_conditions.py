"""Preset initial conditions, exposed as TF callables.

Each preset is an ``InitialCondition`` with two pieces:

    .u_0(x)                       always defined; returns u_0(x)
    .theta_0(x, nu, alpha)        defined when a closed form for
                                  θ_0 = exp(-I^α u_0 / (2ν)) is known,
                                  otherwise None

If ``theta_0`` is provided, ``SpectralSolver.solve`` skips the spectral
``I^α`` step and uses the closed form, which removes one source of
spectral error from the validation pipeline.

All callables take and return ``tf.Tensor`` of dtype ``float64``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import tensorflow as tf


UFunc = Callable[[tf.Tensor], tf.Tensor]
ThetaFunc = Callable[[tf.Tensor, float, float], tf.Tensor]


@dataclass(frozen=True)
class InitialCondition:
    """Bundle of an initial condition with its (optional) closed-form θ_0."""

    name: str
    u_0: UFunc
    theta_0: ThetaFunc | None = None


def sine() -> InitialCondition:
    """``u_0(x) = sin x``.

    Closed form: ``I^α [sin x] = sin(x - α π / 2)``, so
    ``θ_0(x) = exp(-sin(x - α π / 2) / (2ν))``.

    For periodicity make sure the grid's ``L`` is an integer multiple
    of ``π``.
    """
    return InitialCondition(
        name = "sine",
        u_0 = tf.sin,
        theta_0 = lambda x, nu, alpha: tf.exp(-tf.sin(x - alpha * np.pi / 2) / (2 * nu))
    )


def gaussian() -> InitialCondition:
    """``u_0(x) = exp(-x²)``.

    No elementary closed form for ``I^α``, so ``theta_0`` is None and
    the spectral pipeline computes it.
    """
    return InitialCondition(
        name = "gaussian",
        u_0 = lambda x: tf.exp(-x**2),
        theta_0 = None
    )


REGISTRY: dict[str, Callable[[], InitialCondition]] = {
    "sine": sine,
    "gaussian": gaussian,
}


def get(name: str) -> InitialCondition:
    """Look up a preset by name; used by the CLI scripts."""
    try:
        return REGISTRY[name]()
    except KeyError as exc:
        valid = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unknown initial condition {name!r}. Choose one of: {valid}.") from exc
